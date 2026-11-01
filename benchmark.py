"""Matched offline UNSW benchmark; never replaces deployment artifacts or event data."""
import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import config
from src.gwo.feature_selector import GWOFeatureSelector
from src.models.model_factory import get_model_factory
from src.preprocessing.unsw_preprocessing import (
    bucket_rare_protocols, clean_unsw, encode_categoricals, load_unsw_raw, scale_numeric,
)

LOG = logging.getLogger("benchmark")
ROOT = Path(__file__).resolve().parent


def fingerprints(frame, columns):
    # No label or row ID: conflicting labels cannot disguise overlapping inputs.
    return pd.util.hash_pandas_object(frame[columns], index=False)


def bounded_sample(frame, limit, seed):
    if len(frame) <= limit:
        return frame.copy()
    return train_test_split(frame, train_size=limit, stratify=frame.label,
                            random_state=seed)[0].copy()


def prepare_splits(train, test, *, fit_limit, validation_limit, seed):
    """Share rows across both feature pools; keep matching inputs out of holdout."""
    raw_counts = {"train": len(train), "test": len(test)}
    train, test = clean_unsw(train), clean_unsw(test)
    feature_names = [c for c in train if c not in {"label", "attack_cat", "id"}]
    expected = set(config.RUNTIME_FEASIBLE_FEATURES + config.UNSW_EXCLUDED_FEATURES)
    if set(feature_names) != expected or not expected.issubset(test.columns):
        raise ValueError("Expected the supplied UNSW schema with 42 predictive features.")
    for frame in (train, test):
        if set(frame.label.unique()) != {0, 1}:
            raise ValueError("Binary normal/attack labels are required in both splits.")
    cleaned_counts = {"train": len(train), "test": len(test)}
    common = config.RUNTIME_FEASIBLE_FEATURES
    train_keys = fingerprints(train, common)
    overlap = fingerprints(test, common).isin(set(train_keys))
    test = test.loc[~overlap].copy()
    if set(test.label.unique()) != {0, 1}:
        raise ValueError("Independent test set must retain both classes.")
    # Group before sampling: duplicate 35-feature inputs stay on one side.
    fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2,
        random_state=seed).split(train, groups=train_keys))
    fit = bounded_sample(train.iloc[fit_idx], fit_limit, seed)
    validation = bounded_sample(train.iloc[val_idx], validation_limit, seed)
    if set(fingerprints(fit, common)) & set(fingerprints(validation, common)):
        raise ValueError("Selection fit/validation inputs overlap.")
    metadata = {
        "raw_rows": raw_counts, "cleaned_rows": cleaned_counts,
        "removed_test_overlaps_on_35_features": int(overlap.sum()),
        "final_train_rows": len(train), "final_test_rows": len(test),
        "selection_fit_rows": len(fit), "selection_validation_rows": len(validation),
        "selection_fit_class_counts": {str(k): int(v) for k, v in fit.label.value_counts().items()},
        "selection_validation_class_counts": {str(k): int(v) for k, v in validation.label.value_counts().items()},
        "overlap_policy": "35-feature input hashes, excluding labels; shared across both pools",
        "selection_split": "grouped 80/20 training split, then bounded stratified samples",
        "row_index_sha256": {name: hashlib.sha256(np.asarray(frame.index, dtype="<i8").tobytes()).hexdigest()
            for name, frame in [("fit", fit), ("validation", validation), ("test", test)]},
    }
    return train, test, fit, validation, feature_names, metadata


def transform_pair(fit, other, features):
    """Reuse existing training-only preprocessing, without saving any model files."""
    a, b, _ = bucket_rare_protocols(fit[features], other[features],
                                   threshold=config.UNSW_PROTO_RARE_THRESHOLD)
    a, b, _ = encode_categoricals(a, b, config.UNSW_CATEGORICAL_COLUMNS)
    a, b, _ = scale_numeric(a, b, [c for c in features if c not in config.UNSW_CATEGORICAL_COLUMNS])
    return a.to_numpy(dtype=float), b.to_numpy(dtype=float)


def select_features(X_fit, y_fit, X_validation, y_validation, factory,
                    *, seeds, wolves, iterations):
    """Deliberately has no final-test input. Keep all seeds, not a test-selected winner."""
    records = []
    for seed in seeds:
        started = time.perf_counter()
        selector = GWOFeatureSelector(factory, X_fit, y_fit, X_validation, y_validation,
            list(range(X_fit.shape[1])), alpha=config.FITNESS_ALPHA, beta=config.FITNESS_BETA)
        best = selector.run(config.UNSW_GWO_VARIANT, wolves, iterations,
                            config.GWO_LOWER_BOUND, config.GWO_UPPER_BOUND, seed)
        records.append({"seed": seed, "mask": list(map(int, best.features)),
            "validation_accuracy_pct": float(best.score), "fitness": float(best.fitness),
            "unique_masks_evaluated": len(selector._cache),
            "selection_seconds": round(time.perf_counter() - started, 3)})
        LOG.info("Seed %s: selected %s/%s, validation %.3f%%", seed,
                 sum(best.features), X_fit.shape[1], best.score)
    return records


def evaluate_model(model, X, y):
    started = time.perf_counter()
    prediction = model.predict(X)
    predict_seconds = time.perf_counter() - started
    probabilities = model.predict_proba(X)[:, list(model.classes_).index(1)]
    report = classification_report(y, prediction, labels=[0, 1],
                                   target_names=["normal", "attack"], output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    return {"accuracy_pct": float(np.mean(prediction == y) * 100),
        "attack_precision": report["attack"]["precision"],
        "attack_recall": report["attack"]["recall"], "attack_f1": report["attack"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "false_positive_rate_pct": 100 * fp / (tn + fp),
        "false_negative_rate_pct": 100 * fn / (tp + fn),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "predict_seconds": round(predict_seconds, 3),
        "confusion_matrix": matrix.tolist(), "classification_report": report}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_benchmark(args):
    # Never overwrite a completed experiment, model directory or user's database.
    output = Path(args.output).resolve()
    protected = [Path(config.SAVED_MODELS_DIR).resolve(), Path(config.ALERTS_DB_PATH).resolve().parent]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError("Benchmark output must be separate from deployment models and event data.")
    output.mkdir(parents=True, exist_ok=False)
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    started = time.perf_counter()
    try:
        source_files = [Path(config.UNSW_TRAIN_PATH), Path(config.UNSW_TEST_PATH)]
        code_files = [Path(__file__), ROOT / "config.py", *sorted((ROOT / "src").rglob("*.py"))]
        protocol = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
            "argv": sys.argv[1:], "split_seed": args.split_seed,
            "gwo_seeds": args.seeds, "wolves": args.wolves, "iterations": args.iterations,
            "variant": config.UNSW_GWO_VARIANT,
            "fitness": {"formula": "alpha * accuracy_percent + beta / selected_count",
                        "alpha": config.FITNESS_ALPHA, "beta": config.FITNESS_BETA},
            "random_forest": {**config.RANDOM_FOREST_PARAMS, "n_jobs": args.jobs},
            "selection_fit_limit": args.fit_limit, "selection_validation_limit": args.validation_limit,
            "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
            "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
            "limitations": ["Matched comparison within this repository; not an exact reproduction of the friend's unknown code/settings.",
                "GWO selection uses a bounded training subset. Final models fit all cleaned training rows.",
                "Seeds vary the optimizer only; RF seed and data split stay fixed. Not repeated independent datasets.",
                "Previously inspected benchmark holdout, not new external validation or live-network accuracy.",
                "No selection of a winning seed or deployment promotion based on final-test results."],
        }
        write_json(output / "protocol.json", protocol)  # Freeze settings before evaluation.
        train, test, fit, validation, features, split = prepare_splits(
            load_unsw_raw(config.UNSW_TRAIN_PATH), load_unsw_raw(config.UNSW_TEST_PATH),
            fit_limit=args.fit_limit, validation_limit=args.validation_limit, seed=args.split_seed)
        write_json(output / "splits.json", split)
        LOG.info("Split counts: %s", split)
        X_fit, X_val = transform_pair(fit, validation, features)
        base_factory = get_model_factory("random_forest")
        factory = lambda: base_factory().set_params(n_jobs=args.jobs)
        pools = {"offline_42": features, "runtime_35": list(config.RUNTIME_FEASIBLE_FEATURES)}
        selected = {}
        # Freeze every selected mask before opening final-test performance results.
        for pool, names in pools.items():
            indices = [features.index(c) for c in names]
            LOG.info("Starting selection: %s", pool)
            selected[pool] = select_features(X_fit[:, indices], fit.label.to_numpy(),
                X_val[:, indices], validation.label.to_numpy(), factory,
                seeds=args.seeds, wolves=args.wolves, iterations=args.iterations)
            write_json(output / "selection.json", selected)
        X_train, X_test = transform_pair(train, test, features)
        results = []
        for pool, names in pools.items():
            candidates = [{"seed": None, "mask": [1] * len(names), "selection_seconds": 0}, *selected[pool]]
            for candidate in candidates:
                chosen = [name for name, bit in zip(names, candidate["mask"]) if bit]
                indices = [features.index(c) for c in chosen]
                LOG.info("Final fitting %s / seed %s / %s features", pool, candidate["seed"], len(chosen))
                model = factory()
                fit_started = time.perf_counter()
                model.fit(X_train[:, indices], train.label.to_numpy())
                fit_seconds = time.perf_counter() - fit_started
                result = {"pool": pool, "method": "baseline" if candidate["seed"] is None else "bGWO",
                    "seed": candidate["seed"], "feature_count": len(chosen), "candidate_features": len(names),
                    "feature_reduction_pct": 100 * (1 - len(chosen) / len(names)),
                    "features": chosen, "fit_seconds": round(fit_seconds, 3),
                    "selection_seconds": candidate["selection_seconds"],
                    **evaluate_model(model, X_test[:, indices], test.label.to_numpy())}
                results.append(result)
                write_json(output / "results.partial.json", results)
                LOG.info("Measured: %s", {k: result[k] for k in ["pool", "method", "seed", "feature_count", "accuracy_pct", "attack_f1"]})
        flat = pd.DataFrame([{k: v for k, v in row.items()
                             if k not in {"features", "confusion_matrix", "classification_report"}} for row in results])
        flat.to_csv(output / "comparison.csv", index=False)
        plot_comparison(results, output)
        report = {"protocol": protocol, "splits": split, "results": results,
                  "duration_seconds": round(time.perf_counter() - started, 3), "status": "completed"}
        write_json(output / "results.json", report)
        (output / "results.partial.json").unlink()
        LOG.info("Completed benchmark in %.1f seconds: %s", report["duration_seconds"], output)
        return report
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def plot_comparison(results, output):
    labels = [f"{r['pool']}\n{r['method']}" + (f" {r['seed']}" if r["seed"] is not None else "") for r in results]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for ax, key, title, maximum in [(axes[0], "accuracy_pct", "Held-out accuracy (%)", 100),
                                   (axes[1], "feature_count", "Selected feature count", 45)]:
        bars = ax.bar(labels, [r[key] for r in results], color=["#8797a8" if r["method"] == "baseline" else "#087c80" for r in results])
        ax.bar_label(bars, fmt="%.2f" if key == "accuracy_pct" else "%.0f", padding=3)
        ax.set_ylim(0, maximum); ax.set_title(title); ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.suptitle("UNSW-NB15 matched offline benchmark — fixed split; optimizer seeds shown")
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/benchmarks/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--fit-limit", type=int, default=4800)
    parser.add_argument("--validation-limit", type=int, default=1200)
    parser.add_argument("--wolves", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.fit_limit, args.validation_limit) < 10 or args.wolves < 3 or args.iterations < 1 or args.jobs < 1:
        parser.error("Require sample limits >= 10, wolves >= 3, iterations >= 1 and jobs >= 1.")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in [*args.seeds, args.split_seed]):
        parser.error("Use unique seeds and unsigned 32-bit integer seed values.")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    args=parse_args()
    run_benchmark(args)
    from evidence import write_pack
    print(write_pack(Path(args.output)))
