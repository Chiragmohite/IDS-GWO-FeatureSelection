"""
train.py

Offline UNSW-NB15 training pipeline:

    UNSW-NB15
      -> Dataset Loading
      -> Data Validation
      -> Cleaning
      -> Restrict to runtime-feasible feature space (see docs/RUNTIME_FEATURE_AUDIT.md)
      -> Categorical Encoding (rare-protocol bucketing + LabelEncoder)
      -> Feature Scaling (StandardScaler)
      -> Grey Wolf Optimization (reuses the corrected src/gwo implementation)
      -> Feature Selection
      -> Machine Learning Model (REUSES src/models unchanged)
      -> Evaluation (incl. FPR/FNR, ROC-AUC)
      -> Save Model Artifacts (for independent runtime inference)

This trains the DEPLOYMENT model: GWO only ever sees the runtime-feasible
feature set (config.RUNTIME_FEASIBLE_FEATURES), so the offline model and
the real-time VM model are guaranteed to use an identical, honest feature
schema. See docs/RUNTIME_FEATURE_AUDIT.md for why each excluded feature
was excluded.

Run with:  python train.py
"""

import argparse
import hashlib
import time
import sklearn
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import config
from src.evaluation.comparison import build_comparison_table, save_comparison_table
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import (
    compute_accuracy,
    compute_binary_confusion_counts,
    compute_classification_report,
)
from src.gwo.feature_selector import GWOFeatureSelector
from src.models.model_factory import get_model_factory
from src.preprocessing.unsw_preprocessing import (
    bucket_rare_protocols,
    clean_unsw,
    encode_categoricals,
    encode_target,
    load_unsw_raw,
    save_preprocessing_artifacts,
    scale_numeric,
)
from src.preprocessing.unsw_raw_loader import load_unsw_full_raw
from src.utils.helper import ensure_directories, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("train")


def _run_training(quick: bool = False) -> None:
    started = time.perf_counter()
    np.random.seed(config.RANDOM_SEED)
    ensure_directories(config.ALL_UNSW_OUTPUT_DIRS)

    target_column = (
        config.UNSW_BINARY_TARGET if config.CLASSIFICATION_MODE == "binary"
        else config.UNSW_MULTICLASS_TARGET
    )

    # ------------------------------------------------------------------
    # 1. Dataset Loading
    # ------------------------------------------------------------------
    logger.info("STEP 1: Loading UNSW-NB15 (source=%s, mode=%s)...", config.UNSW_DATA_SOURCE, config.CLASSIFICATION_MODE)
    if config.UNSW_DATA_SOURCE == "raw_full" and not quick:
        df_train, df_test = load_unsw_full_raw()
    else:
        df_train = load_unsw_raw(config.UNSW_TRAIN_PATH)
        df_test = load_unsw_raw(config.UNSW_TEST_PATH)
    logger.info("Loaded train=%s test=%s", df_train.shape, df_test.shape)

    # ------------------------------------------------------------------
    # 2. Data Validation
    # ------------------------------------------------------------------
    logger.info("STEP 2: Validating schema...")
    expected_cols = set(config.RUNTIME_FEASIBLE_FEATURES) | {config.UNSW_BINARY_TARGET, config.UNSW_MULTICLASS_TARGET}
    missing_cols = expected_cols - (set(df_train.columns) & set(df_test.columns))
    if missing_cols:
        raise ValueError(f"Training data is missing expected columns: {missing_cols}")
    logger.info("Schema OK -- all %d runtime-feasible feature columns present.", len(config.RUNTIME_FEASIBLE_FEATURES))

    # ------------------------------------------------------------------
    # 3. Cleaning
    # ------------------------------------------------------------------
    logger.info("STEP 3: Cleaning (drop id, duplicates, missing values)...")
    df_train = clean_unsw(df_train)
    df_test = clean_unsw(df_test)
    if quick:
        # Stratified subsets of the supplied benchmark, not synthetic accuracy.
        if len(df_train) > 3000:
            df_train, _ = train_test_split(df_train, train_size=3000,
                stratify=df_train[target_column], random_state=config.RANDOM_SEED)
        if len(df_test) > 1000:
            df_test, _ = train_test_split(df_test, train_size=1000,
                stratify=df_test[target_column], random_state=config.RANDOM_SEED)
        config.RANDOM_FOREST_PARAMS = dict(config.RANDOM_FOREST_PARAMS, n_estimators=40, n_jobs=1)

    # ------------------------------------------------------------------
    # Restrict to the runtime-feasible feature set BEFORE encoding/GWO,
    # so nothing downstream ever sees a feature the real-time pipeline
    # cannot actually supply.
    # ------------------------------------------------------------------
    feature_columns = list(config.RUNTIME_FEASIBLE_FEATURES)
    df_train = df_train[feature_columns + [target_column]].copy()
    df_test = df_test[feature_columns + [target_column]].copy()
    # Remove exact cross-split records before any evaluation.
    train_hashes = set(pd.util.hash_pandas_object(df_train, index=False))
    overlaps = pd.util.hash_pandas_object(df_test, index=False).isin(train_hashes)
    overlap_count = int(overlaps.sum())
    df_test = df_test.loc[~overlaps].reset_index(drop=True)
    if len(df_test) < 2 or df_test[target_column].nunique() < 2:
        raise ValueError("Independent test set must contain both classes after removing overlap.")
    df_fit, df_validation = train_test_split(df_train, test_size=0.2,
        stratify=df_train[target_column], random_state=config.RANDOM_SEED)
    validation_rows = len(df_validation)
    # Transform validation and test together, but fit encoders/scaler ONLY on
    # the fitting partition. GWO receives validation rows only below.
    df_test = pd.concat([df_validation, df_test], ignore_index=True)
    df_train = df_fit
    logger.info(
        "Restricted feature space to %d runtime-feasible features (dropped %d offline-only features -- see docs/RUNTIME_FEATURE_AUDIT.md).",
        len(feature_columns), len(config.UNSW_EXCLUDED_FEATURES),
    )

    # ------------------------------------------------------------------
    # 4. Categorical Encoding (with rare-protocol bucketing)
    # ------------------------------------------------------------------
    logger.info("STEP 4: Encoding categorical features...")
    df_train, df_test, rare_proto_values = bucket_rare_protocols(
        df_train, df_test, threshold=config.UNSW_PROTO_RARE_THRESHOLD
    )
    df_train, df_test, categorical_encoders = encode_categoricals(
        df_train, df_test, config.UNSW_CATEGORICAL_COLUMNS
    )

    # ------------------------------------------------------------------
    # 5. Numerical Processing + Feature Scaling
    # ------------------------------------------------------------------
    logger.info("STEP 5: Scaling numeric features...")
    numeric_columns = [c for c in feature_columns if c not in config.UNSW_CATEGORICAL_COLUMNS]
    df_train, df_test, scaler = scale_numeric(df_train, df_test, numeric_columns)

    # ------------------------------------------------------------------
    # Target encoding
    # ------------------------------------------------------------------
    y_train, y_test, label_encoder = encode_target(df_train, df_test, target_column)
    X_train = df_train[feature_columns].values.astype(float)
    X_test = df_test[feature_columns].values.astype(float)

    X_validation, y_validation = X_test[:validation_rows], y_test[:validation_rows]
    X_test, y_test = X_test[validation_rows:], y_test[validation_rows:]
    df_test = df_test.iloc[validation_rows:]
    X_fit, y_fit = X_train, y_train
    X_train = np.concatenate([X_fit, X_validation])
    y_train = np.concatenate([y_fit, y_validation])
    wolves, iterations, runs = (4, 3, 1) if quick else (
        config.UNSW_GWO_POPULATION_SIZE, config.UNSW_GWO_MAX_ITERATIONS, config.UNSW_GWO_RUNS)

    # ------------------------------------------------------------------
    # Baseline model (all runtime-feasible features, no GWO selection)
    # -- kept for the comparison table, same purpose as in main.py
    # ------------------------------------------------------------------
    logger.info("STEP 6a: Training baseline classifier on all %d runtime-feasible features...", len(feature_columns))
    model_factory = get_model_factory(config.UNSW_CLASSIFIER)
    baseline_model = model_factory()
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_accuracy = compute_accuracy(y_test, baseline_preds)
    logger.info("Baseline accuracy: %.2f%%", baseline_accuracy)

    # ------------------------------------------------------------------
    # 6b. Grey Wolf Optimization -- uses the shared src/gwo implementation.
    # Passing categorical_indices=range(dim) disables GWOFeatureSelector's
    # internal per-subset re-scaling (since we already globally scaled
    # numeric columns above), avoiding double-scaling while still using
    # the existing class exactly as written.
    # ------------------------------------------------------------------
    logger.info(
        "STEP 6b: Running %s feature selection on %d runtime-feasible features "
        "(runs=%d, population=%d, iterations=%d)...",
        config.UNSW_GWO_VARIANT, X_train.shape[1], config.UNSW_GWO_RUNS,
        config.UNSW_GWO_POPULATION_SIZE, config.UNSW_GWO_MAX_ITERATIONS,
    )
    selector = GWOFeatureSelector(
        classifier_factory=model_factory,
        X_train=X_fit,
        y_train=y_fit,
        X_test=X_validation,
        y_test=y_validation,
        categorical_indices=list(range(X_train.shape[1])),  # disables internal re-scaling; data is pre-scaled
        alpha=config.FITNESS_ALPHA,
        beta=config.FITNESS_BETA,
    )
    gwo_result = selector.run_multiple(
        variant=config.UNSW_GWO_VARIANT,
        n_runs=runs,
        n_wolves=wolves,
        max_iter=iterations,
        lb=config.GWO_LOWER_BOUND,
        ub=config.GWO_UPPER_BOUND,
        seed=config.RANDOM_SEED,
    )
    best_features_mask = gwo_result.best_wolf.features
    selected_feature_names = [feature_columns[i] for i in range(len(feature_columns)) if best_features_mask[i] == 1]
    logger.info(
        "GWO complete. Selected %d/%d features: %s",
        len(selected_feature_names), len(feature_columns), selected_feature_names,
    )
    logger.info("AAP (avg accuracy across runs): %.2f%%", gwo_result.average_accuracy_performance)
    logger.info("ANF (avg # features across runs): %.2f", gwo_result.average_selected_features)

    # ------------------------------------------------------------------
    # 7. Train final model on GWO-selected features
    # ------------------------------------------------------------------
    logger.info("STEP 7: Training final deployment model on GWO-selected features...")
    selected_idx = [feature_columns.index(f) for f in selected_feature_names]
    X_train_final = X_train[:, selected_idx]
    X_test_final = X_test[:, selected_idx]

    final_model = model_factory()
    final_model.fit(X_train_final, y_train)
    final_preds = final_model.predict(X_test_final)

    # ------------------------------------------------------------------
    # 8. Evaluation (accuracy, precision, recall, F1, confusion matrix,
    #    ROC-AUC, FPR, FNR)
    # ------------------------------------------------------------------
    logger.info("STEP 8: Evaluating final model...")
    print("\n--- Baseline (All Runtime-Feasible Features) Classification Report ---")
    print(compute_classification_report(y_test, baseline_preds))
    print("\n--- GWO-Selected Features Classification Report ---")
    print(compute_classification_report(y_test, final_preds))

    comparison_df = build_comparison_table(
        y_test=y_test,
        baseline_preds=baseline_preds,
        gwo_preds=final_preds,
        n_total_features=len(feature_columns),
        n_selected_features=len(selected_feature_names),
    )
    print("\n--- Comparison Table ---")
    print(comparison_df.to_string(index=False))
    save_comparison_table(comparison_df, os.path.join(config.REPORTS_DIR, "unsw_comparison_table.csv"))

    binary_metrics = {}
    if config.CLASSIFICATION_MODE == "binary":
        binary_metrics = compute_binary_confusion_counts(y_test, final_preds, positive_label=1)
        try:
            y_scores = final_model.predict_proba(X_test_final)[:, 1]
            binary_metrics["roc_auc"] = float(roc_auc_score(y_test, y_scores))
        except AttributeError:
            logger.warning("Classifier has no predict_proba -- skipping ROC-AUC.")
        print("\n--- FPR / FNR / ROC-AUC ---")
        print(json.dumps(binary_metrics, indent=2))

    plot_confusion_matrix(
        y_test, final_preds, title="UNSW-NB15 - GWO-Selected Features (Deployment Model)",
        save_path=os.path.join(config.GRAPHS_DIR, "unsw_confusion_matrix_gwo.png"),
    )
    plot_confusion_matrix(
        y_test, baseline_preds, title="UNSW-NB15 - Baseline (All Runtime-Feasible Features)",
        save_path=os.path.join(config.GRAPHS_DIR, "unsw_confusion_matrix_baseline.png"),
    )

    # ------------------------------------------------------------------
    # 9. Save Model Artifacts (independent inference package)
    # ------------------------------------------------------------------
    logger.info("STEP 9: Saving model artifacts to %s ...", config.SAVED_MODELS_DIR)

    joblib.dump(final_model, config.MODEL_PATH)
    save_preprocessing_artifacts(categorical_encoders, scaler, label_encoder, rare_proto_values)

    joblib.dump(
        {"selected_feature_names": selected_feature_names, "feature_mask": best_features_mask},
        config.SELECTED_FEATURES_PKL_PATH,
    )
    pd.DataFrame({
        "feature_name": feature_columns,
        "selected": best_features_mask,
    }).to_csv(config.SELECTED_FEATURES_CSV_PATH, index=False)

    # feature_schema.json: the runtime system NEVER guesses feature order.
    feature_schema = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "runtime_feasible_features_in_order": feature_columns,
        "categorical_features": config.UNSW_CATEGORICAL_COLUMNS,
        "numeric_features": numeric_columns,
        "selected_features_in_order": selected_feature_names,
        "encoding_method": "LabelEncoder fitted on fitting partition with a reserved __unknown__ category",
        "rare_protocol_bucketing": {
            "threshold": config.UNSW_PROTO_RARE_THRESHOLD,
            "rare_values_bucketed_to_other": sorted(rare_proto_values),
        },
        "scaling_method": "StandardScaler (fit on training data numeric columns only)",
        "target_column": target_column,
        "classification_mode": config.CLASSIFICATION_MODE,
        "positive_class_label": "attack (label=1)" if config.CLASSIFICATION_MODE == "binary" else "see label_encoder.pkl classes_",
        "excluded_features": config.UNSW_EXCLUDED_FEATURES,
        "excluded_features_reference": "docs/RUNTIME_FEATURE_AUDIT.md",
    }
    save_json(feature_schema, config.FEATURE_SCHEMA_PATH)

    training_config = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UNSW-NB15",
        "train_rows": int(len(X_train)),
        "fit_rows": int(len(X_fit)),
        "validation_rows": validation_rows,
        "removed_test_overlaps": overlap_count,
        "quick_mode": quick,
        "selection_data": "training-only stratified validation; final test excluded",
        "sklearn_version": sklearn.__version__,
        "duration_seconds": round(time.perf_counter() - started, 2),
        "test_rows": int(len(df_test)),
        "classifier": config.UNSW_CLASSIFIER,
        "classification_mode": config.CLASSIFICATION_MODE,
        "gwo_variant": config.UNSW_GWO_VARIANT,
        "gwo_population_size": wolves,
        "gwo_max_iterations": iterations,
        "gwo_runs": runs,
        "fitness_alpha": config.FITNESS_ALPHA,
        "fitness_beta": config.FITNESS_BETA,
        "random_seed": config.RANDOM_SEED,
        "n_features_total_runtime_feasible": len(feature_columns),
        "n_features_selected": len(selected_feature_names),
        "baseline_accuracy": baseline_accuracy,
        "gwo_selected_accuracy": compute_accuracy(y_test, final_preds),
        "aap_average_accuracy_performance": gwo_result.average_accuracy_performance,
        "anf_average_num_features": gwo_result.average_selected_features,
        "baseline_report": classification_report(y_test, baseline_preds, output_dict=True, zero_division=0),
        "selected_report": classification_report(y_test, final_preds, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, final_preds).tolist(),
        "selected_features": selected_feature_names,
        "validation_scores": gwo_result.all_scores,
        "validation_feature_counts": gwo_result.all_feature_counts,
        "feature_importance": dict(zip(selected_feature_names, map(float, getattr(final_model, "feature_importances_", [])))),
        "artifact_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in [config.MODEL_PATH, config.ENCODER_PATH, config.SCALER_PATH, config.LABEL_ENCODER_PATH, config.FEATURE_SCHEMA_PATH]},
        **({"binary_metrics": binary_metrics} if binary_metrics else {}),
    }
    save_json(training_config, config.TRAINING_CONFIG_PATH)
    run_dir = Path(config.REPORTS_DIR) / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(training_config, str(run_dir / (feature_schema["model_version"] + ".json")))

    logger.info("Training complete. Artifacts saved to %s", config.SAVED_MODELS_DIR)
    logger.info("Run 'python realtime.py' to start real-time monitoring using this trained model.")


def run_training(quick: bool = False) -> None:
    """Serialize writers; inference refuses a bundle while it is being updated."""
    if Path(config.SAVED_MODELS_DIR).resolve() == (Path(config.BASE_DIR)/"artifacts"/"compact20").resolve():
        raise ValueError("The packaged compact20 profile is read-only. Use build_candidate.py with a new directory.")
    from src.runtime.training_lock import training_lock
    with training_lock(config.SAVED_MODELS_DIR):
        previous_rf = config.RANDOM_FOREST_PARAMS
        try:
            _run_training(quick)
        finally:
            config.RANDOM_FOREST_PARAMS = previous_rf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GWO IDS with isolated validation and final test sets.")
    parser.add_argument("--quick", action="store_true", help="3000 training / 1000 test rows, 4 wolves, 3 iterations, 1 run")
    parser.add_argument("--classifier", choices=["random_forest", "decision_tree", "svm", "knn"], default=config.UNSW_CLASSIFIER)
    parser.add_argument("--variant", choices=["GWO", "bGWO"], default=config.UNSW_GWO_VARIANT)
    args = parser.parse_args()
    config.UNSW_CLASSIFIER, config.UNSW_GWO_VARIANT = args.classifier, args.variant
    run_training(quick=args.quick)
