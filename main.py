"""
main.py

Full end-to-end pipeline:

    Load Dataset
      -> Clean Dataset
      -> Encode Features
      -> Normalize Features
      -> Run Grey Wolf Optimization (feature selection)
      -> Select Features
      -> Train Classifier (baseline + GWO-selected)
      -> Predict
      -> Evaluate
      -> Generate Graphs
      -> Save Outputs

Run with:  python main.py
"""

import argparse
import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split

import config
from src.evaluation.comparison import build_comparison_table, save_comparison_table
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.metrics import (
    compute_accuracy,
    compute_classification_report,
)
from src.gwo.feature_selector import GWOFeatureSelector
from src.models.model_factory import get_model_factory
from src.preprocessing.clean_data import clean_dataset
from src.preprocessing.encoding import encode_categorical_features, split_features_and_target
from src.preprocessing.load_dataset import load_train_test
from src.preprocessing.normalization import numeric_indices_excluding, scale_numeric_features
from src.utils.constants import FEATURE_COLUMNS
from src.utils.helper import ensure_directories, save_json, save_model, save_selected_features
from src.utils.visualization import plot_feature_count_comparison

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


def run_pipeline(quick: bool = False) -> None:
    np.random.seed(config.RANDOM_SEED)

    # ------------------------------------------------------------------
    # 0. Prepare output directories
    # ------------------------------------------------------------------
    ensure_directories(config.ALL_OUTPUT_DIRS)

    # ------------------------------------------------------------------
    # 1. Load Dataset
    # ------------------------------------------------------------------
    logger.info("STEP 1: Loading raw dataset...")
    df_train, df_test = load_train_test(config.TRAIN_RAW_PATH, config.TEST_RAW_PATH)
    logger.info("Loaded train=%s test=%s", df_train.shape, df_test.shape)

    # ------------------------------------------------------------------
    # 2. Clean Dataset
    # ------------------------------------------------------------------
    logger.info("STEP 2: Cleaning dataset (duplicates, missing values)...")
    df_train = clean_dataset(df_train)
    df_test = clean_dataset(df_test)
    if quick:
        # Retain at least ten genuine records of rare classes for validation.
        # No duplicated or synthetic training samples are introduced.
        groups = []
        for _, group in df_train.groupby(config.TARGET_COLUMN):
            count = min(len(group), max(10, round(3000 * len(group) / len(df_train))))
            groups.append(group.sample(count, random_state=config.RANDOM_SEED))
        import pandas as pd
        df_train = pd.concat(groups).sample(frac=1, random_state=config.RANDOM_SEED)
        df_test, _ = train_test_split(df_test, train_size=min(1000, len(df_test)-1),
            stratify=df_test[config.TARGET_COLUMN], random_state=config.RANDOM_SEED)
        config.RANDOM_FOREST_PARAMS = dict(config.RANDOM_FOREST_PARAMS, n_estimators=40, n_jobs=1)
    wolves, iterations, runs = (4, 3, 1) if quick else (
        config.GWO_POPULATION_SIZE, config.GWO_MAX_ITERATIONS, config.GWO_RUNS)
    df_train.to_csv(config.TRAIN_CLEAN_PATH, index=False)
    df_test.to_csv(config.TEST_CLEAN_PATH, index=False)

    # ------------------------------------------------------------------
    # 3. Encode Features
    # ------------------------------------------------------------------
    logger.info("STEP 3: Encoding categorical features...")
    X_train, y_train = split_features_and_target(df_train)
    X_test, y_test = split_features_and_target(df_test)

    X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=config.RANDOM_SEED)
    # Search preprocessing sees training-fold categories only.
    X_fit, X_val, _ = encode_categorical_features(
        X_fit_raw, X_val_raw, config.CATEGORICAL_FEATURE_INDICES)
    X_train, X_test, _encoders = encode_categorical_features(
        X_train, X_test, config.CATEGORICAL_FEATURE_INDICES)

    # ------------------------------------------------------------------
    # 4. Normalize Features (baseline / full-feature model only --
    #    GWO's per-candidate scaling happens inside GWOFeatureSelector,
    #    since the numeric-column set changes with each feature subset)
    # ------------------------------------------------------------------
    logger.info("STEP 4: Scaling numeric features for the baseline model...")
    numeric_idx = numeric_indices_excluding(config.N_FEATURES, config.CATEGORICAL_FEATURE_INDICES)
    X_train_scaled, X_test_scaled, _scaler = scale_numeric_features(X_train, X_test, numeric_idx)
    X_train_scaled = X_train_scaled.astype(float)
    X_test_scaled = X_test_scaled.astype(float)

    # ------------------------------------------------------------------
    # 5. Train Baseline Classifier (all features)
    # ------------------------------------------------------------------
    logger.info("STEP 5: Training baseline classifier (%s) on ALL features...", config.CLASSIFIER)
    model_factory = get_model_factory(config.CLASSIFIER)
    baseline_model = model_factory()
    baseline_model.fit(X_train_scaled, y_train)
    baseline_preds = baseline_model.predict(X_test_scaled)
    baseline_accuracy = compute_accuracy(y_test, baseline_preds)
    logger.info("Baseline accuracy (all %d features): %.2f%%", config.N_FEATURES, baseline_accuracy)

    # ------------------------------------------------------------------
    # 6. Run Grey Wolf Optimization for Feature Selection
    # ------------------------------------------------------------------
    logger.info(
        "STEP 6: Running %s feature selection (runs=%d, population=%d, iterations=%d)...",
        config.GWO_VARIANT, config.GWO_RUNS, config.GWO_POPULATION_SIZE, config.GWO_MAX_ITERATIONS,
    )
    selector = GWOFeatureSelector(
        classifier_factory=model_factory,
        X_train=X_fit,
        y_train=y_fit,
        X_test=X_val,
        y_test=y_val,
        categorical_indices=config.CATEGORICAL_FEATURE_INDICES,
        alpha=config.FITNESS_ALPHA,
        beta=config.FITNESS_BETA,
    )
    gwo_result = selector.run_multiple(
        variant=config.GWO_VARIANT,
        n_runs=runs,
        n_wolves=wolves,
        max_iter=iterations,
        lb=config.GWO_LOWER_BOUND,
        ub=config.GWO_UPPER_BOUND,
        seed=config.RANDOM_SEED,
    )
    best_features = gwo_result.best_wolf.features
    n_selected = sum(best_features)
    logger.info(
        "GWO complete. Best fitness=%.4f, accuracy=%.2f%%, selected %d/%d features.",
        gwo_result.best_wolf.fitness, gwo_result.best_wolf.score, n_selected, config.N_FEATURES,
    )
    logger.info("AAP (avg accuracy across runs): %.2f%%", gwo_result.average_accuracy_performance)
    logger.info("ANF (avg # features across runs): %.2f", gwo_result.average_selected_features)

    # ------------------------------------------------------------------
    # 7. Select Features & Retrain Final Model on GWO-selected subset
    # ------------------------------------------------------------------
    logger.info("STEP 7: Training final model on GWO-selected features...")
    selected_idx = [i for i in range(config.N_FEATURES) if best_features[i] == 1]
    scale_idx_within_subset = [
        selected_idx.index(i) for i in selected_idx if i not in config.CATEGORICAL_FEATURE_INDICES
    ]

    X_train_gwo = X_train[:, selected_idx].astype(float)
    X_test_gwo = X_test[:, selected_idx].astype(float)

    from sklearn.preprocessing import StandardScaler
    gwo_scaler = StandardScaler()
    if scale_idx_within_subset:
        X_train_gwo[:, scale_idx_within_subset] = gwo_scaler.fit_transform(X_train_gwo[:, scale_idx_within_subset])
        X_test_gwo[:, scale_idx_within_subset] = gwo_scaler.transform(X_test_gwo[:, scale_idx_within_subset])

    gwo_model = model_factory()
    gwo_model.fit(X_train_gwo, y_train)

    # ------------------------------------------------------------------
    # 8. Predict
    # ------------------------------------------------------------------
    logger.info("STEP 8: Predicting on held-out test set...")
    gwo_preds = gwo_model.predict(X_test_gwo)

    # ------------------------------------------------------------------
    # 9. Evaluate
    # ------------------------------------------------------------------
    logger.info("STEP 9: Evaluating baseline vs GWO-selected model...")
    print("\n--- Baseline (All Features) Classification Report ---")
    print(compute_classification_report(y_test, baseline_preds))
    print("\n--- GWO-Selected Features Classification Report ---")
    print(compute_classification_report(y_test, gwo_preds))

    comparison_df = build_comparison_table(
        y_test=y_test,
        baseline_preds=baseline_preds,
        gwo_preds=gwo_preds,
        n_total_features=config.N_FEATURES,
        n_selected_features=n_selected,
    )
    print("\n--- Comparison Table ---")
    print(comparison_df.to_string(index=False))
    save_comparison_table(comparison_df, os.path.join(config.REPORTS_DIR, "comparison_table.csv"))

    # ------------------------------------------------------------------
    # 10. Generate Graphs
    # ------------------------------------------------------------------
    logger.info("STEP 10: Generating graphs...")
    plot_confusion_matrix(
        y_test, baseline_preds, title="Confusion Matrix - Baseline (All Features)",
        save_path=os.path.join(config.GRAPHS_DIR, "confusion_matrix_baseline.png"),
    )
    plot_confusion_matrix(
        y_test, gwo_preds, title="Confusion Matrix - GWO-Selected Features",
        save_path=os.path.join(config.GRAPHS_DIR, "confusion_matrix_gwo.png"),
    )
    plot_feature_count_comparison(
        labels=["All Features", "GWO-Selected"],
        feature_counts=[config.N_FEATURES, n_selected],
        save_path=os.path.join(config.GRAPHS_DIR, "feature_count_comparison.png"),
    )

    # ------------------------------------------------------------------
    # 11. Save Outputs (models + selected features + summary report)
    # ------------------------------------------------------------------
    logger.info("STEP 11: Saving outputs...")
    save_model(baseline_model, os.path.join(config.MODELS_DIR, f"baseline_{config.CLASSIFIER}.joblib"))
    save_model(gwo_model, os.path.join(config.MODELS_DIR, f"gwo_{config.CLASSIFIER}.joblib"))
    save_selected_features(
        best_features, FEATURE_COLUMNS,
        os.path.join(config.SELECTED_FEATURES_DIR, "selected_features.csv"),
    )
    save_selected_features(
        best_features, FEATURE_COLUMNS, config.SELECTED_FEATURES_PATH,
    )
    save_json(
        {
            "classifier": config.CLASSIFIER,
            "gwo_variant": config.GWO_VARIANT,
            "baseline_accuracy": baseline_accuracy,
            "gwo_best_accuracy": compute_accuracy(y_test, gwo_preds),
            "validation_best_accuracy": gwo_result.best_wolf.score,
            "quick_mode": quick,
            "selection_data": "training-only validation",
            "gwo_best_fitness": gwo_result.best_wolf.fitness,
            "n_features_total": config.N_FEATURES,
            "n_features_selected": n_selected,
            "average_accuracy_performance_AAP": gwo_result.average_accuracy_performance,
            "average_num_features_ANF": gwo_result.average_selected_features,
        },
        os.path.join(config.REPORTS_DIR, "run_summary.json"),
    )

    logger.info("Pipeline complete. See outputs/ for graphs, reports, models, and selected features.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Original NSL-KDD research pipeline")
    parser.add_argument("--quick", action="store_true")
    run_pipeline(quick=parser.parse_args().quick)
