from pathlib import Path
import json
import pandas as pd
import pytest
import config


@pytest.fixture
def db(tmp_path):
    from src.runtime.alerting import init_db
    path = str(tmp_path / "nested" / "events.db")
    init_db(path)
    return path


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory):
    """Fresh model from small benchmark subsets; never load uploaded pickle files."""
    import train
    root = tmp_path_factory.mktemp("trained")
    patch = pytest.MonkeyPatch()
    attrs = {"SAVED_MODELS_DIR": root / "model", "REPORTS_DIR": root / "reports",
             "GRAPHS_DIR": root / "graphs", "PROCESSED_DATA_DIR": root / "processed",
             "LOGS_DIR": root / "logs", "ALERTS_DB_PATH": root / "db" / "test.db"}
    names = {"MODEL_PATH":"model.pkl", "SCALER_PATH":"scaler.pkl", "ENCODER_PATH":"encoder.pkl",
             "LABEL_ENCODER_PATH":"label_encoder.pkl", "SELECTED_FEATURES_PKL_PATH":"selected_features.pkl",
             "SELECTED_FEATURES_CSV_PATH":"selected_features.csv", "FEATURE_SCHEMA_PATH":"feature_schema.json",
             "TRAINING_CONFIG_PATH":"training_config.json"}
    for attr, name in names.items(): attrs[attr] = root / "model" / name
    # Real supplied benchmark, bounded and isolated for tests.
    for attr, size in [("UNSW_TRAIN_PATH", 240), ("UNSW_TEST_PATH", 120)]:
        data = pd.read_csv(getattr(config, attr)).sample(size, random_state=17)
        path = root / (attr + ".csv")
        data.to_csv(path, index=False)
        attrs[attr] = path
    for attr, value in attrs.items(): patch.setattr(config, attr, str(value))
    patch.setattr(config, "ALL_UNSW_OUTPUT_DIRS", [str(root / x) for x in ["model","reports","graphs","processed","logs"]])
    patch.setattr(config, "UNSW_DATA_SOURCE", "presplit")
    patch.setattr(config, "RANDOM_FOREST_PARAMS", {"n_estimators": 8, "random_state":42,"n_jobs":1})
    patch.setattr(config, "UNSW_GWO_POPULATION_SIZE", 3)
    patch.setattr(config, "UNSW_GWO_MAX_ITERATIONS", 1)
    patch.setattr(config, "UNSW_GWO_RUNS", 1)
    patch.setattr(config, "CLASSIFICATION_MODE", "binary")
    train.run_training()
    yield root
    patch.undo()


@pytest.fixture
def engine(trained_bundle):
    from src.runtime.inference import RuntimeInferenceEngine
    return RuntimeInferenceEngine()
