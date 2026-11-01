"""Inspect locally trained bundles before loading trusted joblib files."""
import hashlib
import json
from pathlib import Path
import sklearn
import config
from src.runtime.training_lock import training_in_progress


def inspect_bundle(check_training=True):
    root = Path(config.SAVED_MODELS_DIR)
    if check_training and training_in_progress(root):
        raise ValueError("Training is active. Inference will be available after the bundle is complete.")
    metadata = root / "training_config.json"
    if not metadata.is_file():
        raise FileNotFoundError("No trained bundle. Run python train.py --quick, or use Model lab.")
    report = json.loads(metadata.read_text(encoding="utf-8"))
    if report.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model and installed scikit-learn versions differ. Retrain in this environment.")
    hashes = report.get("artifact_sha256", {})
    expected = {"model.pkl", "scaler.pkl", "encoder.pkl", "label_encoder.pkl", "feature_schema.json"}
    if not expected.issubset(hashes):
        raise ValueError("Legacy or incomplete bundle: retrain with the current train.py.")
    for name in expected:
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"Model bundle integrity check failed for {name}. Retrain before inference.")
    return report
