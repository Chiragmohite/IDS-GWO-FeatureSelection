"""
src/runtime/inference.py

Loads the artifacts saved by train.py (model, scaler, categorical
encoders, feature schema) and applies the EXACT SAME preprocessing
transforms to a live feature dictionary produced by
src/runtime/feature_extraction.py, then runs prediction.

This is the module that guarantees offline/runtime schema compatibility:
it reads feature_schema.json to know the exact feature order and never
guesses it, applies the same rare-protocol bucketing rule the training
data used, reuses the fitted (not refit) LabelEncoders and StandardScaler,
and applies the same GWO-selected feature mask before calling the model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np

import config
from src.runtime.artifacts import inspect_bundle

logger = logging.getLogger(__name__)


class RuntimeInferenceEngine:
    """
    Loads all artifacts once at startup and exposes `predict(features)`
    for the real-time monitoring loop to call per-flow.
    """

    def __init__(self):
        from src.runtime.training_lock import training_lock
        # Hold the writer lock through verification and deserialization so
        # a concurrent training process cannot replace half of the bundle.
        with training_lock(config.SAVED_MODELS_DIR):
            self._load()

    def _load(self):
        inspect_bundle(check_training=False)
        logger.info("Loading model artifacts from %s ...", config.SAVED_MODELS_DIR)
        required = [config.MODEL_PATH, config.SCALER_PATH, config.ENCODER_PATH,
                    config.LABEL_ENCODER_PATH, config.FEATURE_SCHEMA_PATH]
        if not all(Path(path).is_file() for path in required):
            raise FileNotFoundError("No complete model bundle. Run python train.py --quick first.")
        self.model = joblib.load(config.MODEL_PATH)
        if not hasattr(self.model, "predict_proba"):
            raise ValueError("Runtime model must provide class probabilities; retrain with probability support.")
        self.scaler = joblib.load(config.SCALER_PATH)
        encoder_bundle = joblib.load(config.ENCODER_PATH)
        self.categorical_encoders = encoder_bundle["categorical_encoders"]
        self.rare_proto_values = encoder_bundle["rare_proto_values"]
        self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

        with open(config.FEATURE_SCHEMA_PATH) as f:
            self.schema = json.load(f)

        if self.schema.get("classification_mode") != "binary":
            raise ValueError("Live monitoring requires a binary deployment model. Multiclass remains offline-only.")
        self.alert_threshold = float(self.schema.get("alert_threshold", config.ALERT_CONFIDENCE_THRESHOLD))
        if not np.isfinite(self.alert_threshold) or not .5 <= self.alert_threshold <= 1:
            raise ValueError("Invalid model alert threshold.")
        self.feature_order = self.schema["runtime_feasible_features_in_order"]
        self.categorical_features = self.schema["categorical_features"]
        self.numeric_features = self.schema["numeric_features"]
        self.selected_features = self.schema["selected_features_in_order"]
        self.selected_idx = [self.feature_order.index(f) for f in self.selected_features]

        logger.info(
            "Loaded model expecting %d runtime features, using %d GWO-selected features.",
            len(self.feature_order), len(self.selected_features),
        )

    def _prepare_frame(self, frame):
        import pandas as pd
        missing = set(self.feature_order) - set(frame.columns)
        if missing: raise ValueError(f"Feature vector is missing required features: {missing}")
        data = frame[self.feature_order].copy()
        if data.empty or data.isna().any().any():raise ValueError("Feature values cannot be empty.")
        for col in self.categorical_features:
            encoder=self.categorical_encoders[col]
            values=data[col].astype(str)
            if col=="proto":values=values.mask(values.isin(self.rare_proto_values),"other")
            unknown=~values.isin(encoder.classes_)
            if unknown.any() and "__unknown__" not in encoder.classes_:raise ValueError("Unknown category: retrain the model.")
            data[col]=encoder.transform(values.mask(unknown,"__unknown__"))
        numeric=data[self.numeric_features].apply(pd.to_numeric,errors="raise")
        if not np.isfinite(numeric.to_numpy()).all() or (numeric<0).any().any():
            raise ValueError("All numeric features must be finite and nonnegative.")
        matrix=data.to_numpy(dtype=float)
        matrix[:,[self.feature_order.index(k) for k in self.numeric_features]]=self.scaler.transform(numeric)
        return matrix

    def _apply_preprocessing(self, raw_features: Dict) -> np.ndarray:
        import pandas as pd
        return self._prepare_frame(pd.DataFrame([raw_features]))

    def predict_many(self, frame):
        matrix=self._prepare_frame(frame)[:,self.selected_idx]
        labels=self.model.predict(matrix).astype(int)
        scores=self.model.predict_proba(matrix)
        mapping={int(v):i for i,v in enumerate(self.model.classes_)}
        confidence=scores[np.arange(len(labels)),[mapping[int(v)] for v in labels]]
        return labels,confidence

    def predict(self, raw_features: Dict) -> Tuple[int, float]:
        import pandas as pd
        labels,scores=self.predict_many(pd.DataFrame([raw_features]))
        return int(labels[0]),float(scores[0])
