"""Bounded CSV inference; no execution of uploaded code or model files."""
import io
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from src.runtime.alerting import _connection

MAX_ROWS = 1000
MAX_BYTES = 10 * 1024 * 1024


def read_csv_bytes(data):
    if len(data) > MAX_BYTES:
        raise ValueError("CSV exceeds the 10 MB limit.")
    try:
        frame = pd.read_csv(io.BytesIO(data), nrows=MAX_ROWS + 1)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Upload a valid UTF-8 CSV with a header.") from exc
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"CSV must contain between 1 and {MAX_ROWS} rows.")
    return frame


def analyze_frame(frame, engine):
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Analyze between 1 and {MAX_ROWS} rows at a time.")
    missing = set(engine.feature_order) - set(frame.columns)
    if missing:
        raise ValueError("Missing feature columns: " + ", ".join(sorted(missing)))
    features = frame[engine.feature_order].copy()
    if features.isna().any().any():
        raise ValueError("Feature values cannot be empty.")
    try:
        numeric = features[engine.numeric_features].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric features must contain numbers.") from exc
    if not np.isfinite(numeric.to_numpy()).all() or (numeric < 0).any().any():
        raise ValueError("Numeric features must be finite and nonnegative.")
    features[engine.numeric_features] = numeric
    labels, confidence = engine.predict_many(features)
    results = pd.DataFrame({"row": np.arange(1,len(features)+1),
        "prediction": np.where(labels==1,"attack","normal"), "confidence":confidence,
        "alert": (labels==1) & (confidence>=engine.alert_threshold)})
    return results, features.to_dict("records")


def store_batch(results, features, batch_id, source="csv_import", model_version=None, db_path=None, *, metadata=None):
    """All-or-nothing, idempotent replay/import. Same batch token cannot duplicate events."""
    uuid.UUID(batch_id)
    if source not in {"csv_import", "dataset_replay", "pcap_replay", "loopback_probe"} or len(results) != len(features) or not len(results):
        raise ValueError("Invalid batch.")
    if metadata is None:
        metadata = [{} for _ in features]
    if len(metadata) != len(features):
        raise ValueError("Flow metadata count does not match predictions.")
    with _connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM import_batches WHERE id=?", (batch_id,)).fetchone():
            return 0
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("INSERT INTO import_batches(id,source,created_at,row_count) VALUES(?,?,?,?)",
                     (batch_id, source, now, len(results)))
        for result, feature, meta in zip(results.to_dict("records"), features, metadata):
            conn.execute("""INSERT INTO events(timestamp,prediction,confidence,proto,service,
                is_alert,features_json,source,model_version,batch_id,observed_at,src_ip,src_port,dst_ip,dst_port) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now, result["prediction"], float(result["confidence"]), str(feature["proto"]),
                 str(feature["service"]), int(result["alert"]), json.dumps(feature, allow_nan=False),
                 source, model_version, batch_id, meta.get("observed_at"), meta.get("src_ip"),
                 meta.get("src_port"), meta.get("dst_ip"), meta.get("dst_port")))
    return len(results)
