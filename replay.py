"""Replay benchmark rows through the real model and event store. No network traffic generated."""
import argparse
import json
import uuid
import pandas as pd
import config
from src.runtime.alerting import init_db
from src.runtime.batch import analyze_frame, store_batch
from src.runtime.inference import RuntimeInferenceEngine


def replay(limit=100):
    if not 1 <= limit <= 1000:
        raise ValueError("Replay limit must be 1..1000.")
    engine = RuntimeInferenceEngine()
    frame = pd.read_csv(config.UNSW_TEST_PATH)
    frame = frame.sample(min(limit, len(frame)), random_state=config.RANDOM_SEED)
    results, features = analyze_frame(frame, engine)
    init_db()
    count = store_batch(results, features, str(uuid.uuid4()), source="dataset_replay",
                        model_version=engine.schema.get("model_version"))
    return {"rows_stored": count, "predicted_attacks": int((results.prediction == "attack").sum()),
            "alerts": int(results.alert.sum()), "source": "dataset_replay", "live_capture": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    print(json.dumps(replay(parser.parse_args().limit), indent=2))
