"""
config.py

Central configuration for the IDS-GWO-FeatureSelection project.
Every tunable value used anywhere in the pipeline lives here so that
nothing is hardcoded inside the src/ modules.
"""

import os

# --------------------------------------------------------------------------
# PATHS
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, "processed")

TRAIN_RAW_PATH = os.path.join(RAW_DATA_DIR, "KDDTrain.csv")
TEST_RAW_PATH = os.path.join(RAW_DATA_DIR, "KDDTest.csv")

TRAIN_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "Clean_KDDTrain.csv")
TEST_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "Clean_KDDTest.csv")

SELECTED_FEATURES_PATH = os.path.join(DATASET_DIR, "selected_features.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
SELECTED_FEATURES_DIR = os.path.join(OUTPUT_DIR, "selected_features")

ALL_OUTPUT_DIRS = [
    PROCESSED_DATA_DIR,
    GRAPHS_DIR,
    REPORTS_DIR,
    MODELS_DIR,
    SELECTED_FEATURES_DIR,
]

# --------------------------------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# DATASET SCHEMA
# --------------------------------------------------------------------------
# The NSL-KDD-derived dataset used by the original repository has 41
# independent features followed by a single target column named
# "classification" (multi-class: normal / Dos / Probe / R2L / U2R).
TARGET_COLUMN = "classification"
N_FEATURES = 41

# Column indices (0-based, within the 41 feature columns) that are
# categorical and require label encoding instead of scaling:
#   1 -> protocol_type, 2 -> service, 3 -> flag
CATEGORICAL_FEATURE_INDICES = [1, 2, 3]

# --------------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------------------------------
# The dataset ships with a pre-defined train/test split (KDDTrain / KDDTest).
# TEST_SIZE below is only used if a person wants to additionally carve a
# validation split out of the training data.
TEST_SIZE = 0.2
STRATIFY_SPLIT = True

# --------------------------------------------------------------------------
# CLASSIFIER SELECTION
# --------------------------------------------------------------------------
# Which model to use as the "primary" classifier during GWO fitness
# evaluation and in main.py's default pipeline run.
# Options: "random_forest", "decision_tree", "svm", "knn"
CLASSIFIER = "random_forest"

RANDOM_FOREST_PARAMS = {
    "n_estimators": 150,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

DECISION_TREE_PARAMS = {
    "random_state": RANDOM_SEED,
}

SVM_PARAMS = {
    "C": 10,
    "kernel": "rbf",
    "gamma": "auto",
    "probability": True,  # Probability output is required for runtime confidence thresholds.
    "random_state": RANDOM_SEED,
}

KNN_PARAMS = {
    "n_neighbors": 5,
    "n_jobs": -1,
}

# --------------------------------------------------------------------------
# GREY WOLF OPTIMIZATION (GWO) HYPERPARAMETERS
# --------------------------------------------------------------------------
GWO_POPULATION_SIZE = 12      # N: number of wolves in the pack
GWO_MAX_ITERATIONS = 20       # max_iter: iterations per GWO run
GWO_RUNS = 3                  # number of independent GWO runs (for AAP/ANF stats)
GWO_LOWER_BOUND = 0.0         # lb: lower bound of continuous wolf position
GWO_UPPER_BOUND = 1.0         # ub: upper bound of continuous wolf position
GWO_VARIANT = "bGWO"          # "GWO" (sigmoid-only) or "bGWO" (binary + crossover)

# Fitness function weights: fitness = ALPHA * accuracy + BETA * (1 / n_features)
# ALPHA rewards classification performance, BETA rewards fewer features.
FITNESS_ALPHA = 0.6
FITNESS_BETA = 0.4

# --------------------------------------------------------------------------
# EVALUATION
# --------------------------------------------------------------------------
# For ROC curves on a multi-class target, we use one-vs-rest binarization.
ROC_AVERAGE = "macro"

# ============================================================================
# UNSW-NB15 EXTENSION
# ============================================================================
# Everything below this line supports the UNSW-NB15 dataset adaptation and
# the real-time VM deployment extension. The NSL-KDD settings above are left
# untouched so the original main.py / notebooks continue to work unchanged.

ACTIVE_DATASET = "unsw"  # "kdd" (original) or "unsw" (this extension)

# --------------------------------------------------------------------------
# UNSW-NB15 PATHS
# --------------------------------------------------------------------------
UNSW_RAW_DIR = os.path.join(DATASET_DIR, "raw_unsw")

# IMPORTANT: the publicly distributed UNSW-NB15 CSVs are sometimes named
# backwards relative to their actual content (a well-documented dataset
# quirk). These paths point at files that have been VERIFIED (by row count
# against the official published split: 175,341 train / 82,332 test, cross
# checked against the parquet distribution of the same dataset) to contain
# the correct data for their role -- i.e. UNSW_TRAIN_PATH genuinely has
# 175,341 rows and UNSW_TEST_PATH genuinely has 82,332 rows, regardless of
# what the original downloaded filenames said.
UNSW_TRAIN_PATH = os.path.join(UNSW_RAW_DIR, "UNSW_NB15_train.csv")
UNSW_TEST_PATH = os.path.join(UNSW_RAW_DIR, "UNSW_NB15_test.csv")
UNSW_FEATURES_DESC_PATH = os.path.join(UNSW_RAW_DIR, "NUSW-NB15_features.csv")

UNSW_PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "UNSW_NB15_train_clean.csv")
UNSW_PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "UNSW_NB15_test_clean.csv")

# --------------------------------------------------------------------------
# UNSW-NB15 DATA SOURCE
# --------------------------------------------------------------------------
# "presplit"  -> the official 257K-row (175,341 train / 82,332 test)
#                benchmark split shipped in dataset/raw_unsw/. Faster,
#                comparable to published papers using the same split.
# "raw_full"  -> the full ~2.54M-row raw release (dataset/raw_unsw_full/,
#                4 partitions), combined and freshly stratified-split by
#                src/preprocessing/unsw_raw_loader.py. ~10x more training
#                data, but GWO's per-candidate retraining cost scales
#                with it -- expect a substantially longer train.py run.
UNSW_DATA_SOURCE = os.environ.get("IDS_DATA_SOURCE", "presplit")

UNSW_RAW_FULL_DIR = os.path.join(DATASET_DIR, "raw_unsw_full")
UNSW_RAW_PARTITION_PATHS = [
    os.path.join(UNSW_RAW_FULL_DIR, f"UNSW-NB15_{i}.csv") for i in range(1, 5)
]
UNSW_RAW_LIST_EVENTS_PATH = os.path.join(UNSW_RAW_FULL_DIR, "UNSW-NB15_LIST_EVENTS.csv")

UNSW_RAW_TEST_SIZE = 0.2
# Stratified subsample size drawn from the ~2.54M combined raw rows
# BEFORE the train/test split, to keep GWO's runtime manageable. Set to
# None to use the full combined dataset (slow -- hundreds of GWO
# retrains on ~2M+ rows). 400,000 is a reasonable default: still ~1.5x
# larger than the presplit option's total, with a much more tractable
# runtime.
UNSW_RAW_SAMPLE_SIZE = 400_000

# --------------------------------------------------------------------------
# UNSW-NB15 SCHEMA
# --------------------------------------------------------------------------
UNSW_ID_COLUMN = "id"
UNSW_BINARY_TARGET = "label"          # 0 = normal, 1 = attack
UNSW_MULTICLASS_TARGET = "attack_cat"  # 9 attack categories + 'Normal'

UNSW_CATEGORICAL_COLUMNS = ["proto", "service", "state"]

# `proto` has 130+ distinct values in the raw data, the overwhelming
# majority of which are extremely rare/exotic protocols. Label-encoding
# all of them as distinct categories creates a very sparse, noisy
# dimension. Any proto seen fewer than PROTO_RARE_THRESHOLD times in the
# training data is bucketed into a single "other" category before
# encoding -- this also make the encoder robust to protocols that appear
# live at runtime but were never seen (or barely seen) during training.
UNSW_PROTO_RARE_THRESHOLD = 20

# --------------------------------------------------------------------------
# CLASSIFICATION MODE
# --------------------------------------------------------------------------
# "binary"     -> label column (Normal vs Attack). PRIMARY mode, required
#                 for real-time deployment (see spec).
# "multiclass" -> attack_cat column (9 attack types + Normal). Supported
#                 for offline research/report purposes only; NOT used by
#                 the real-time pipeline.
CLASSIFICATION_MODE = "binary"

# --------------------------------------------------------------------------
# RUNTIME-FEASIBLE FEATURE SET
# --------------------------------------------------------------------------
# See docs/RUNTIME_FEATURE_AUDIT.md for the full feature-by-feature
# justification. These are the ONLY features the deployment model is
# trained on, so the offline model and the real-time VM model are
# guaranteed to share an identical, honest feature schema. Features that
# cannot be reliably reconstructed from passively captured live traffic
# (HTTP/FTP application-layer fields, TCP loss counts requiring full
# sequence-number stream analysis) are excluded entirely -- never faked,
# never filled with placeholders.
RUNTIME_FEASIBLE_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sinpkt", "dinpkt", "sjit",
    "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]

# Excluded entirely from the deployment feature space (see audit doc):
UNSW_EXCLUDED_FEATURES = [
    "sloss", "dloss", "trans_depth", "response_body_len",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
]

# --------------------------------------------------------------------------
# UNSW GWO / MODEL SETTINGS
# --------------------------------------------------------------------------
# Reuses the existing src/gwo package and src/models package unchanged --
# only the dataset, feature space, and target column differ from the
# NSL-KDD pipeline in main.py.
UNSW_CLASSIFIER = "random_forest"
UNSW_GWO_POPULATION_SIZE = 10
UNSW_GWO_MAX_ITERATIONS = 15
UNSW_GWO_RUNS = 3
UNSW_GWO_VARIANT = "bGWO"

# --------------------------------------------------------------------------
# MODEL ARTIFACTS (for independent, offline-trained, runtime-loaded inference)
# --------------------------------------------------------------------------
MODEL_PROFILE = os.environ.get("IDS_MODEL_PROFILE","deployment")
if MODEL_PROFILE not in {"deployment","compact20"}:
    raise ValueError("IDS_MODEL_PROFILE must be deployment or compact20.")
SAVED_MODELS_DIR = os.environ.get("IDS_MODEL_DIR", os.path.join(BASE_DIR, "artifacts", MODEL_PROFILE))
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(SAVED_MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(SAVED_MODELS_DIR, "encoder.pkl")
LABEL_ENCODER_PATH = os.path.join(SAVED_MODELS_DIR, "label_encoder.pkl")
SELECTED_FEATURES_PKL_PATH = os.path.join(SAVED_MODELS_DIR, "selected_features.pkl")
SELECTED_FEATURES_CSV_PATH = os.path.join(SAVED_MODELS_DIR, "selected_features.csv")
FEATURE_SCHEMA_PATH = os.path.join(SAVED_MODELS_DIR, "feature_schema.json")
TRAINING_CONFIG_PATH = os.path.join(SAVED_MODELS_DIR, "training_config.json")

# --------------------------------------------------------------------------
# REAL-TIME MONITORING
# --------------------------------------------------------------------------
# Network interface to sniff. On a Kali VM this is commonly "eth0" (NAT/
# bridged adapter). Run `ip a` inside the VM to confirm the actual name --
# this MUST be overridden per-machine, it is not guessable in advance.
RUNTIME_INTERFACE = os.environ.get("IDS_INTERFACE", "eth0")

# How often (seconds) the monitor flushes completed/timed-out flows for
# feature extraction + prediction, independent of individual flow duration.
MONITORING_INTERVAL_SECONDS = 5

# A flow with no new packets for this many seconds is considered complete
# and is pushed through feature extraction + prediction.
FLOW_IDLE_TIMEOUT_SECONDS = 15

# Number of most-recent completed flows kept in memory to compute the
# "connections in last N" family of approximate features (see audit doc).
FLOW_HISTORY_WINDOW = 100

# Probability threshold above which a flow predicted as "attack" (label=1)
# triggers an alert (vs. just being logged as a low-confidence attack call).
ALERT_CONFIDENCE_THRESHOLD = 0.6

# --------------------------------------------------------------------------
# LOGGING / ALERTING
# --------------------------------------------------------------------------
LOGS_DIR = os.path.join(BASE_DIR, "logs")
ALERTS_DB_PATH = os.environ.get("IDS_DB_PATH", os.path.join(BASE_DIR, "logs", "alerts.db"))
RUNTIME_LOG_PATH = os.path.join(LOGS_DIR, "realtime.log")

ALL_UNSW_OUTPUT_DIRS = [
    PROCESSED_DATA_DIR, SAVED_MODELS_DIR, LOGS_DIR,
    GRAPHS_DIR, REPORTS_DIR,
]
