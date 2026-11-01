"""
src/utils/constants.py

Static, non-tunable values shared across modules: column name lists,
class label sets, and other schema-level constants that describe the
*shape* of the data rather than pipeline behaviour (behaviour-level
knobs belong in config.py, not here).
"""

# The 41 feature columns of the NSL-KDD-derived dataset, in order,
# followed by the target column. Kept here as the single source of
# truth so no module has to re-derive this list from a live dataframe.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

TARGET_COLUMN = "classification"

ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Categorical (non-numeric) feature names -- must match
# config.CATEGORICAL_FEATURE_INDICES positionally.
CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Broad attack-category labels found in the "classification" column.
# (Exact label set depends on the specific KDD/NSL-KDD variant used;
# this list documents the ones the original repository's dataset
# contains and is used for readable report headers, not for filtering.)
KNOWN_CLASS_LABELS = ["normal", "Dos", "Probe", "R2L", "U2R"]
