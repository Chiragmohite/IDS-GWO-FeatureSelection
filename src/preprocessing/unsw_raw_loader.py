"""
src/preprocessing/unsw_raw_loader.py

Loads and combines the FULL raw UNSW-NB15 release (the four
UNSW-NB15_1.csv..4.csv partitions, ~2.54M rows total) instead of the
smaller 257K-row pre-split benchmark files. This is a genuinely larger,
richer training pool -- but the raw files have a different shape than
the pre-split ones and need real reconciliation before they can flow
through the existing preprocessing/GWO/model pipeline unchanged:

  - No header row (columns must be assigned by position).
  - 49 columns instead of 45: includes `srcip`, `sport`, `dstip`,
    `dsport` (dropped here -- see below) and raw `Stime`/`Ltime`
    Unix timestamps (dropped -- `dur` already captures duration).
  - Several columns are named differently than the pre-split files:
    `smeansz`/`dmeansz` -> `smean`/`dmean`, `sintpkt`/`dintpkt` ->
    `sinpkt`/`dinpkt`, `res_bdy_len` -> `response_body_len`,
    `ct_src_ ltm` (note the stray space in the official schema) ->
    `ct_src_ltm`, `Label` -> `label`.
  - `attack_cat` is blank (NaN) for normal traffic in the raw files,
    rather than the literal string 'Normal' the pre-split files use --
    normalized here to match.

After loading, cleaning, and column reconciliation, this module performs
ITS OWN stratified train/test split (the raw partitions are not
pre-divided by the original authors -- the pre-split files are a
separate, independently-curated 10% benchmark subset). The output
dataframes have IDENTICAL column names to what
src/preprocessing/unsw_preprocessing.py and train.py already expect, so
no other module needed to change to support this larger data source.

`srcip`/`dstip`/`sport`/`dport` are dropped, not encoded as features --
training on literal source/destination addresses would make the model
memorize specific hosts from this capture rather than learn generalizable
attack patterns, and those addresses are meaningless outside this dataset.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

# Official 49-column order per NUSW-NB15_features.csv, as they appear
# (positionally) in the unheadered raw partition files.
_RAW_COLUMN_ORDER = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "sjit", "djit",
    "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm_raw", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "label",
]

# Rename map: raw column name -> pre-split-file-compatible column name.
# Only columns that differ are listed; everything else keeps its name.
_RENAME_MAP = {
    "smeansz": "smean",
    "dmeansz": "dmean",
    "sintpkt": "sinpkt",
    "dintpkt": "dinpkt",
    "res_bdy_len": "response_body_len",
    "ct_src_ltm_raw": "ct_src_ltm",
}

# Columns dropped entirely: raw identifiers/timestamps not used as
# model features anywhere in this project (see module docstring).
_DROP_COLUMNS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]


def load_raw_partition(path: str) -> pd.DataFrame:
    """Load a single unheadered UNSW-NB15_N.csv partition with correct column names."""
    df = pd.read_csv(path, header=None, names=_RAW_COLUMN_ORDER, low_memory=False)
    return df


def load_and_combine_raw_partitions(paths: List[str]) -> pd.DataFrame:
    """
    Load and concatenate all raw partitions into a single dataframe,
    then reconcile column names/values to match the pre-split schema
    the rest of the pipeline already expects.
    """
    logger.info("Loading %d raw partition file(s)...", len(paths))
    frames = []
    for p in paths:
        df = load_raw_partition(p)
        logger.info("  %s -> %d rows", p, len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined raw partitions: %d total rows.", len(combined))

    combined = combined.drop(columns=_DROP_COLUMNS)
    combined = combined.rename(columns=_RENAME_MAP)

    # Normalize attack_cat: raw files use blank/NaN for normal traffic;
    # the pre-split files (and this project's schema) use the literal
    # string 'Normal'. Also strip stray whitespace seen in some raw
    # attack_cat values (a documented quirk of this dataset release).
    combined["attack_cat"] = combined["attack_cat"].fillna("Normal").astype(str).str.strip()
    combined.loc[combined["attack_cat"] == "", "attack_cat"] = "Normal"
    # A handful of known inconsistent category spellings in the raw
    # release (e.g. 'Backdoors' vs 'Backdoor') -- normalize to the
    # pre-split files' canonical spelling so multiclass mode stays usable.
    combined["attack_cat"] = combined["attack_cat"].replace({"Backdoors": "Backdoor"})

    combined["label"] = pd.to_numeric(combined["label"], errors="coerce").fillna(0).astype(int)

    # `ct_flw_http_mthd` and `is_ftp_login` are NaN in the raw release
    # whenever they don't apply (e.g. a non-HTTP flow has no HTTP method
    # count, a non-FTP flow has no FTP login flag) -- this is "not
    # applicable", not missing/corrupt data, so it's filled with 0
    # rather than causing the row to be dropped by the generic
    # missing-value cleaning step. Both columns are excluded from the
    # runtime-feasible feature set regardless (see
    # config.UNSW_EXCLUDED_FEATURES / docs/RUNTIME_FEATURE_AUDIT.md),
    # so this only matters for anyone using CLASSIFICATION_MODE to
    # inspect the full offline feature set.
    for col in ("ct_flw_http_mthd", "is_ftp_login"):
        if col in combined.columns:
            combined[col] = combined[col].fillna(0)

    # `rate` is a derived feature present in the pre-split benchmark
    # files (computed by the original authors) but absent from the raw
    # partitions. Computed here using the same (spkts+dpkts)/dur
    # definition applied consistently in
    # src/runtime/feature_extraction.py, so offline and runtime
    # computations of this feature match.
    if "rate" not in combined.columns:
        total_pkts = combined["spkts"] + combined["dpkts"]
        combined["rate"] = np.where(combined["dur"] > 0, total_pkts / combined["dur"], 0.0)

    return combined


def build_train_test_split(
    combined: pd.DataFrame,
    test_size: float,
    random_state: int,
    sample_size: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a fresh, stratified (on `label`) train/test split over the
    combined raw dataset. Optionally subsample first (see
    config.UNSW_RAW_SAMPLE_SIZE) to keep GWO's per-candidate retraining
    cost manageable -- GWO retrains the classifier hundreds of times, so
    running it directly against 2.5M+ rows is a genuinely different
    runtime commitment than the 257K-row pre-split version.

    Parameters
    ----------
    combined : pd.DataFrame
    test_size : float
    random_state : int
    sample_size : Optional[int]
        If given and smaller than len(combined), a stratified subsample
        of this size is drawn BEFORE splitting, preserving the original
        label balance.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test)
    """
    if sample_size is not None and sample_size < len(combined):
        logger.info(
            "Subsampling %d of %d rows (stratified on label) to keep GWO runtime manageable...",
            sample_size, len(combined),
        )
        combined, _ = train_test_split(
            combined, train_size=sample_size, stratify=combined["label"], random_state=random_state,
        )
        combined = combined.reset_index(drop=True)

    df_train, df_test = train_test_split(
        combined, test_size=test_size, stratify=combined["label"], random_state=random_state,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    logger.info(
        "Fresh stratified split: train=%d rows (%.1f%% attack), test=%d rows (%.1f%% attack).",
        len(df_train), 100 * df_train["label"].mean(),
        len(df_test), 100 * df_test["label"].mean(),
    )
    return df_train, df_test


def load_unsw_full_raw(
    partition_paths: List[str] = None,
    test_size: float = None,
    random_state: int = None,
    sample_size: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    One-call entry point: load all raw partitions, reconcile schema,
    and produce a fresh stratified train/test split, ready to hand
    directly to the same cleaning/encoding functions used for the
    pre-split files.

    Defaults are pulled from config.py if not explicitly overridden.
    """
    partition_paths = partition_paths or config.UNSW_RAW_PARTITION_PATHS
    test_size = test_size if test_size is not None else config.UNSW_RAW_TEST_SIZE
    random_state = random_state if random_state is not None else config.RANDOM_SEED
    sample_size = sample_size if sample_size is not None else config.UNSW_RAW_SAMPLE_SIZE

    combined = load_and_combine_raw_partitions(partition_paths)

    n_before = len(combined)
    combined = combined.drop_duplicates().reset_index(drop=True)
    logger.info("Dropped %d duplicate rows from combined raw data (%d -> %d).", n_before - len(combined), n_before, len(combined))

    df_train, df_test = build_train_test_split(combined, test_size, random_state, sample_size)
    return df_train, df_test
