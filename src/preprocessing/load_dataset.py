"""
src/preprocessing/load_dataset.py

Single responsibility: read the raw dataset CSV files off disk and
return them as pandas DataFrames with normalized column names.
No cleaning, encoding, or scaling happens here -- see clean_data.py,
encoding.py, and normalization.py respectively.
"""

import re

import pandas as pd


def _normalize_column_name(name: str) -> str:
    """
    Normalize a raw column name: strip whitespace, lowercase, drop dots,
    and strip any stray non-ASCII characters.

    NOTE ON A FIXED DATA ISSUE: the source CSVs shipped by the original
    repository contain a mangled column header --
    "num_failed_loginsì" (with a stray non-ASCII byte appended) instead
    of "num_failed_logins". Silently loading this as-is would produce a
    dataframe whose column name never matches the clean schema in
    src/utils/constants.py, breaking any code that references the
    column by name. Stripping non-ASCII characters here fixes it at
    the source rather than special-casing it downstream.
    """
    name = name.strip().lower().replace(".", "")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load a single raw CSV file into a DataFrame and normalize its
    column names (strip whitespace, lowercase, remove stray dots and
    non-ASCII characters -- see _normalize_column_name for why this
    matters for this specific dataset).

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe with normalized column names.
    """
    df = pd.read_csv(path)
    df.rename(columns=_normalize_column_name, inplace=True)
    return df


def load_train_test(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper to load both the training and testing raw
    datasets in one call.

    Parameters
    ----------
    train_path : str
        Path to the raw training CSV.
    test_path : str
        Path to the raw testing CSV.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test)
    """
    df_train = load_raw_dataset(train_path)
    df_test = load_raw_dataset(test_path)
    return df_train, df_test
