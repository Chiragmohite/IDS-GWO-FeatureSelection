"""
src/preprocessing/clean_data.py

Single responsibility: remove duplicate rows and handle missing values.
Mirrors the cleaning steps performed inline in the original repository's
notebook (df.duplicated() / df.drop_duplicates() / df.isnull()), but
made explicit, reusable, and logged instead of silently run in a cell.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate rows from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        A new dataframe with duplicate rows removed and the index reset.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    logger.info("Removed %d duplicate rows (%d -> %d).", n_removed, n_before, len(df))
    return df


def report_missing_values(df: pd.DataFrame) -> int:
    """
    Count total missing (NaN) values in a dataframe and log the result.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    int
        Total number of missing cells across the whole dataframe.
    """
    total_missing = int(df.isnull().sum().sum())
    logger.info("Total missing values: %d", total_missing)
    return total_missing


def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in the dataframe. The original NSL-KDD-derived
    dataset used by this project contains no missing values in practice,
    but this function exists so the pipeline degrades gracefully if a
    different or dirtier dataset variant is substituted later.

    Parameters
    ----------
    df : pd.DataFrame
    strategy : str
        "drop"  -> drop any row containing a NaN
        "zero"  -> fill numeric NaNs with 0
        "ffill" -> forward-fill NaNs

    Returns
    -------
    pd.DataFrame
    """
    n_missing = report_missing_values(df)
    if n_missing == 0:
        return df

    if strategy == "drop":
        df = df.dropna().reset_index(drop=True)
    elif strategy == "zero":
        df = df.fillna(0)
    elif strategy == "ffill":
        df = df.fillna(method="ffill")
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")

    logger.info("Handled missing values using strategy='%s'.", strategy)
    return df


def clean_dataset(df: pd.DataFrame, missing_strategy: str = "drop") -> pd.DataFrame:
    """
    Run the full cleaning pipeline: duplicate removal followed by
    missing-value handling.

    Parameters
    ----------
    df : pd.DataFrame
    missing_strategy : str
        Passed through to handle_missing_values().

    Returns
    -------
    pd.DataFrame
        The cleaned dataframe.
    """
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy=missing_strategy)
    return df
