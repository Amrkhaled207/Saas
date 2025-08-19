from __future__ import annotations
import pandas as pd
import numpy as np
import re

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r'\s+', '_', c.strip().lower()) for c in df.columns]
    df.columns = [re.sub(r'[^0-9a-zA-Z_]', '', c) for c in df.columns]
    return df

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def infer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        # try parse datetimes; ignore errors
        try:
            parsed = pd.to_datetime(df[col], errors='raise', utc=False)
            # if conversion succeeded and has at least, say, 50% non-NaT -> accept
            non_na_ratio = parsed.notna().mean()
            if non_na_ratio > 0.5:
                df[col] = parsed
        except Exception:
            pass
    return df

def handle_missing(df: pd.DataFrame, strategy_numeric: str='median', strategy_categorical: str='most_frequent', fill_constant: str='missing') -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(exclude=['number', 'datetime64[ns]']).columns

    if strategy_numeric == 'mean':
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    elif strategy_numeric == 'median':
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif strategy_numeric == 'zero':
        df[num_cols] = df[num_cols].fillna(0)

    if strategy_categorical == 'most_frequent':
        for c in cat_cols:
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else fill_constant)
    elif strategy_categorical == 'constant':
        df[cat_cols] = df[cat_cols].fillna(fill_constant)

    return df

def auto_clean(df: pd.DataFrame, *, drop_duplicates=True, strip_ws=True, std_names=True, datetime_infer=True, missing_cfg=None) -> pd.DataFrame:
    if drop_duplicates:
        df = df.drop_duplicates()

    if strip_ws:
        df = strip_whitespace(df)

    if std_names:
        df = standardize_column_names(df)

    if datetime_infer:
        df = infer_datetimes(df)

    missing_cfg = missing_cfg or {}
    df = handle_missing(
        df,
        strategy_numeric=missing_cfg.get('strategy_numeric','median'),
        strategy_categorical=missing_cfg.get('strategy_categorical','most_frequent'),
        fill_constant=missing_cfg.get('fill_constant','missing')
    )

    return df
