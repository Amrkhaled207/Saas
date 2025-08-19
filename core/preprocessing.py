from __future__ import annotations
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from category_encoders.one_hot import OneHotEncoder
from category_encoders.target_encoder import TargetEncoder
from feature_engine.encoding import OrdinalEncoder

def select_scaler(name: str):
    if name == 'standard':
        return StandardScaler()
    if name == 'minmax':
        return MinMaxScaler()
    if name == 'robust':
        return RobustScaler()
    raise ValueError(f'Unknown scaler: {name}')

def encode(df: pd.DataFrame, encoder_type: str='onehot', target: str|None=None):
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    enc = None
    if encoder_type == 'onehot':
        if cat_cols:
            enc = OneHotEncoder(cols=cat_cols, use_cat_names=True, return_df=True, handle_unknown='ignore')
            df = enc.fit_transform(df)
    elif encoder_type == 'target':
        if target is None: 
            raise ValueError('TargetEncoder requires target column name')
        cat_cols = [c for c in cat_cols if c != target]
        if cat_cols:
            enc = TargetEncoder(cols=cat_cols, return_df=True)
            df[cat_cols] = enc.fit_transform(df[cat_cols], df[target])
    elif encoder_type == 'ordinal':
        if cat_cols:
            enc = OrdinalEncoder(encoding_method='arbitrary', variables=cat_cols)
            df = enc.fit_transform(df)
    else:
        raise ValueError(f'Unknown encoder: {encoder_type}')
    return df, enc

def scale(df: pd.DataFrame, scaler_name: str = 'standard'):
    import numpy as np
    df = df.copy()

    # numeric columns only
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not num_cols:  # nothing to scale
        return df, None

    # coerce suspicious values to numeric, handle infs/nans
    X = df[num_cols].apply(pd.to_numeric, errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)

    # keep columns that have at least one non-NaN value
    keep = [c for c in X.columns if X[c].notna().any()]
    if not keep:  # all became NaN -> skip scaling
        return df, None

    scaler = select_scaler(scaler_name)

    # simple policy: fill remaining NaNs with 0 before scaling
    X_scaled = scaler.fit_transform(X[keep].fillna(0))

    # write back
    df[keep] = X_scaled
    return df, scaler

def train_test(df: pd.DataFrame, target: str|None=None, test_size: float=0.2, random_state: int=42):
    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_test_split(df, test_size=test_size, random_state=random_state)
