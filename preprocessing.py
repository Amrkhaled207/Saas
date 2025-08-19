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

def scale(df: pd.DataFrame, scaler_name: str='standard'):
    df = df.copy()
    num_cols = df.select_dtypes(include=['number']).columns
    scaler = select_scaler(scaler_name)
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler

def train_test(df: pd.DataFrame, target: str|None=None, test_size: float=0.2, random_state: int=42):
    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = df[target]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    return train_test_split(df, test_size=test_size, random_state=random_state)
