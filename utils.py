from __future__ import annotations
import yaml
from pydantic import BaseModel
from typing import Optional, Literal

class MissingCfg(BaseModel):
    strategy_numeric: Literal['mean','median','zero']='median'
    strategy_categorical: Literal['most_frequent','constant']='most_frequent'
    fill_constant: str='missing'

class CleaningCfg(BaseModel):
    drop_duplicates: bool=True
    strip_whitespace: bool=True
    standardize_colnames: bool=True
    datetime_infer: bool=True
    missing: MissingCfg=MissingCfg()

class PreprocessingCfg(BaseModel):
    encode_categoricals: bool=True
    encoder: Literal['onehot','target','ordinal']='onehot'
    scale_numeric: bool=True
    scaler: Literal['standard','minmax','robust']='standard'

class QaCfg(BaseModel):
    llm_enabled: bool=False
    provider: Literal['openai','gemini']='openai'

class Config(BaseModel):
    cleaning: CleaningCfg=CleaningCfg()
    preprocessing: PreprocessingCfg=PreprocessingCfg()
    qa: QaCfg=QaCfg()

def load_config(path: str='config.yaml') -> Config:
    with open(path,'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)
