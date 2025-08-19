from __future__ import annotations
import pandas as pd
import io

def read_any(file_bytes: bytes, filename: str) -> pd.DataFrame:
    # Simple loader: CSV / Excel
    name = filename.lower()
    if name.endswith(('.csv', '.txt')):
        return pd.read_csv(io.BytesIO(file_bytes))
    if name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(io.BytesIO(file_bytes))
    # Fallback: try csv
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f'Unsupported file type for {filename}: {e}')
