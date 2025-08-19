from __future__ import annotations
import pandas as pd
import duckdb

def sql_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    # Register and query via DuckDB
    con = duckdb.connect()
    con.register('t', df)
    try:
        out = con.execute(query).df()
        return out
    finally:
        con.close()

def quick_stats(df):
    # Works across pandas versions (2.0–2.3+)
    try:
        # Newer pandas may accept this
        desc = df.describe(include='all', datetime_is_numeric=True).transpose()
    except TypeError:
        # Fallback for builds where the kwarg isn’t supported
        desc = df.describe(include='all').transpose()
    return desc.reset_index().rename(columns={'index': 'column'})

