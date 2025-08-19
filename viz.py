from __future__ import annotations
import pandas as pd
import altair as alt
import plotly.express as px

def chart_distribution(df: pd.DataFrame, col: str):
    if pd.api.types.is_numeric_dtype(df[col]):
        return px.histogram(df, x=col)
    return px.bar(df[col].value_counts().reset_index(), x='index', y=col)

def chart_relationship(df: pd.DataFrame, x: str, y: str, color: str|None=None):
    if pd.api.types.is_numeric_dtype(df[y]):
        return px.scatter(df, x=x, y=y, color=color, trendline='ols')
    return px.box(df, x=x, y=y, color=color)
