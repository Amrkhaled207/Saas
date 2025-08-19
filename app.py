import streamlit as st
import pandas as pd
from core import ingestion, cleaning, preprocessing, analysis, viz, qa
from core.utils import load_config

st.set_page_config(page_title='Universal Data Cleaner & Q&A Dashboard', layout='wide')
st.title('🧼 Universal Data Cleaner & Q&A Dashboard')

cfg = load_config()

with st.sidebar:
    st.header('Upload / Options')
    uploaded = st.file_uploader('Upload CSV/Excel', type=['csv','txt','xlsx','xls'])
    use_sample = st.checkbox('Use sample dataset (Iris)', value=not uploaded)
    target_col = st.text_input('Target column (optional)', value='')
    st.divider()
    st.subheader('Cleaning')
    drop_dup = st.checkbox('Drop duplicates', value=cfg.cleaning.drop_duplicates)
    strip_ws = st.checkbox('Strip whitespace', value=cfg.cleaning.strip_whitespace)
    std_names = st.checkbox('Standardize colnames', value=cfg.cleaning.standardize_colnames)
    datetime_inf = st.checkbox('Infer datetimes', value=cfg.cleaning.datetime_infer)

    st.subheader('Missing')
    strat_num = st.selectbox('Numeric strategy', options=['mean','median','zero'], index=['mean','median','zero'].index(cfg.cleaning.missing.strategy_numeric))
    strat_cat = st.selectbox('Categorical strategy', options=['most_frequent','constant'], index=['most_frequent','constant'].index(cfg.cleaning.missing.strategy_categorical))
    fill_const = st.text_input('Fill constant', value=cfg.cleaning.missing.fill_constant)

    st.subheader('Preprocessing')
    enc_enable = st.checkbox('Encode categoricals', value=cfg.preprocessing.encode_categoricals)
    enc_type = st.selectbox('Encoder', options=['onehot','target','ordinal'], index=['onehot','target','ordinal'].index(cfg.preprocessing.encoder))
    scale_enable = st.checkbox('Scale numeric', value=cfg.preprocessing.scale_numeric)
    scaler_name = st.selectbox('Scaler', options=['standard','minmax','robust'], index=['standard','minmax','robust'].index(cfg.preprocessing.scaler))

if use_sample:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True).frame
    df_raw = iris.rename(columns={'target':'species'})
else:
    if not uploaded:
        st.stop()
    df_raw = ingestion.read_any(uploaded.read(), uploaded.name)

st.subheader('1) Preview')
st.dataframe(df_raw.head(20), use_container_width=True)

st.subheader('2) Cleaning')
df_clean = cleaning.auto_clean(
    df_raw,
    drop_duplicates=drop_dup,
    strip_ws=strip_ws,
    std_names=std_names,
    datetime_infer=datetime_inf,
    missing_cfg={
        'strategy_numeric': strat_num,
        'strategy_categorical': strat_cat,
        'fill_constant': fill_const
    }
)
st.write('Shape after cleaning:', df_clean.shape)
st.dataframe(df_clean.head(20), use_container_width=True)

st.subheader('3) Quick Stats')
st.dataframe(analysis.quick_stats(df_clean), use_container_width=True)

st.subheader('4) Preprocessing (optional)')
df_proc = df_clean.copy()
enc_obj = scale_obj = None
if enc_enable:
    try:
        df_proc, enc_obj = preprocessing.encode(df_proc, encoder_type=enc_type, target=target_col if target_col else None)
    except Exception as e:
        st.warning(f'Encoding skipped: {e}')
if scale_enable:
    df_proc, scale_obj = preprocessing.scale(df_proc, scaler_name=scaler_name)
st.write('Shape after preprocessing:', df_proc.shape)
st.dataframe(df_proc.head(20), use_container_width=True)

st.subheader('5) Ask a Question')
q = st.text_input('Example: "distribution of sepal_length", "relationship between sepal_length and petal_length", "average of sepal_length by species", "summary"')
if q:
    intent = qa.parse_intent(q)
    if intent['type'] == 'distribution':
        col = intent['col']
        if col in df_clean.columns:
            fig = viz.chart_distribution(df_clean, col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f'Column not found: {col}')
    elif intent['type'] == 'relationship':
        x, y = intent['x'], intent['y']
        color = None
        if x in df_clean.columns and y in df_clean.columns:
            fig = viz.chart_relationship(df_clean, x, y, color)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error('Columns not found.')
    elif intent['type'] == 'sql':
        query = intent['sql']
        out = analysis.sql_query(df_clean, query)
        st.dataframe(out, use_container_width=True)
    elif intent['type'] == 'stats':
        st.dataframe(analysis.quick_stats(df_clean), use_container_width=True)
    else:
        st.info('Could not understand the question. Try the examples above.')

st.subheader('6) SQL Console (DuckDB)')
sql = st.text_area('Run SQL on your cleaned table `t` (Registered automatically)', value='SELECT * FROM t LIMIT 5')
if st.button('Run SQL'):
    out = analysis.sql_query(df_clean, sql)
    st.dataframe(out, use_container_width=True)

st.caption('Tip: Export cleaned data with the "Download as CSV" button below.')

csv = df_clean.to_csv(index=False).encode('utf-8')
st.download_button('Download cleaned CSV', data=csv, file_name='cleaned.csv', mime='text/csv')
