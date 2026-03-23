Supplementary Note 1: Python Hdbscan script
# Python Hdbscan for clustering and get representative papers from the literature 
import streamlit as st 
import pandas as pd 
import numpy as np 
import hdbscan 
from sklearn.preprocessing import OrdinalEncoder 
import io 
import plotly.express as px 
import umap 
import os 
 
st.set_page_config(page_title="Research Paper Explorer", layout="wide") 
st.title("     Research Paper Explorer") 
 
# ------------------------------ 
# Load dataset 
# ------------------------------ 
uploaded_file = st.file_uploader("Upload your papers Excel file", type=["xlsx"]) 
 
if uploaded_file is not None: 
    df = pd.read_excel(uploaded_file) 
    st.success(f"Loaded {len(df)} papers from uploaded file") 
else: 
    sample_file = "sample_papers.xlsx" 
    if os.path.exists(sample_file): 
        df = pd.read_excel(sample_file) 
        st.info(f"No file uploaded → loaded sample dataset ({len(df)} papers)") 
    else: 
        st.warning("Please upload an Excel file or include 'sample_papers.xlsx' in the repo.") 
        st.stop() 
 
# ------------------------------ 
# Auto-detect categorical columns 
# ------------------------------ 
cat_cols = df.select_dtypes(include=["object"]).columns.tolist() 
if "year_bin" not in df.columns: 
    if "Year" in df.columns: 
        df["year_bin"] = pd.cut(df["Year"], bins=4, labels=False) 
    cat_cols.append("year_bin") 
 
# Optional: column descriptions 
column_descriptions = { 
    "Hazard": "Type of climate/environmental hazard.", 
    "CRM_Stage": "Stage of Climate Risk Management.", 
    "DDMT_Subgroups": "Subgroups of Digital Decision-Making Tools.", 
    "DDMT_Combined": "Combined DDMT categories.", 
    "DDMT_Class": "Advanced or Early DDMT classification.", 
    "Funding Texts": "Funding source information.", 
    "Open Access": "Whether paper is open access.", 
    "Continent": "Continent of the study.", 
    "ClimateRiskRate": "Risk rating of the climate hazard.", 
    "Development_Status": "Developed or developing country.", 
    "year_bin": "Year of publication (binned)." 
} 
 
st.sidebar.header("Preferences & Settings") 
 
# ------------------------------ 
# Feature weighting 
# ------------------------------ 
st.sidebar.subheader("Feature Weights") 
weights = {} 
for col in cat_cols: 
    help_text = column_descriptions.get(col, "") 
    #weights[col] = st.sidebar.slider(f"{col}", 1, 5, 2, help=help_text) 
    # explanation: weights[col] = st.sidebar.slider(f"{col}", min value, max value, defult value, 
help=help_text) 
    weights[col] = st.sidebar.slider(f"{col}", 0, 5, 2, help=help_text) 
 
# ------------------------------ 
# TOP-N representatives 
# ------------------------------ 
top_n = st.sidebar.slider("Number of representative papers per cluster", min_value=1, 
max_value=10, value=3) 
 
# ------------------------------ 
# High-priority selection (collapsible) 
# ------------------------------ 
high_priority_rules = {} 
with st.sidebar.expander("     High-Priority Columns"): 
    for col in cat_cols: 
        help_text = column_descriptions.get(col, "") 
        if st.checkbox(f"Prioritize column: {col}", help=help_text): 
            unique_vals = df[col].dropna().unique().tolist() 
            selected_vals = st.multiselect(f"High-priority values for {col}", options=unique_vals, 
help=f"Select one or more values to prioritize ({col})") 
            if selected_vals: 
                high_priority_rules[col] = selected_vals 
 
# ------------------------------ 
# Low-priority selection (collapsible) 
# ------------------------------ 
low_priority_rules = {} 
with st.sidebar.expander("    Low-Priority Columns"): 
    for col in cat_cols: 
        help_text = column_descriptions.get(col, "") 
        if st.checkbox(f"De-prioritize column: {col}", help=help_text): 
            unique_vals = df[col].dropna().unique().tolist() 
            selected_vals = st.multiselect(f"Low-priority values for {col}", options=unique_vals, 
help=f"Select one or more values to de-prioritize ({col})") 
            if selected_vals: 
                low_priority_rules[col] = selected_vals 
 
# ------------------------------ 
# Encode categorical columns 
# ------------------------------ 
X_cat = df[cat_cols].astype(str) 
encoder = OrdinalEncoder() 
X_encoded = encoder.fit_transform(X_cat) 
 
# Apply feature weights 
for i, col in enumerate(cat_cols): 
    X_encoded[:, i] = X_encoded[:, i] * weights[col] 
 
# ------------------------------ 
# HDBSCAN clustering 
# ------------------------------ 
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1, metric='hamming') 
df["cluster"] = clusterer.fit_predict(X_encoded) 
df["prob"] = clusterer.probabilities_ 
 
st.write("### Cluster Counts") 
st.dataframe(df["cluster"].value_counts()) 
 
# ------------------------------ 
# Representative paper selection 
# ------------------------------ 
representatives_list = [] 
 
for cl in sorted(df["cluster"].unique()): 
    if cl == -1: 
        continue 
    cluster_df = df[df["cluster"] == cl] 
 
    # High-priority papers 
    if high_priority_rules: 
        high_mask = np.ones(len(cluster_df), dtype=bool) 
        for col, vals in high_priority_rules.items(): 
            high_mask &= cluster_df[col].isin(vals) 
        high_df = cluster_df[high_mask].sort_values(by="prob", ascending=False) 
    else: 
        high_df = pd.DataFrame(columns=cluster_df.columns) 
 
    # Low-priority papers 
    if low_priority_rules: 
        low_mask = np.zeros(len(cluster_df), dtype=bool) 
        for col, vals in low_priority_rules.items(): 
            low_mask |= cluster_df[col].isin(vals) 
        low_df = cluster_df[low_mask].sort_values(by="prob", ascending=False) 
    else: 
        low_df = pd.DataFrame(columns=cluster_df.columns) 
 
    # Normal papers 
    normal_df = cluster_df[~cluster_df.index.isin(high_df.index) & 
~cluster_df.index.isin(low_df.index)] 
    normal_df = normal_df.sort_values(by="prob", ascending=False) 
 
    # Top-N 
    top_n_df = pd.concat([high_df, normal_df, low_df]).head(top_n) 
    representatives_list.append(top_n_df) 
 
reps_df = pd.concat(representatives_list) 
 
st.write("### Top Representative Papers") 
st.dataframe(reps_df) 
# ------------------------------ 
# Download results 
# ------------------------------ 
buffer = io.BytesIO() 
reps_df.to_excel(buffer, index=False) 
st.download_button("Download as Excel", buffer, file_name="representative_papers.xlsx") 
# ------------------------------ 
# Optional: UMAP visualization 
# ------------------------------ 
if st.checkbox("Show Cluster Visualization (UMAP 2D)"): 
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='hamming', 
random_state=42) 
embedding = reducer.fit_transform(X_encoded) 
df['x'] = embedding[:,0] 
df['y'] = embedding[:,1] 
fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['Title'] + cat_cols) 
st.plotly_chart(fig, use_container_width=True) 
