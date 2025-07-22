import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config("🔗 Visual EDA Pipeline", layout="wide")
st.title("🧱 Visual EDA Builder (Repeatable + Reorderable)")

# Upload CSV
file = st.sidebar.file_uploader("📤 Upload your CSV file", type=["csv"])

# Available EDA operations
eda_ops = {
    "Drop Nulls": "drop_nulls",
    "Drop Columns": "drop_columns",
    "Encode Categorical": "encode",
    "Scale Numerical": "scale",
    "Histogram Plot": "plot_hist"
}

# Session state for steps
if "eda_pipeline" not in st.session_state:
    st.session_state.eda_pipeline = []

# Add new step
st.sidebar.header("➕ Add Step")
step_to_add = st.sidebar.selectbox("Choose step to add", list(eda_ops.keys()))
if st.sidebar.button("➕ Add Step"):
    st.session_state.eda_pipeline.append({"type": eda_ops[step_to_add], "params": {}})

# Display and reorder steps
st.sidebar.markdown("### 🔁 Reorder/Delete Steps")
for i, step in enumerate(st.session_state.eda_pipeline):
    col1, col2 = st.sidebar.columns([4, 1])
    col1.write(f"{i+1}. {step['type'].replace('_',' ').title()}")
    if col2.button("❌", key=f"del_{i}"):
        st.session_state.eda_pipeline.pop(i)
        st.rerun()

if file:
    df = pd.read_csv(file)
    df_working = df.copy()

    st.subheader("📄 Original Data")
    st.dataframe(df.head())

    st.markdown("---")
    st.subheader("🔄 Pipeline Execution")

    for idx, step in enumerate(st.session_state.eda_pipeline):
        st.markdown(f"### Step {idx+1}: `{step['type'].replace('_',' ').title()}`")

        if step["type"] == "drop_nulls":
            df_working.dropna(inplace=True)
            st.info("✅ Dropped all rows with nulls.")

        elif step["type"] == "drop_columns":
            col_list = df_working.columns.tolist()
            s
