import streamlit as st
from streamlit_elements import elements, reactflow
import pandas as pd

# Initialize session state for nodes and edges
if "nodes" not in st.session_state:
    st.session_state.nodes = [
        {"id": "1", "type": "input", "data": {"label": "Upload CSV"}, "position": {"x": 100, "y": 50}},
        {"id": "2", "data": {"label": "Drop Nulls"}, "position": {"x": 300, "y": 150}},
        {"id": "3", "data": {"label": "Drop Columns"}, "position": {"x": 500, "y": 250}},
        {"id": "4", "type": "output", "data": {"label": "Result"}, "position": {"x": 700, "y": 350}},
    ]

if "edges" not in st.session_state:
    st.session_state.edges = [
        {"id": "e1-2", "source": "1", "target": "2"},
        {"id": "e2-3", "source": "2", "target": "3"},
        {"id": "e3-4", "source": "3", "target": "4"},
    ]

st.title("🔧 Drag & Connect EDA Pipeline")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
else:
    st.stop()

with elements("flow_editor"):
    with react_flow.reactflow(
        nodes=st.session_state.nodes,
        edges=st.session_state.edges,
        on_nodes_change=lambda changes: st.session_state.nodes.extend(changes),
        on_edges_change=lambda changes: st.session_state.edges.extend(changes),
        fit_view=True,
        style={"width": "100%", "height": 600}
    ):
        react_flow.background(color="#aaa")

# Execute transformations
execution_order = ["1", "2", "3", "4"]
df_exec = st.session_state.df.copy()

for node_id in execution_order:
    if node_id == "2":  # Drop Nulls
        df_exec.dropna(inplace=True)
    elif node_id == "3":  # Drop Columns
        cols_to_drop = st.multiselect("Select columns to drop", df_exec.columns)
        df_exec.drop(columns=cols_to_drop, inplace=True)

# Show output
st.subheader("🔍 Final Output")
st.dataframe(df_exec)

# Download final result
st.download_button("Download Cleaned CSV", df_exec.to_csv(index=False), "cleaned_data.csv")
