import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
import io
import string
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import zscore
import ppscore as pps


st.set_page_config(layout="wide")
st.markdown("""
<h1 style='text-align: center; font-size: 40px;'>💡 EDAFlow – Smooth, Modular Exploratory Data Analysis</h1>
<p style='text-align: center; font-size: 18px;'>✨ Build your EDA pipeline effortlessly with interactive components 🔍📊</p>
""", unsafe_allow_html=True)
st.divider()

with st.expander("📘 EDAFlow – Quick Visual Guide", expanded=False):
    st.markdown("""
    <style>
        .workflow-expander {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #1c7430;
            padding: 14px 18px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 500;
            color: #1c1c1c;
            margin-top: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        }
        .workflow-expander span {
            display: flex;
            align-items: center;
        }
        .workflow-expander span::after {
            content: "⟶";
            margin: 0 10px;
            color: #999;
        }
        .workflow-expander span:last-child::after {
            content: "";
            margin: 0;
        }
    </style>

    <div class="workflow-expander">
        <span>🗂️ Upload</span>
        <span>⚙️ Choose Step</span>
        <span>🔄 Switch Steps</span>
        <span>📊 View Output</span>
        <span>💾 Download</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()



# Upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])


# EDA Operation List
eda_ops = {
    "Treat Nulls": "treat_nulls",
    "Drop Columns": "drop_columns",
    "Encode Categorical": "encode_categorical",
    "Scale Numerical": "scale_numerical",
    "Histogram Plot": "histogram",
    "Boxplot": "boxplot",
    "Correlation Matrix": "correlation",
    "PPS Matrix": "pps_matrix",
    "Remove Outliers": "remove_outliers",
    "Change Dtypes":"change_dtypes",
    "Drop Duplicates": "drop_duplicates",
    "Feature Selection": "feature_selection",
    "Rename Columns": "rename_columns",
    "Balance Target Classes": "balance_classes",
    "GroupBy Summary": "groupby_summary",
    "Pairplot": "pairplot",
    "Skewness & Kurtosis": "distribution_stats",
    "Unique Value Count": "unique_values",
    "Correlation Matrix": "correlation_matrix",
    "Descriptive Stats": "describe_stats"
}

if 'eda_pipeline' not in st.session_state:
    st.session_state.eda_pipeline = []

# Add EDA Step
st.sidebar.header("📋 Add EDA Step")
new_step = st.sidebar.selectbox("Select operation", list(eda_ops.keys()))
if st.sidebar.button("➕ Add Step"):
    st.session_state.eda_pipeline.append(eda_ops[new_step])

# Save/Load Pipeline
with st.sidebar.expander("💾 Save/Load Pipeline"):
    if st.button("💾 Save Pipeline"):
        pipeline_json = json.dumps(st.session_state.eda_pipeline)
        st.download_button("⬇️ Download Pipeline", pipeline_json, "pipeline.json", "application/json")

    uploaded_pipeline = st.file_uploader("📤 Load Pipeline JSON", type="json")
    if uploaded_pipeline:
        st.session_state.eda_pipeline = json.load(uploaded_pipeline)
        st.rerun()

# Display Current Pipeline
st.sidebar.markdown("### 🔄 Current EDA Steps")
for i, step in enumerate(st.session_state.eda_pipeline):
    cols = st.sidebar.columns([5, 1, 1])
    cols[0].write(f"{i+1}. {step}")
    if i > 0 and cols[1].button("⬆️", key=f"up_{i}"):
        st.session_state.eda_pipeline[i-1], st.session_state.eda_pipeline[i] = \
            st.session_state.eda_pipeline[i], st.session_state.eda_pipeline[i-1]
        st.rerun()
    if i < len(st.session_state.eda_pipeline) - 1 and cols[2].button("⬇️", key=f"down_{i}"):
        st.session_state.eda_pipeline[i+1], st.session_state.eda_pipeline[i] = \
            st.session_state.eda_pipeline[i], st.session_state.eda_pipeline[i+1]
        st.rerun()

# Process File
if uploaded_file:
        # 📊 Load CSV or Excel
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        excel_file = pd.ExcelFile(uploaded_file, engine='openpyxl')
        if len(excel_file.sheet_names) > 1:
            sheet = st.radio('📄 Select sheet', excel_file.sheet_names)
            df = pd.read_excel(excel_file, sheet_name=sheet)
        else:
            df = pd.read_excel(uploaded_file)

    df_working = df.copy()

    st.markdown("### 🧪 Original Data Preview")
    st.dataframe(df)
    st.divider()
 
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_string = buffer.getvalue()
    st.code(info_string)
    

    for idx, step_type in enumerate(st.session_state.eda_pipeline):
        st.divider()
        st.markdown(f"### ⚙️ Step {idx+1}: {step_type.replace('_', ' ').title()}")

        if step_type == "treat_nulls":
            null_counts = df_working.isnull().sum()
            null_cols = null_counts[null_counts > 0]
           
            if null_cols.empty:
                st.success("✅ No null values found in the dataset!")
            else:
                st.subheader("⚠️ Null values detected")
                st.write(df_working[df_working.isnull().any(axis=1)])
                st.dataframe(null_cols.to_frame(name="Missing Count"))

                option = st.radio("Choose null treatment:", ["select drop/fill","Drop rows", "Fill values"], key=f"null_option_{idx}")

                if option == "select drop/fill":
                    st.write("Choose One")
                elif option == "Drop rows":
                    df_working.dropna(inplace=True)
                    st.success("✅ Null values dropped")
                else:
                    fill_method = st.selectbox("Fill strategy", ["Mean", "Median", "Mode", "Custom Value"], key=f"fill_method_{idx}")
                    selected_cols = st.multiselect("Select columns to fill", df_working.columns[df_working.isnull().any()], key=f"fill_cols_{idx}")
                    if fill_method == "Mean":
                        for col in selected_cols:
                            df_working[col].fillna(df_working[col].mean(), inplace=True)
                    elif fill_method == "Median":
                        for col in selected_cols:
                            df_working[col].fillna(df_working[col].median(), inplace=True)
                    elif fill_method == "Mode":
                        for col in selected_cols:
                            df_working[col].fillna(df_working[col].mode()[0], inplace=True)
                    elif fill_method == "Custom Value":
                        custom_value = st.text_input("Enter custom value", key=f"custom_fill_{idx}")
                        for col in selected_cols:
                            df_working[col].fillna(custom_value, inplace=True)
                    st.success("✅ Null values filled")
            

        elif step_type == "drop_columns":
            drop_cols = st.multiselect("Select columns to drop", df_working.columns.tolist(), key=f"drop_{idx}")
            df_working.drop(columns=drop_cols, inplace=True)

        elif step_type == "encode_categorical":
            cat_cols = df_working.select_dtypes(include=["object", "category"]).columns.tolist()
            cols = st.multiselect("Select categorical columns", cat_cols, key=f"cat_{idx}")
            
            if cols:
                for col in cols:
                    le = LabelEncoder()
                    df_working[col] = le.fit_transform(df_working[col].astype(str))

                    # Show the mapping
                    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                    st.write(f"📊 Encoding for **{col}**:")
                    st.json(mapping)
                st.success("✅ Selected columns encoded successfully.")
            else:
                st.warning("⚠️ Please select at least one categorical column.")
        
 
        elif step_type == "scale_numerical":
            num_cols = df_working.select_dtypes(include=["int64", "float64"]).columns.tolist()
            
            st.radio("Apply scaling to:", ["Selected Columns", "All Numeric Columns"], key=f"scale_scope_{idx}")
            scope = st.session_state[f"scale_scope_{idx}"]

            scaler_option = st.selectbox(
                "Choose scaling method",
                ["StandardScaler", "MinMaxScaler", "RobustScaler", "Log Transform (log1p)", "Z-score"],
                key=f"scaler_{idx}"
            )

            if scope == "Selected Columns":
                cols_to_scale = st.multiselect("Select numerical columns", num_cols, key=f"scale_cols_{idx}")
            else:
                cols_to_scale = num_cols

            if st.button(f"Apply Scaling - Step {idx+1}"):
                if not cols_to_scale:
                    st.warning("⚠️ Please select at least one column to scale.")
                else:
                    try:
                        if scaler_option == "StandardScaler":
                            scaler_obj = StandardScaler()
                            df_working[cols_to_scale] = scaler_obj.fit_transform(df_working[cols_to_scale])

                        elif scaler_option == "MinMaxScaler":
                            scaler_obj = MinMaxScaler()
                            df_working[cols_to_scale] = scaler_obj.fit_transform(df_working[cols_to_scale])

                        elif scaler_option == "RobustScaler":
                            scaler_obj = RobustScaler()
                            df_working[cols_to_scale] = scaler_obj.fit_transform(df_working[cols_to_scale])

                        elif scaler_option == "Log Transform (log1p)":
                            df_working[cols_to_scale] = df_working[cols_to_scale].apply(lambda x: np.log1p(x))

                        elif scaler_option == "Z-score":
                            df_working[cols_to_scale] = (df_working[cols_to_scale] - df_working[cols_to_scale].mean()) / df_working[cols_to_scale].std()

                        st.success(f"✅ Applied {scaler_option} to {len(cols_to_scale)} column(s).")
                    except Exception as e:
                        st.error(f"⚠️ Error applying scaling: {e}")


        elif step_type == "histogram":
            cols = st.multiselect("Select columns for histogram", df_working.columns.tolist(), key=f"hist_{idx}")
            for col in cols:
                fig, ax = plt.subplots()
                sns.histplot(df_working[col], kde=True, ax=ax)
                st.pyplot(fig)

        elif step_type == "boxplot":
            cols = st.multiselect("Select columns for boxplot", df_working.columns.tolist(), key=f"box_{idx}")
            for col in cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=df_working[col], ax=ax)
                st.pyplot(fig)

        elif step_type == "correlation":
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(df_working.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif step_type == "pps_matrix":
            st.subheader("🔍 Predictive Power Score (PPS)")

            target_col = st.selectbox("🎯 Select Target (Y) Column", df_working.columns, key=f"pps_target_{idx}")

            if target_col:
                pps_df = pps.matrix(df_working)
                filtered_pps = pps_df[pps_df["y"] == target_col][["x", "ppscore"]]
                filtered_pps = filtered_pps.sort_values(by="ppscore", ascending=False).reset_index(drop=True)

                st.markdown(f"📌 **Top Predictors for `{target_col}`**")
                st.dataframe(filtered_pps) 
        elif step_type == "remove_outliers":
            method = st.selectbox("Choose method", ["IQR", "Z-Score"], key=f"outlier_method_{idx}")
            num_cols = df_working.select_dtypes(include=["int64", "float64"]).columns.tolist()
            selected = st.multiselect("Select numeric columns:", num_cols, key=f"outlier_cols_{idx}")
            for col in selected:
                if method == "IQR":
                    Q1 = df_working[col].quantile(0.25)
                    Q3 = df_working[col].quantile(0.75)
                    IQR = Q3 - Q1
                    df_working = df_working[(df_working[col] >= Q1 - 1.5 * IQR) & (df_working[col] <= Q3 + 1.5 * IQR)]
                else:
                    df_working = df_working[np.abs(zscore(df_working[col])) < 3]
            st.success(f"✅ Outliers removed using {method}.")

        elif step_type == "change_dtypes":
            st.subheader("🔄 Change Column Data Types")

            selected_columns = st.multiselect("🧱 Select Columns to Convert", df_working.columns, key=f"{idx}_cols")

            conversions = {}
            for col in selected_columns:
                dtype = st.selectbox(
                    f"🔁 Convert `{col}` to:",
                    ["object", "string", "int64", "float64", "bool", "datetime64"],
                    key=f"{idx}_{col}_dtype"
                )
                conversions[col] = dtype

            if st.button("✅ Apply Conversions", key=f"{idx}_apply"):
                success, failed = [], []

                for col, dtype in conversions.items():
                    try:
                        if dtype == "datetime64":
                            df_working[col] = pd.to_datetime(df_working[col], errors="coerce")
                        elif dtype in ["int64", "float64"]:
                            df_working[col] = pd.to_numeric(df_working[col], errors="coerce")
                        elif dtype == "bool":
                            df_working[col] = df_working[col].astype(bool)
                        elif dtype == "string":
                            df_working[col] = df_working[col].astype("string")
                        else:
                            df_working[col] = df_working[col].astype(dtype)

                        success.append(col)
                    except Exception as e:
                        failed.append((col, str(e)))

                if success:
                    st.success(f"✅ Converted: {', '.join(success)}")
                if failed:
                    for col, err in failed:
                        st.error(f"❌ `{col}`: {err}")

        elif step_type == "drop_duplicates":
            before = df_working.shape[0]
            df_working.drop_duplicates(inplace=True)
            after = df_working.shape[0]
            st.success(f"✅ Dropped {before - after} duplicate rows.")


        elif step_type == "feature_selection":
            target = st.selectbox("Select target column", df_working.columns.tolist(), key=f"target_{idx}")
            num_feats = st.slider("Number of top features to keep", 1, min(10, df_working.shape[1]-1), 5, key=f"feat_k_{idx}")
            X = df_working.drop(columns=[target])
            y = df_working[target]
            if X.select_dtypes(include=["int64", "float64"]).shape[1] > 0:
                selector = SelectKBest(score_func=f_classif, k=num_feats)
                X_new = selector.fit_transform(X.select_dtypes(include=["int64", "float64"]), y)
                selected_cols = X.select_dtypes(include=["int64", "float64"]).columns[selector.get_support()]
                df_working = pd.concat([df_working[selected_cols], df_working[target]], axis=1)
                st.success(f"✅ Selected top {num_feats} features: {list(selected_cols)}")
        
        elif step_type == "rename_columns":
            st.subheader("🔤 Rename Columns")
            old_names = st.multiselect("Select columns to rename", df_working.columns.tolist(), key=f"rename_old_{i}")
            new_names = []

            for j, old in enumerate(old_names):
                new_name = st.text_input(f"Rename '{old}' to:", key=f"rename_new_{i}_{j}")
                new_names.append(new_name)

            if st.button("Apply Renaming", key=f"apply_rename_{i}"):
                if len(old_names) == len(new_names):
                    rename_dict = dict(zip(old_names, new_names))
                    df_working.rename(columns=rename_dict, inplace=True)
                    st.success("✅ Columns renamed successfully")
                else:
                    st.warning("⚠️ Mismatch in number of columns and new names.")


        elif step_type == "balance_classes":
            st.subheader("⚖️ Balance Target Classes")
            target_col = st.selectbox("Select target column", df_working.columns, key=f"target_balance_{i}")
            
            if target_col:
                st.markdown("**📊 Class Distribution Before Balancing**")
                value_counts_before = df_working[target_col].value_counts().reset_index()
                value_counts_before.columns = [target_col, "Count"]
                st.dataframe(value_counts_before)
                    # Show histogram plot
                fig, ax = plt.subplots()
                sns.countplot(data=df_working, x=target_col, ax=ax)
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width() / 2, height + 0.5, f"{int(height)}", 
                            ha="center", va="bottom", fontsize=9)
                ax.set_title("Histogram: Class Distribution Before Balancing")
                ax.set_ylabel("Count")
                ax.set_xlabel(target_col)
                plt.xticks(rotation=45)  # Rotate labels if too long
                st.pyplot(fig)
                        
            if st.button("Apply Balancing", key=f"apply_balance_{i}"):
                try:
                    classes = df_working[target_col].value_counts().index
                    min_count = df_working[target_col].value_counts().min()
                    df_working = pd.concat([
                        resample(df_working[df_working[target_col] == cls], 
                                replace=True, 
                                n_samples=min_count, 
                                random_state=42)
                        for cls in classes
                    ])
                    df_working = df_working.sample(frac=1, random_state=42).reset_index(drop=True)
                    st.success("✅ Class balancing complete.")

                    st.markdown("**📊 Class Distribution After Balancing**")
                    value_counts_after = df_working[target_col].value_counts().reset_index()
                    value_counts_after.columns = [target_col, "Count"]
                    st.dataframe(value_counts_after)
                except Exception as e:
                    st.error(f"Error: {e}")

        elif step_type == "groupby_summary":
            group_col = st.selectbox("Select Grouping Column", df_working.columns, key=f"group_col_{idx}")
            agg_col = st.selectbox("Select Column to Aggregate", df_working.columns, key=f"agg_col_{idx}")
            agg_func = st.selectbox("Select Aggregation Function", ["mean", "sum", "count", "min", "max"], key=f"agg_func_{idx}")

            if group_col and agg_col:
                grouped_df = df_working.groupby(group_col)[agg_col].agg(agg_func).reset_index()
                st.dataframe(grouped_df)

        elif step_type == "pairplot":
            selected_cols = st.multiselect("Select columns for pairplot", df_working.select_dtypes(include=["int64", "float64"]).columns.tolist(), key=f"pair_{idx}")
            if selected_cols:
                fig = sns.pairplot(df_working[selected_cols])
                st.pyplot(fig)
        
        elif step_type == "distribution_stats":
            num_cols = df_working.select_dtypes(include=["int64", "float64"]).columns.tolist()
            stats_df = pd.DataFrame({
                "Skewness": df_working[num_cols].skew(),
                "Kurtosis": df_working[num_cols].kurt()
            })
            st.dataframe(stats_df)

        elif step_type == "unique_values":
            uniques = df_working.nunique().sort_values(ascending=False).reset_index()
            uniques.columns = ["Column", "Unique Values"]
            st.dataframe(uniques)

        elif step_type == "correlation_matrix":
            num_cols = df_working.select_dtypes(include=["int64", "float64"]).columns
            if len(num_cols) >= 2:
                corr_df = df_working[num_cols].corr().round(3)
                st.dataframe(corr_df)
            else:
                st.warning("Not enough numeric columns for correlation.")

        elif step_type == "describe_stats":
            desc_df = df_working.describe(include='all').transpose().round(3)
            st.dataframe(desc_df)


        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button(f"🗑️ Delete Step {idx + 1}", key=f"delete_step_{idx}"):
                st.session_state.eda_pipeline.pop(idx)
                st.rerun()  # Refresh UI after deletion

        with col2:
            if idx > 0 and st.button("🔼 Move Up", key=f"move_up_{idx}"):
                st.session_state.eda_pipeline[idx], st.session_state.eda_pipeline[idx - 1] = (
                    st.session_state.eda_pipeline[idx - 1],
                    st.session_state.eda_pipeline[idx],
                )
                st.experimental_rerun()

        with col3:
            if idx < len(st.session_state.eda_pipeline) - 1 and st.button("🔽 Move Down", key=f"move_down_{idx}"):
                st.session_state.eda_pipeline[idx], st.session_state.eda_pipeline[idx + 1] = (
                    st.session_state.eda_pipeline[idx + 1],
                    st.session_state.eda_pipeline[idx],
                )
                st.experimental_rerun()

    # Final output
    st.divider()
    st.markdown("### ✅ Final Processed Data")
    st.dataframe(df_working.head())
    csv = df_working.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Processed CSV", csv, "processed_data.csv", "text/csv")
