# ⚙️ EDAFlow – Visual, Modular Exploratory Data Analysis App

**EDAFlow** is an intuitive Streamlit-based application for performing end-to-end Exploratory Data Analysis (EDA) with zero code. Built for data scientists, ML engineers, and analysts, it lets you upload a dataset, drag-and-drop preprocessing steps, and explore insights visually.

---

## 🚀 Features

- 🗂️ **Upload Support**: CSV and Excel (XLSX) files  
- 🧼 **Clean Data**: Handle missing values, drop duplicates  
- 📊 **Descriptive Stats**: Summary stats, skewness, kurtosis  
- 📉 **Outlier Detection**: IQR-based outlier filtering  
- 🔁 **Encode & Scale**: Label encoding, standardization, min-max scaling  
- 🧠 **Feature Selection**: SelectKBest, correlation, and PPS  
- 📈 **Visualizations**: Histogram, pairplot, heatmap, boxplot  
- 🔄 **Class Balancing**: Upsample or downsample imbalanced data  
- 🧮 **PPS Matrix**: Predictive Power Score analysis  
- 📌 **GroupBy Tools**: Interactive group-wise summaries  
- 🔍 **Top-N Analyzer**: View most frequent & unique values  
- 📄 **Export Options**: Save cleaned data and profiling reports  

---

## 🖥️ How to Use

1. **Upload your dataset** (CSV or Excel).  
2. **Build your EDA pipeline** step-by-step using the sidebar.  
3. **Visualize outputs** – charts, stats, and results update dynamically.  
4. **Download cleaned data** and reports in one click.  

> No coding required – just drag, drop, explore! 🙌

---

## 📦 Installation

```bash
git clone https://github.com/vamshi-M96/EDAFlow.git
cd EDAFlow
pip install -r requirements.txt
streamlit run edaflow_app.py

