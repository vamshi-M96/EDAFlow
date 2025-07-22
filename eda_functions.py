import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def dropna(df):
    df.dropna(inplace=True)
    return df

def labelencode(df):
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale(df, method="standard"):
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def split(df):
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[[target_col]]
    return X, y
