"""
Train one Logistic Regression model per disease.
Run:  python train.py
Put the downloaded CSV files inside the data/ folder (see DATASETS below).
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------
# 1. Dataset settings. Change file names / target columns if your CSV differs.
#    to_binary: function that turns the target column into 1 (disease) / 0 (healthy)
# ---------------------------------------------------------------
DATASETS = {
    "Breast Cancer": dict(csv=None, target="target", drop=[],
                          to_binary=lambda s: 1 - s),            # built into sklearn
    "Diabetes": dict(csv="data/diabetes.csv", target="Outcome", drop=[],
                     to_binary=lambda s: s.astype(int)),
    "Heart Disease": dict(csv="data/heart.csv", target="target", drop=[],
                          to_binary=lambda s: s.astype(int)),
    "Kidney Disease": dict(csv="data/kidney_disease.csv", target="classification", drop=["id"],
                           to_binary=lambda s: s.astype(str).str.strip().str.lower().eq("ckd").astype(int)),
    "Liver Disease": dict(csv="data/indian_liver_patient.csv", target="Dataset", drop=[],
                          to_binary=lambda s: (s == 1).astype(int)),   # 1 = disease, 2 = no disease
}


def load_data(name, cfg):
    if cfg["csv"] is None:
        from sklearn.datasets import load_breast_cancer
        return load_breast_cancer(as_frame=True).frame
    return pd.read_csv(cfg["csv"])


def clean(df):
    """Generic cleaning: strip text, treat '?' as missing, convert number-like text to numbers."""
    df = df.copy()
    df = df.drop_duplicates()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan, "": np.nan})
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= 0.8 * df[col].notna().sum():
            df[col] = converted
    return df


def build_pipeline(X):
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def main():
    os.makedirs("models", exist_ok=True)
    results = []
    for name, cfg in DATASETS.items():
        if cfg["csv"] and not os.path.exists(cfg["csv"]):
            print(f"[skip] {name}: {cfg['csv']} not found")
            continue

        df = clean(load_data(name, cfg))
        df = df.dropna(subset=[cfg["target"]])
        y = cfg["to_binary"](df[cfg["target"]])
        X = df.drop(columns=[cfg["target"]] + cfg["drop"], errors="ignore")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        model = build_pipeline(X_tr).fit(X_tr, y_tr)
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]

        print(f"\n===== {name} =====")
        print("Confusion matrix:\n", confusion_matrix(y_te, pred))
        print(classification_report(y_te, pred, target_names=["Healthy", "Disease"]))

        results.append(dict(Disease=name,
                            Accuracy=accuracy_score(y_te, pred),
                            Precision=precision_score(y_te, pred),
                            Recall=recall_score(y_te, pred),
                            F1=f1_score(y_te, pred),
                            ROC_AUC=roc_auc_score(y_te, prob)))

        # Save what the app needs to build input forms
        meta = {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(X[c]):
                meta[c] = dict(type="num", min=float(X[c].min()), max=float(X[c].max()),
                               default=float(X[c].median()))
            else:
                cats = sorted(X[c].dropna().unique().tolist())
                meta[c] = dict(type="cat", options=cats, default=cats[0])
        joblib.dump(dict(model=model, meta=meta), f"models/{name.replace(' ', '_').lower()}.joblib")

    table = pd.DataFrame(results).round(3)
    print("\n===== SUMMARY =====\n", table.to_string(index=False))
    table.to_csv("results_summary.csv", index=False)


if __name__ == "__main__":
    main()
