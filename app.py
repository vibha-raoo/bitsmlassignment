"""
Streamlit app — Breast Cancer Classification Demo
Assignment 2 (Machine Learning) — M.Tech AIML/DSE

Features:
  a. Dataset upload option (CSV)
  b. Model selection dropdown
  c. Display of evaluation metrics
  d. Confusion matrix / classification report
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

st.set_page_config(page_title="Breast Cancer Classifier Demo", layout="wide")


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))


@st.cache_resource
def load_model(model_name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[model_name]))


@st.cache_data
def load_target_names():
    with open(os.path.join(MODEL_DIR, "target_names.json")) as f:
        return json.load(f)


st.title("🔬 Breast Cancer Classification — Model Comparison App")
st.write(
    "This app demonstrates 5 classification models trained on the "
    "**Breast Cancer Wisconsin (Diagnostic)** dataset (UCI / sklearn), "
    "569 instances, 30 numeric features, binary classification "
    "(malignant vs benign)."
)

# ---------------------------------------------------------------------------
# a. Dataset upload
# ---------------------------------------------------------------------------
st.header("1. Upload Test Data (CSV)")
st.caption(
    "Upload the provided `test_data.csv` (or any CSV with the same 30 feature "
    "columns plus a `target` column: 0 = malignant, 1 = benign)."
)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded data with shape {df.shape}")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("The uploaded CSV must contain a 'target' column with true labels.")
        st.stop()

    X = df.drop(columns=["target"])
    y_true = df["target"]

    # -----------------------------------------------------------------
    # b. Model selection dropdown
    # -----------------------------------------------------------------
    st.header("2. Select a Model")
    model_name = st.selectbox("Choose a classification model", list(MODEL_FILES.keys()))

    scaler = load_scaler()
    model = load_model(model_name)

    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        st.error(f"Could not scale uploaded features — check column names/order match "
                  f"the training data. Error: {e}")
        st.stop()

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # -----------------------------------------------------------------
    # c. Display evaluation metrics
    # -----------------------------------------------------------------
    st.header("3. Evaluation Metrics")
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("AUC", f"{auc:.3f}")
    c3.metric("Precision", f"{prec:.3f}")
    c4.metric("Recall", f"{rec:.3f}")
    c5.metric("F1 Score", f"{f1:.3f}")
    c6.metric("MCC", f"{mcc:.3f}")

    # -----------------------------------------------------------------
    # d. Confusion matrix + classification report
    # -----------------------------------------------------------------
    st.header("4. Confusion Matrix & Classification Report")
    target_names = load_target_names()

    col_left, col_right = st.columns(2)
    with col_left:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig)

    with col_right:
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3))

    # -----------------------------------------------------------------
    # Bonus: compare all models on the uploaded data
    # -----------------------------------------------------------------
    st.header("5. Compare All Models on This Data")
    if st.checkbox("Run all 5 models and compare"):
        rows = []
        for name in MODEL_FILES:
            m = load_model(name)
            yp = m.predict(X_scaled)
            ypr = m.predict_proba(X_scaled)[:, 1]
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_true, yp),
                "AUC": roc_auc_score(y_true, ypr),
                "Precision": precision_score(y_true, yp),
                "Recall": recall_score(y_true, yp),
                "F1": f1_score(y_true, yp),
                "MCC": matthews_corrcoef(y_true, yp),
            })
        st.dataframe(pd.DataFrame(rows).round(4))

else:
    st.info("👆 Upload a CSV file to get started. You can use the `test_data.csv` "
            "included in this repository.")
