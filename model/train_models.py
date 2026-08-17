"""
train_models.py
----------------
Trains 5 classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
evaluates them with 6 metrics, saves the trained models (for the Streamlit app),
and writes out test_data.csv (used for the app's CSV-upload demo) plus a
metrics comparison table (metrics.csv) that feeds the README.

Dataset source: UCI Machine Learning Repository / scikit-learn built-in loader
(sklearn.datasets.load_breast_cancer) — originally from UCI ML Repository,
"Breast Cancer Wisconsin (Diagnostic)" dataset, 569 instances, 30 numeric features.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant, 1 = benign
target_names = list(data.target_names)

print(f"Dataset shape: {X.shape}, classes: {target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# Scale features (helps Logistic Regression / KNN in particular)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 2. Define models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE
    ),
}

results = []
os.makedirs(os.path.join(HERE), exist_ok=True)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    # Save model
    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".joblib"
    joblib.dump(model, os.path.join(HERE, fname))
    print(f"Saved {fname}")

# Save the scaler too (app needs it to transform uploaded CSVs)
joblib.dump(scaler, os.path.join(HERE, "scaler.joblib"))

# ---------------------------------------------------------------------------
# 3. Save metrics comparison table
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df[["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
metrics_df.to_csv(os.path.join(HERE, "metrics.csv"), index=False)
print("\nComparison table:\n", metrics_df.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Save test_data.csv — features + true label, for the Streamlit upload demo
#    (kept small since Streamlit Community Cloud free tier has limited resources)
# ---------------------------------------------------------------------------
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
print(f"\nSaved test_data.csv with shape {test_df.shape}")

# Save target name mapping for the app
with open(os.path.join(HERE, "target_names.json"), "w") as f:
    json.dump(target_names, f)

print("\nDone.")
