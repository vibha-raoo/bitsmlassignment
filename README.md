# Breast Cancer Classification — ML Assignment 2

## a. Problem Statement

The goal of this assignment is to build, evaluate, and deploy multiple
machine learning classification models that predict whether a breast
tumor is **malignant** or **benign** based on numeric features computed
from a digitized image of a fine needle aspirate (FNA) of a breast mass.
This is a **binary classification** problem. Five classifiers are trained
on the same dataset, evaluated with six standard metrics, and made
interactively explorable through a Streamlit web application deployed on
Streamlit Community Cloud.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also available as a
  built-in loader in scikit-learn: `sklearn.datasets.load_breast_cancer`,
  which mirrors the original UCI dataset)
- **Instances:** 569
- **Features:** 30 numeric, real-valued features (mean, standard error,
  and "worst"/largest value of 10 measurements per cell nucleus, e.g.
  radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension)
- **Target classes:** `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign
- **Train/test split:** 75% / 25%, stratified by class, `random_state=42`
- Features were standardized (`StandardScaler`) before training.

## c. GitHub Repository Link

> `<PASTE YOUR GITHUB REPO LINK HERE>`

Repository structure:
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- test_data.csv
│-- model/
    │-- train_models.py
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- random_forest_ensemble.joblib
    │-- scaler.joblib
    │-- target_names.json
    │-- metrics.csv
```

## d. Models Used

All 5 models were trained on the **same** train/test split of the
Breast Cancer Wisconsin dataset described above.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9860 | 0.9977 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |
| Decision Tree | 0.9231 | 0.9234 | 0.9540 | 0.9222 | 0.9379 | 0.8378 |
| kNN | 0.9790 | 0.9845 | 0.9677 | 1.0000 | 0.9836 | 0.9555 |
| Naive Bayes | 0.9371 | 0.9878 | 0.9355 | 0.9667 | 0.9508 | 0.8644 |
| Random Forest (Ensemble) | 0.9580 | 0.9950 | 0.9565 | 0.9778 | 0.9670 | 0.9098 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer on this dataset. The classes are close to linearly separable after scaling, so a simple linear decision boundary works extremely well, giving the highest accuracy, F1, and MCC of all five models. |
| Decision Tree | Weakest performer. A single unpruned tree overfits the training data and generalizes less well than the ensemble/linear methods, resulting in the lowest accuracy and MCC. |
| kNN | Very strong performer, achieving perfect recall (1.0) on the test set — it never misses a benign case — but precision is slightly lower than Logistic Regression, meaning it occasionally misclassifies a malignant case as benign-leaning. Performance is sensitive to feature scaling, which was applied here. |
| Naive Bayes | Middling accuracy but the second-highest AUC, showing its predicted probabilities rank cases well even though its hard classification threshold is less accurate. The independence assumption between features (which are correlated in this dataset, e.g. radius/perimeter/area) limits its ceiling. |
| Random Forest (Ensemble) | Strong, well-balanced performer with very high AUC (0.995), close to Logistic Regression. Averaging many trees reduces the overfitting seen in the single Decision Tree and gives more stable predictions. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it achieves the best or near-best score on every single metric (Accuracy, AUC, Precision, Recall, F1, MCC), indicating the dataset's classes are highly linearly separable after standardization. |

## Live Streamlit App

> `<PASTE YOUR DEPLOYED STREAMLIT APP LINK HERE>`

### App Features
- **Dataset upload (CSV):** Upload `test_data.csv` (or any CSV with the
  same 30 feature columns + a `target` column).
- **Model selection dropdown:** Choose from Logistic Regression, Decision
  Tree, kNN, Naive Bayes, or Random Forest.
- **Evaluation metrics display:** Accuracy, AUC, Precision, Recall, F1,
  MCC shown live for the selected model on the uploaded data.
- **Confusion matrix & classification report:** Visual heatmap plus a
  full per-class classification report table.
- **Bonus:** "comparing all 5 models" view on the same
  uploaded data.

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # retrains models & regenerates test_data.csv (optional, already included)
streamlit run app.py
```

## How This Was Deployed

1. Pushed this repository to GitHub (public repo, includes
   `requirements.txt`).
2. Went to [streamlit.io/cloud](https://streamlit.io/cloud) and signed in
   with GitHub.
3. Clicked **New App**, selected this repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Clicked **Deploy**.
