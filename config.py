"""
Configuration file for Customer Churn Prediction App
"""

from pathlib import Path

# ----------------------------
# Project Directories
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images"

# ----------------------------
# Model Files
# ----------------------------

MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# ----------------------------
# Report Files
# ----------------------------

MODEL_COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"

# ----------------------------
# SHAP Images
# ----------------------------

SHAP_BAR = IMAGES_DIR / "shap_bar.png"
SHAP_SUMMARY = IMAGES_DIR / "shap_summary.png"
SHAP_WATERFALL = IMAGES_DIR / "shap_waterfall.png"

# ----------------------------
# Raw Data
# ----------------------------

RAW_DATA_PATH = DATA_DIR / "raw" / "Customer_Churn_Dataset.csv"

# ----------------------------
# Decision Threshold
# ----------------------------
# notebooks/08_XGBoost.ipynb searched thresholds 0.30-0.70 in steps of 0.01
# and found the F1-optimal cut-off to be 0.60 (Best F1 : 0.6299). The shipped
# model must use this threshold instead of the sklearn default of 0.5.

DECISION_THRESHOLD = 0.60

# ----------------------------
# Numeric Columns (scaled with StandardScaler)
# ----------------------------

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# ----------------------------
# Trained Feature Order
# ----------------------------
# Exact column order of data/processed/X_train.csv, produced by
# pd.get_dummies(df, drop_first=True) in notebooks/03_Preprocessing.ipynb.
# The model expects features in exactly this order.

FEATURE_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]

# ----------------------------
# Categorical Field Options
# ----------------------------
# Raw category values as they appear in data/raw/Customer_Churn_Dataset.csv,
# in the order Yes/No/etc. appear in the source data. The "baseline" value
# for each field is the one that pd.get_dummies(drop_first=True) drops (the
# first category in alphabetical order) and therefore has no corresponding
# column in FEATURE_COLUMNS above.

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# Baseline (dropped) category per field - i.e. the value that produces an
# all-zero one-hot encoding for that field.
BASELINE_CATEGORY = {
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "No",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
}

# Sensible slider/input bounds for numeric fields, based on the raw dataset.
NUMERIC_RANGES = {
    "tenure": {"min": 0, "max": 72, "default": 12},
    "MonthlyCharges": {"min": 18.0, "max": 120.0, "default": 70.0},
    "TotalCharges": {"min": 0.0, "max": 9000.0, "default": 840.0},
}