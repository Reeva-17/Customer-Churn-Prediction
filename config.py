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