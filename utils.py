"""
utils.py

Shared helper functions for the Streamlit app: loading artifacts (model,
scaler, reports), running predictions with the F1-optimal decision
threshold, categorizing risk, and computing live SHAP explanations for a
single customer.
"""

import joblib
import pandas as pd
import shap
import streamlit as st

from config import (
    MODEL_PATH,
    SCALER_PATH,
    MODEL_COMPARISON_PATH,
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
)
from preprocessing import preprocess_input


@st.cache_resource(show_spinner=False)
def load_model():
    """Loads the trained XGBoost model (models/xgboost_model.pkl)."""
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler():
    """Loads the fitted StandardScaler (models/scaler.pkl)."""
    return joblib.load(SCALER_PATH)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    """Loads the model comparison table (reports/model_comparison.csv)."""
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    """Builds a SHAP TreeExplainer for the given model.

    The leading underscore on `_model` tells Streamlit's cache_resource not
    to try to hash the (unhashable) model object.
    """
    return shap.TreeExplainer(_model)


def predict_churn(model, scaler, raw_input: dict, threshold: float = DECISION_THRESHOLD):
    """
    Runs the full pipeline for a single customer:
        raw_input -> preprocess -> model.predict_proba -> threshold

    Returns a dict with:
        probability   : float, predicted probability of churn (class 1)
        prediction    : int, 1 = churn, 0 = no churn (using `threshold`)
        risk_level    : str, "Low" / "Medium" / "High"
        input_row     : pd.DataFrame, the scaled, model-ready feature row
                         (useful for SHAP explanations)
    """

    input_row = preprocess_input(raw_input, scaler)
    probability = float(model.predict_proba(input_row[FEATURE_COLUMNS])[0, 1])
    prediction = int(probability >= threshold)
    risk_level = get_risk_level(probability)

    return {
        "probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "input_row": input_row,
    }


def get_risk_level(probability: float) -> str:
    """Buckets a churn probability into a human-readable risk level."""
    if probability < 0.30:
        return "Low"
    elif probability < DECISION_THRESHOLD:
        return "Medium"
    else:
        return "High"


def explain_prediction(model, input_row: pd.DataFrame):
    """
    Computes SHAP values for a single, already-preprocessed customer row
    using shap.TreeExplainer, matching the approach used in
    notebooks/10_SHAP.ipynb.

    Returns a shap.Explanation object for the single row (index 0), suitable
    for shap.plots.waterfall().
    """
    explainer = get_shap_explainer(model)
    shap_values = explainer(input_row[FEATURE_COLUMNS])
    return shap_values[0]


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"