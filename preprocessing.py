"""
preprocessing.py

Turns a single raw customer record (as collected from the Streamlit form)
into the exact feature representation the trained XGBoost model expects.

This mirrors notebooks/03_Preprocessing.ipynb:
    1. pd.get_dummies(df, drop_first=True)  -> one-hot encoding
    2. StandardScaler on ["tenure", "MonthlyCharges", "TotalCharges"]

Because a single-row input does not contain every category value, we cannot
call pd.get_dummies() directly on it (it would only ever produce a column for
whichever value is present). Instead we build the one-hot vector explicitly
from config.CATEGORICAL_OPTIONS / config.BASELINE_CATEGORY, which were
extracted directly from the training data and validated against the header
of data/processed/X_train.csv.
"""

import pandas as pd

from config import (
    FEATURE_COLUMNS,
    NUMERIC_COLS,
    CATEGORICAL_OPTIONS,
    BASELINE_CATEGORY,
)


def validate_raw_input(raw_input: dict) -> None:
    """Raises ValueError if raw_input is missing fields or has invalid values."""

    required_numeric = set(NUMERIC_COLS) | {"SeniorCitizen"}
    required_categorical = set(CATEGORICAL_OPTIONS.keys())
    required_fields = required_numeric | required_categorical

    missing = required_fields - set(raw_input.keys())
    if missing:
        raise ValueError(f"Missing required input fields: {sorted(missing)}")

    for field, options in CATEGORICAL_OPTIONS.items():
        if raw_input[field] not in options:
            raise ValueError(
                f"Invalid value {raw_input[field]!r} for field {field!r}. "
                f"Expected one of {options}."
            )

    if raw_input["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1.")

    for field in NUMERIC_COLS:
        value = raw_input[field]
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{field} must be a non-negative number.")


def encode_input(raw_input: dict) -> pd.DataFrame:
    """
    Converts a raw customer dict into a single-row DataFrame with columns
    exactly matching FEATURE_COLUMNS (unscaled numeric values, one-hot
    encoded categoricals). Column order matches the trained model.
    """

    validate_raw_input(raw_input)

    row = {col: 0 for col in FEATURE_COLUMNS}

    row["SeniorCitizen"] = int(raw_input["SeniorCitizen"])
    row["tenure"] = float(raw_input["tenure"])
    row["MonthlyCharges"] = float(raw_input["MonthlyCharges"])
    row["TotalCharges"] = float(raw_input["TotalCharges"])

    for field in CATEGORICAL_OPTIONS:
        value = raw_input[field]
        if value == BASELINE_CATEGORY[field]:
            continue
        column_name = f"{field}_{value}"
        if column_name in row:
            row[column_name] = 1
        else:
            raise ValueError(
                f"Unexpected category value {value!r} for field {field!r}: "
                f"no matching column {column_name!r} in FEATURE_COLUMNS."
            )

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df


def scale_input(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Applies the fitted StandardScaler to the numeric columns only,
    returning a new DataFrame (does not mutate the input)."""

    scaled_df = df.copy()
    scaled_df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    return scaled_df


def preprocess_input(raw_input: dict, scaler) -> pd.DataFrame:
    """Full pipeline: raw dict -> one-hot encoded, scaled, model-ready row."""

    df = encode_input(raw_input)
    return scale_input(df, scaler)