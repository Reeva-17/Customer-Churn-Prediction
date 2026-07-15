"""
generate_scaler.py

Regenerates models/scaler.pkl.

The training notebook (notebooks/03_Preprocessing.ipynb) fits a StandardScaler
on the numeric columns ["tenure", "MonthlyCharges", "TotalCharges"] using the
80% training split (test_size=0.2, random_state=42, stratify=y) produced from
data/raw/Customer_Churn_Dataset.csv, but never persists the fitted scaler to
disk. This script reproduces that exact pipeline step-for-step so the fitted
StandardScaler matches the one implicitly used to train models/xgboost_model.pkl,
and saves it to models/scaler.pkl for use at inference time.

Run once from the repository root:

    python generate_scaler.py
"""

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, MODELS_DIR, SCALER_PATH, NUMERIC_COLS

RAW_DATA_PATH = DATA_DIR / "raw" / "Customer_Churn_Dataset.csv"


def build_training_split():
    """Reproduces notebooks/03_Preprocessing.ipynb exactly, up to (but not
    including) the scaling step, so we can fit a scaler on the correct rows."""

    df = pd.read_csv(RAW_DATA_PATH)

    # Convert TotalCharges to numeric (blank strings -> NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove rows with missing values
    df.dropna(inplace=True)

    # Drop customerID as it is not useful for prediction
    df.drop("customerID", axis=1, inplace=True)

    # Encode the target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Convert categorical features into numeric
    df = pd.get_dummies(df, drop_first=True)

    # Defining feature and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = build_training_split()

    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_COLS])

    joblib.dump(scaler, SCALER_PATH)

    print("Scaler fitted on columns:", NUMERIC_COLS)
    print("mean_ :", scaler.mean_)
    print("scale_:", scaler.scale_)
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()