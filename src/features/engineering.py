"""
Feature Engineering: Transform cleaned data into ML-ready features.
"""
import logging
import os
import joblib
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new derived features from raw columns."""
    logger.info("Engineering features...")
    df = df.copy()

    # Charges per unit tenure (avoid divide by zero)
    df["charges_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)

    # Ratio of total to expected charges
    expected = df["monthly_charges"] * (df["tenure"] + 1)
    df["total_to_expected_ratio"] = df["total_charges"] / (expected + 1)

    # Engagement score: number of active services
    service_cols = ["has_tech_support", "has_streaming", "has_online_security", "has_online_backup"]
    existing_service_cols = [c for c in service_cols if c in df.columns]
    df["services_count"] = df[existing_service_cols].sum(axis=1)

    # High value customer flag
    df["is_high_value"] = ((df["monthly_charges"] > 80) & (df["tenure"] > 24)).astype(int)

    # Risk indicators
    df["short_tenure_flag"] = (df["tenure"] < 6).astype(int)
    df["long_tenure_flag"] = (df["tenure"] > 48).astype(int)

    # Support burden
    df["high_support_burden"] = (df["num_support_tickets"] > 3).astype(int)

    # Paperless + electronic check combo (known churn risk)
    if "paperless_billing" in df.columns and "payment_method" in df.columns:
        df["digital_risk_combo"] = (
            (df["paperless_billing"].astype(int) == 1) &
            (df["payment_method"] == "Electronic check")
        ).astype(int)

    # Log-transform skewed numeric features
    df["log_total_charges"] = np.log1p(df["total_charges"])
    df["log_monthly_charges"] = np.log1p(df["monthly_charges"])

    logger.info(f"Feature engineering complete. New shape: {df.shape}")
    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Build sklearn ColumnTransformer for numeric + categorical features."""
    cfg = config["features"]

    numeric_features = cfg["numeric_features"] + [
        "charges_per_tenure",
        "total_to_expected_ratio",
        "services_count",
        "is_high_value",
        "short_tenure_flag",
        "long_tenure_flag",
        "high_support_burden",
        "digital_risk_combo",
        "log_total_charges",
        "log_monthly_charges",
    ]

    categorical_features = cfg["categorical_features"]

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    config: dict,
    preprocessor=None,
    fit: bool = True,
) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Full feature preparation: engineer + preprocess.
    Returns X_processed, y, preprocessor
    """
    target = config["features"]["target_column"]

    df = engineer_features(df)

    y = df[target].values
    df = df.drop(columns=[target, "customer_id"], errors="ignore")

    if preprocessor is None:
        preprocessor = build_preprocessor(config)

    if fit:
        X = preprocessor.fit_transform(df)
        logger.info("Fitted preprocessor on training data.")
    else:
        X = preprocessor.transform(df)
        logger.info("Transformed data with existing preprocessor.")

    logger.info(f"Final feature matrix: {X.shape}")
    return X, y, preprocessor


def get_feature_names(preprocessor: ColumnTransformer, config: dict) -> list:
    """Extract feature names after transformation."""
    cfg = config["features"]
    numeric_features = cfg["numeric_features"] + [
        "charges_per_tenure", "total_to_expected_ratio", "services_count",
        "is_high_value", "short_tenure_flag", "long_tenure_flag",
        "high_support_burden", "digital_risk_combo",
        "log_total_charges", "log_monthly_charges",
    ]

    try:
        cat_encoder = preprocessor.named_transformers_["cat"]["encoder"]
        cat_names = cat_encoder.get_feature_names_out(cfg["categorical_features"]).tolist()
    except Exception:
        cat_names = []

    return numeric_features + cat_names


if __name__ == "__main__":
    import yaml

    config = load_config()
    df = pd.read_csv(config["data"]["processed_path"])
    X, y, preprocessor = prepare_data(df, config, fit=True)

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Saved preprocessor to models/preprocessor.pkl")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
