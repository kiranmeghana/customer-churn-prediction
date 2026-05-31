"""
ETL Pipeline: Load, validate, and clean raw customer data.
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataLoader:
    def __init__(self, raw_path: str):
        self.raw_path = raw_path

    def load(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.raw_path}")
        if not Path(self.raw_path).exists():
            raise FileNotFoundError(
                f"Data file not found: {self.raw_path}\n"
                "Run: python src/etl/generate_data.py"
            )
        df = pd.read_csv(self.raw_path)
        logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
        return df


class DataValidator:
    REQUIRED_COLUMNS = [
        "customer_id", "tenure", "monthly_charges", "total_charges",
        "contract_type", "payment_method", "internet_service", "churn",
    ]

    def validate(self, df: pd.DataFrame) -> dict:
        report = {"passed": True, "issues": []}

        # Check required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            report["issues"].append(f"Missing columns: {missing}")
            report["passed"] = False

        # Check nulls
        null_pct = df.isnull().mean()
        high_null = null_pct[null_pct > 0.3].to_dict()
        if high_null:
            report["issues"].append(f"High null columns (>30%): {high_null}")

        # Check target column
        if "churn" in df.columns:
            unique_vals = df["churn"].unique()
            if not set(unique_vals).issubset({0, 1}):
                report["issues"].append(f"Target 'churn' has unexpected values: {unique_vals}")
                report["passed"] = False

        # Check numeric ranges
        if "tenure" in df.columns:
            if (df["tenure"] < 0).any():
                report["issues"].append("Negative tenure values found")

        if "monthly_charges" in df.columns:
            if (df["monthly_charges"] < 0).any():
                report["issues"].append("Negative monthly_charges found")

        logger.info(f"Validation: {'PASSED' if report['passed'] else 'FAILED'}")
        for issue in report["issues"]:
            logger.warning(f"  Issue: {issue}")

        return report


class DataCleaner:
    def __init__(self, config: dict):
        self.config = config

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning...")
        df = df.copy()

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["customer_id"])
        logger.info(f"Dropped {before - len(df)} duplicate customers")

        # Handle missing values — numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col != "churn" and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"  Filled {col} nulls with median={median_val:.2f}")

        # Handle missing values — categorical
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"  Filled {col} nulls with mode={mode_val}")

        # Clip extreme outliers
        if "monthly_charges" in df.columns:
            df["monthly_charges"] = df["monthly_charges"].clip(0, 200)
        if "total_charges" in df.columns:
            df["total_charges"] = df["total_charges"].clip(0, 15000)
        if "tenure" in df.columns:
            df["tenure"] = df["tenure"].clip(0, 100)

        # Convert boolean columns to int
        bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
        for col in bool_cols:
            df[col] = df[col].astype(int)

        logger.info(f"Cleaning complete. Final shape: {df.shape}")
        return df


def run_etl(config_path: str = "config.yaml") -> pd.DataFrame:
    config = load_config(config_path)

    loader = DataLoader(config["data"]["raw_path"])
    validator = DataValidator()
    cleaner = DataCleaner(config)

    df = loader.load()
    report = validator.validate(df)

    if not report["passed"]:
        logger.error("Validation failed. Fix issues before continuing.")
        raise ValueError(f"Data validation failed: {report['issues']}")

    df_clean = cleaner.clean(df)

    # Save cleaned data
    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv(config["data"]["processed_path"], index=False)
    logger.info(f"Saved cleaned data to {config['data']['processed_path']}")

    return df_clean


if __name__ == "__main__":
    run_etl()
