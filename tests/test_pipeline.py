"""
Test Suite for Churn Prediction Pipeline
Run: pytest tests/ -v
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a small synthetic DataFrame for testing."""
    from src.etl.generate_data import generate_churn_data
    return generate_churn_data(n_samples=200, random_state=0)


@pytest.fixture
def config():
    import yaml
    config_path = ROOT / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── ETL Tests ───────────────────────────────────────────────────────────────
class TestDataGeneration:
    def test_generates_correct_rows(self, sample_df):
        assert len(sample_df) == 200

    def test_required_columns_present(self, sample_df):
        required = ["customer_id", "tenure", "monthly_charges", "total_charges",
                    "contract_type", "payment_method", "churn"]
        for col in required:
            assert col in sample_df.columns, f"Missing column: {col}"

    def test_churn_binary(self, sample_df):
        assert set(sample_df["churn"].unique()).issubset({0, 1})

    def test_no_negative_tenure(self, sample_df):
        assert (sample_df["tenure"] >= 0).all()

    def test_no_negative_charges(self, sample_df):
        assert (sample_df["monthly_charges"] >= 0).all()
        assert (sample_df["total_charges"] >= 0).all()

    def test_customer_ids_unique(self, sample_df):
        assert sample_df["customer_id"].nunique() == len(sample_df)

    def test_churn_rate_reasonable(self, sample_df):
        """Churn rate should be between 5% and 50%."""
        rate = sample_df["churn"].mean()
        assert 0.05 <= rate <= 0.50, f"Unusual churn rate: {rate:.2%}"


class TestDataValidator:
    def test_valid_data_passes(self, sample_df):
        from src.etl.pipeline import DataValidator
        v = DataValidator()
        report = v.validate(sample_df)
        assert report["passed"] is True
        assert len(report["issues"]) == 0

    def test_missing_columns_caught(self):
        from src.etl.pipeline import DataValidator
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        v = DataValidator()
        report = v.validate(df)
        assert report["passed"] is False


class TestDataCleaner:
    def test_removes_duplicates(self, config, sample_df):
        from src.etl.pipeline import DataCleaner
        # Duplicate a row
        df_dup = pd.concat([sample_df, sample_df.head(5)], ignore_index=True)
        cleaner = DataCleaner(config)
        cleaned = cleaner.clean(df_dup)
        assert len(cleaned) == len(sample_df)

    def test_fills_nulls(self, config, sample_df):
        from src.etl.pipeline import DataCleaner
        df = sample_df.copy()
        df.loc[0:5, "monthly_charges"] = None
        cleaner = DataCleaner(config)
        cleaned = cleaner.clean(df)
        assert cleaned["monthly_charges"].isnull().sum() == 0


# ── Feature Engineering Tests ───────────────────────────────────────────────
class TestFeatureEngineering:
    def test_creates_new_features(self, sample_df):
        from src.features.engineering import engineer_features
        df_eng = engineer_features(sample_df)
        assert "charges_per_tenure" in df_eng.columns
        assert "services_count" in df_eng.columns
        assert "log_total_charges" in df_eng.columns

    def test_no_inf_values(self, sample_df):
        from src.features.engineering import engineer_features
        df_eng = engineer_features(sample_df)
        numeric = df_eng.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()

    def test_preprocessor_output_shape(self, sample_df, config):
        from src.features.engineering import prepare_data
        X, y, preprocessor = prepare_data(sample_df, config, fit=True)
        assert X.shape[0] == len(sample_df)
        assert y.shape[0] == len(sample_df)
        assert X.ndim == 2

    def test_preprocessor_transform_consistency(self, sample_df, config):
        from src.features.engineering import prepare_data
        X1, y1, preprocessor = prepare_data(sample_df, config, fit=True)
        X2, y2, _ = prepare_data(sample_df, config, preprocessor=preprocessor, fit=False)
        np.testing.assert_array_almost_equal(X1, X2)


# ── Model Tests ─────────────────────────────────────────────────────────────
class TestModelTraining:
    def test_model_trains_without_error(self, sample_df, config):
        from src.features.engineering import prepare_data
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier

        X, y, _ = prepare_data(sample_df, config, fit=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBClassifier(n_estimators=10, max_depth=3, verbosity=0, random_state=42)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert 0 <= score <= 1

    def test_prediction_probabilities_valid(self, sample_df, config):
        from src.features.engineering import prepare_data
        from xgboost import XGBClassifier

        X, y, _ = prepare_data(sample_df, config, fit=True)
        model = XGBClassifier(n_estimators=10, verbosity=0, random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_evaluate_model(self, sample_df, config):
        from src.features.engineering import prepare_data
        from src.training.train import evaluate_model
        from xgboost import XGBClassifier

        X, y, _ = prepare_data(sample_df, config, fit=True)
        model = XGBClassifier(n_estimators=10, verbosity=0, random_state=42)
        model.fit(X, y)
        metrics, _, _ = evaluate_model(model, X, y)
        for m in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            assert m in metrics
            assert 0 <= metrics[m] <= 1


# ── Monitoring Tests ─────────────────────────────────────────────────────────
class TestMonitoring:
    def test_drift_detector_no_drift(self, sample_df):
        from src.monitoring.monitor import DataDriftDetector
        numeric_cols = ["tenure", "monthly_charges", "total_charges"]
        detector = DataDriftDetector(sample_df[numeric_cols])
        report = detector.detect_drift(sample_df[numeric_cols])
        assert "feature_drift" in report
        assert "drifted_features" in report

    def test_drift_detector_detects_drift(self):
        from src.monitoring.monitor import DataDriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
        cur = pd.DataFrame({"x": np.random.normal(5, 1, 500)})  # Huge shift
        detector = DataDriftDetector(ref, threshold=0.1)
        report = detector.detect_drift(cur)
        assert "x" in report["drifted_features"]

    def test_performance_monitor_logs(self, tmp_path):
        from src.monitoring.monitor import PerformanceMonitor
        log_path = str(tmp_path / "perf_log.json")
        monitor = PerformanceMonitor(log_path)
        monitor.log_metrics({"roc_auc": 0.85, "f1": 0.72})
        trend = monitor.get_trend("roc_auc", last_n=5)
        assert "values" in trend


# ── API Tests (without running server) ──────────────────────────────────────
class TestAPISchemas:
    def test_customer_features_schema(self):
        from src.serving.api import CustomerFeatures
        customer = CustomerFeatures(
            customer_id="TEST001",
            tenure=12,
            monthly_charges=65.5,
            total_charges=786.0,
            contract_type="Month-to-month",
            payment_method="Electronic check",
            internet_service="Fiber optic",
        )
        assert customer.customer_id == "TEST001"
        assert customer.tenure == 12

    def test_invalid_tenure_rejected(self):
        from src.serving.api import CustomerFeatures
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            CustomerFeatures(
                customer_id="TEST",
                tenure=-5,  # Invalid
                monthly_charges=65.5,
                total_charges=786.0,
                contract_type="Month-to-month",
                payment_method="Electronic check",
                internet_service="Fiber optic",
            )
