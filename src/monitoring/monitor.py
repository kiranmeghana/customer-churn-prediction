"""
Model Monitoring: Drift detection, performance tracking, alerting.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect distribution shift between reference and current data."""

    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.1):
        self.reference = reference_data
        self.threshold = threshold
        self.numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = reference_data.select_dtypes(include=["object"]).columns.tolist()

    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """Run KS test for numeric features, chi-square for categorical."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "num_reference": len(self.reference),
            "num_current": len(current_data),
            "feature_drift": {},
            "drifted_features": [],
            "overall_drift": False,
        }

        # Numeric: KS test
        for col in self.numeric_cols:
            if col not in current_data.columns:
                continue
            ks_stat, p_value = stats.ks_2samp(
                self.reference[col].dropna(),
                current_data[col].dropna(),
            )
            drift_detected = p_value < 0.05 or ks_stat > self.threshold
            report["feature_drift"][col] = {
                "type": "ks_test",
                "statistic": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "drift_detected": drift_detected,
            }
            if drift_detected:
                report["drifted_features"].append(col)

        # Categorical: chi-square
        for col in self.cat_cols:
            if col not in current_data.columns:
                continue
            try:
                ref_counts = self.reference[col].value_counts()
                cur_counts = current_data[col].value_counts()
                all_cats = set(ref_counts.index) | set(cur_counts.index)
                ref_freq = np.array([ref_counts.get(c, 0) for c in all_cats])
                cur_freq = np.array([cur_counts.get(c, 0) for c in all_cats])
                # Normalize
                ref_freq = ref_freq / ref_freq.sum()
                cur_freq = cur_freq / cur_freq.sum()
                # Chi-square
                chi2 = np.sum((cur_freq - ref_freq) ** 2 / (ref_freq + 1e-10))
                drift_detected = chi2 > self.threshold
                report["feature_drift"][col] = {
                    "type": "chi_square_approx",
                    "statistic": round(float(chi2), 4),
                    "drift_detected": drift_detected,
                }
                if drift_detected:
                    report["drifted_features"].append(col)
            except Exception:
                pass

        report["overall_drift"] = len(report["drifted_features"]) > 0
        drift_pct = len(report["drifted_features"]) / max(1, len(report["feature_drift"]))
        report["drift_percentage"] = round(drift_pct * 100, 1)

        return report


class PerformanceMonitor:
    """Track model performance over time."""

    def __init__(self, metrics_log_path: str = "models/performance_log.json"):
        self.log_path = metrics_log_path
        self._load_log()

    def _load_log(self):
        if Path(self.log_path).exists():
            with open(self.log_path) as f:
                self.log = json.load(f)
        else:
            self.log = []

    def _save_log(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(self.log, f, indent=2)

    def log_metrics(self, metrics: Dict, period: str = "daily"):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "period": period,
            **metrics,
        }
        self.log.append(entry)
        self._save_log()
        logger.info(f"Logged performance metrics: {metrics}")

    def get_trend(self, metric: str = "roc_auc", last_n: int = 10) -> Dict:
        if not self.log:
            return {"metric": metric, "values": [], "trend": "insufficient_data"}

        values = [e.get(metric) for e in self.log[-last_n:] if metric in e]
        if len(values) < 2:
            return {"metric": metric, "values": values, "trend": "insufficient_data"}

        slope = (values[-1] - values[0]) / len(values)
        trend = "improving" if slope > 0.001 else "degrading" if slope < -0.001 else "stable"

        return {
            "metric": metric,
            "values": values,
            "current": values[-1],
            "trend": trend,
            "slope": round(slope, 6),
        }

    def check_alerts(self, baseline_metrics: Dict, current_metrics: Dict,
                     threshold: float = 0.05) -> List[str]:
        alerts = []
        for metric in ["roc_auc", "f1", "accuracy"]:
            if metric in baseline_metrics and metric in current_metrics:
                drop = baseline_metrics[metric] - current_metrics[metric]
                if drop > threshold:
                    alerts.append(
                        f"ALERT: {metric} dropped by {drop:.4f} "
                        f"(baseline={baseline_metrics[metric]:.4f}, "
                        f"current={current_metrics[metric]:.4f})"
                    )
        return alerts


def generate_monitoring_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    baseline_metrics: Optional[Dict] = None,
) -> Dict:
    """Generate a full monitoring report."""
    drift_detector = DataDriftDetector(reference_df)
    drift_report = drift_detector.detect_drift(current_df)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "data_drift": drift_report,
        "alerts": [],
    }

    if drift_report["overall_drift"]:
        report["alerts"].append(
            f"DATA DRIFT detected in {len(drift_report['drifted_features'])} features: "
            f"{drift_report['drifted_features']}"
        )

    monitor = PerformanceMonitor()
    if baseline_metrics:
        trend = monitor.get_trend("roc_auc")
        report["performance_trend"] = trend

    return report


if __name__ == "__main__":
    import pandas as pd

    # Demo monitoring
    df = pd.read_csv("data/processed/features.csv")
    reference = df.sample(5000, random_state=42)
    current = df.sample(1000, random_state=99)

    report = generate_monitoring_report(reference, current)
    print(json.dumps(report, indent=2, default=str))
