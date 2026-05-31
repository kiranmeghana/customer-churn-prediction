"""
Full ML Pipeline Runner — Run everything end-to-end with one command.
Usage: python scripts/run_pipeline.py
"""
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def step(name: str, fn, *args, **kwargs):
    logger.info(f"\n{SEPARATOR}")
    logger.info(f"STEP: {name}")
    logger.info(SEPARATOR)
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    logger.info(f"✅ {name} completed in {elapsed:.1f}s")
    return result


def main():
    logger.info(f"\n{'#'*60}")
    logger.info("# Customer Churn Prediction Pipeline")
    logger.info(f"{'#'*60}\n")

    total_start = time.time()

    # ── Step 1: Generate Data ──────────────────────────────────────
    def generate_data():
        os.makedirs("data/raw", exist_ok=True)
        raw_path = "data/raw/customers.csv"
        if Path(raw_path).exists():
            import pandas as pd
            existing = pd.read_csv(raw_path)
            logger.info(f"Found existing data: {len(existing)} rows. Regenerating...")
        from src.etl.generate_data import generate_churn_data
        df = generate_churn_data(n_samples=10000, random_state=42)
        df.to_csv(raw_path, index=False)
        logger.info(f"Saved {len(df)} rows to {raw_path}")
        return df

    step("Generate Synthetic Data", generate_data)

    # ── Step 2: ETL Pipeline ───────────────────────────────────────
    def run_etl():
        from src.etl.pipeline import run_etl
        return run_etl()

    df_clean = step("ETL Pipeline (Load, Validate, Clean)", run_etl)

    # ── Step 3: Feature Engineering Preview ───────────────────────
    def preview_features():
        import pandas as pd
        from src.features.engineering import engineer_features
        df = pd.read_csv("data/processed/features.csv")
        df_eng = engineer_features(df)
        new_cols = set(df_eng.columns) - set(df.columns)
        logger.info(f"New engineered features: {sorted(new_cols)}")
        return df_eng

    step("Feature Engineering Preview", preview_features)

    # ── Step 4: Train Models ───────────────────────────────────────
    def train():
        from src.training.train import run_training
        return run_training()

    best_model, results = step("Model Training (XGBoost + LightGBM)", train)

    # ── Step 5: Verify Artifacts ───────────────────────────────────
    def verify():
        required = [
            "models/best_model.pkl",
            "models/preprocessor.pkl",
            "models/metrics_summary.json",
        ]
        for p in required:
            exists = Path(p).exists()
            status = "✅" if exists else "❌"
            size = f"({Path(p).stat().st_size / 1024:.1f} KB)" if exists else ""
            logger.info(f"  {status} {p} {size}")
        missing = [p for p in required if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(f"Missing artifacts: {missing}")

    step("Verify Artifacts", verify)

    # ── Summary ────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info(f"\n{'#'*60}")
    logger.info("# PIPELINE COMPLETE!")
    logger.info(f"# Total time: {total_elapsed:.1f}s")
    logger.info(f"{'#'*60}")
    logger.info("\n📊 Model Results:")
    for model_name, metrics in results.items():
        logger.info(f"  {model_name}: AUC={metrics['roc_auc']:.4f} | F1={metrics['f1']:.4f} | Acc={metrics['accuracy']:.4f}")

    logger.info("\n🚀 Next Steps:")
    logger.info("  1. Start API:       uvicorn src.serving.api:app --reload --port 8000")
    logger.info("  2. Open Swagger:    http://localhost:8000/docs")
    logger.info("  3. Dashboard:       streamlit run dashboard/app.py")
    logger.info("  4. MLflow UI:       mlflow ui --backend-store-uri mlflow_tracking/ --port 5000")
    logger.info("  5. Run tests:       pytest tests/ -v")


if __name__ == "__main__":
    main()
