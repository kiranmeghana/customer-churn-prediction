.PHONY: install pipeline api dashboard mlflow test docker-up docker-down clean

# Install dependencies
install:
	pip install -r requirements.txt

# Run full ML pipeline
pipeline:
	python scripts/run_pipeline.py

# Start FastAPI server
api:
	uvicorn src.serving.api:app --reload --port 8000

# Start Streamlit dashboard
dashboard:
	streamlit run dashboard/app.py

# Start MLflow UI
mlflow:
	mlflow ui --backend-store-uri mlflow_tracking/ --port 5000

# Run all tests
test:
	pytest tests/ -v --tb=short

# Test coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Docker full stack
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# Generate sample data only
data:
	python src/etl/generate_data.py

# Clean generated artifacts
clean:
	rm -rf models/*.pkl models/*.json models/plots/
	rm -rf data/processed/
	rm -f pipeline.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

# Quick demo — run everything and print summary
demo: install pipeline
	@echo ""
	@echo "======================================"
	@echo "  Pipeline complete!"
	@echo "  Start API:       make api"
	@echo "  Dashboard:       make dashboard"
	@echo "  MLflow UI:       make mlflow"
	@echo "======================================"
