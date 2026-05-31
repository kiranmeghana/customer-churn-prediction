# 🚀 End-to-End ML Pipeline: Customer Churn Prediction

A production-grade ML system from raw data to deployed REST API with monitoring dashboard.

## 📁 Project Structure

```
churn_prediction/
├── data/
│   ├── raw/                    # Raw CSV data
│   └── processed/              # Processed features
├── src/
│   ├── etl/                    # Data ingestion & cleaning
│   ├── features/               # Feature engineering
│   ├── training/               # Model training & evaluation
│   ├── serving/                # FastAPI REST endpoint
│   └── monitoring/             # Model monitoring utilities
├── models/                     # Saved model artifacts
├── dashboard/                  # Streamlit dashboard
├── tests/                      # Unit & integration tests
├── docker/                     # Docker configs
├── mlflow_tracking/            # MLflow experiment tracking
├── scripts/                    # Helper scripts
├── notebooks/                  # Jupyter exploration
├── requirements.txt
├── docker-compose.yml
└── Makefile
```

## ⚡ Quick Start (VS Code)

### 1. Prerequisites
```
Python 3.10+ required
Docker Desktop (optional, for containerized run)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline (One Command)
```bash
python scripts/run_pipeline.py
```

This will:
- Generate synthetic churn data (~10,000 customers)
- Run ETL cleaning & validation
- Engineer features (scaling, encoding, interactions)
- Train XGBoost + LightGBM models with cross-validation
- Evaluate with SHAP explainability plots
- Save model artifacts to models/
- Log everything to MLflow

### 4. Start the API
```bash
uvicorn src.serving.api:app --reload --port 8000
```
Visit: http://localhost:8000/docs  (Swagger UI)

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```
Visit: http://localhost:8501

### 6. Start MLflow UI
```bash
mlflow ui --backend-store-uri mlflow_tracking/ --port 5000
```
Visit: http://localhost:5000

---

## 🐳 Docker (Full Stack)
```bash
docker-compose up --build
```
Services:
- API:       http://localhost:8000
- Dashboard: http://localhost:8501
- MLflow:    http://localhost:5000

---

## 🔌 API Endpoints

| Method | Endpoint          | Description              |
|--------|-------------------|--------------------------|
| GET    | /health           | Health check             |
| POST   | /predict          | Single prediction        |
| POST   | /predict/batch    | Batch predictions        |
| GET    | /model/info       | Model metadata           |
| GET    | /metrics          | Model performance metrics|

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST001",
    "tenure": 12,
    "monthly_charges": 65.5,
    "total_charges": 786.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "num_products": 2,
    "has_tech_support": false,
    "has_streaming": true,
    "has_online_security": false,
    "has_online_backup": true,
    "paperless_billing": true,
    "gender": "Male",
    "senior_citizen": "No",
    "num_support_tickets": 3,
    "avg_call_duration": 8.5,
    "days_since_last_contact": 45
  }'
```

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

## 📊 Tech Stack
| Layer       | Technology                         |
|-------------|-------------------------------------|
| Models      | XGBoost, LightGBM, SHAP            |
| Pipeline    | Scikit-learn, Pandas, NumPy        |
| Serving     | FastAPI, Uvicorn, Pydantic         |
| Tracking    | MLflow                              |
| Dashboard   | Streamlit, Plotly                  |
| Containers  | Docker, Docker Compose             |
| Testing     | Pytest                             |
