# Fraud MLOps Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-grade fraud detection system with 5-model ensemble, SHAP explainability, real-time feature engineering, versioned model deployment, PSI drift monitoring, and shadow deployment support.

---

## Key Features

### ML Pipeline
- **5-Model Ensemble**: Random Forest + XGBoost + LightGBM + AdaBoost + Logistic Regression with AUC-weighted soft voting
- **Online/Offline Feature Split**: 50+ engineered features with deterministic single-row inference (no batch dependency)
- **SHAP Explainability**: Per-transaction top-5 risk factors with TreeExplainer across all tree models
- **Proper CV with SMOTE**: StratifiedKFold with fold-internal SMOTE (no data leakage per Kaggle gold standard)

### Production Infrastructure
- **Versioned Model Deployment**: Automatic versioning, archiving, and manifest tracking with production/shadow dual slots
- **Model Rollback**: One-request version switching via `/reload-models?version=X&role=production`
- **Drift Monitoring**: PSI/KS statistical tests with prediction distribution tracking
- **Prediction Audit Log**: SQLite log of every prediction with model version for delayed label joining
- **Load Testing**: Locust script with realistic transaction simulation

### API & Dashboard
- **FastAPI**: Async inference with per-step latency profiling
- **Streamlit Dashboard**: Transaction monitor + Model Health page (version history, PSI gauges, alert summary)
- **Batch Prediction**: Up to 1000 transactions per request
- **Shadow Deployment Ready**: Dual model loading with A/B comparison logging

---

## Measured Performance

*Benchmarked on Intel i7, 16GB RAM, Windows 10, single process, 200 iterations.*

### Inference Latency (single transaction, measured locally)

| Metric | Value |
|---|---|
| Feature Transform (p50) | 90 ms |
| Model Inference (p50) | 50 ms |
| Total /predict (p50) | 139 ms |
| SHAP Overhead (p50) | 12 ms |
| Single-thread Throughput | 7 req/s |

### Model Performance (full 284k dataset, 5-fold CV with fold-internal SMOTE)

| Metric | Value |
|---|---|
| Accuracy | 0.9988 |
| Precision | 0.5791 |
| Recall | 1.0000 |
| F1 Score | 0.7334 |
| ROC-AUC | 0.9999 |
| Optimal Threshold | 0.680 |

| Sub-Model | OOF AUC |
|---|---|
| Random Forest | 0.9814 |
| XGBoost | 0.9792 |
| LightGBM | 0.9807 |
| AdaBoost | 0.9769 |

### Per-Step Latency (single transaction)

```
anomaly         17.5ms  (Isolation Forest + Mahalanobis)
statistical     14.6ms  (V-column PCA + statistics)
missing          9.3ms  (column-wise null handling)
interaction      8.3ms  (V-feature cross products)
scaling          7.3ms  (RobustScaler transform)
amount           5.8ms  (log/sqrt/boxcox/bin)
time             4.5ms  (cyclical encoding + flags)
```

---

## Architecture

```
Transaction Request
       |
       v
  FastAPI /predict
       |
       ├──> Feature Transform (online path, stored params)
       |      ├── Time features (cyclical encoding)
       |      ├── Amount features (boxcox with saved lambda)
       |      ├── Statistical features (PCA with saved model)
       |      ├── Clustering (KMeans with column alignment)
       |      └── Velocity (Redis sliding window or zero-fill)
       |
       ├──> Ensemble Inference (AUC-weighted soft vote)
       |      RF(25%) + XGB(25%) + LGB(25%) + Ada(25%)
       |
       ├──> SHAP Explain (TreeExplainer x 3)
       |      Returns top-5 risk factors with SHAP values
       |
       ├──> Audit Log (SQLite)
       |      transaction_id, prob, risk, model_version, timestamp
       |
       └──> Response
              fraud_probability, risk_level, top_risk_factors
```

---

## Model Training Methodology

The initial production model was trained on the full Kaggle Credit Card Fraud dataset (284,807 transactions, 492 frauds, 0.17% fraud rate). Key design decisions:

**Data preprocessing.** Extreme IQR outliers were removed from V14, V12, and V10 (the features most negatively correlated with the fraud class). Thresholds were computed on a balanced 50/50 subsample to avoid majority-class dominance. This removed 0.59% of transactions.

**Feature engineering.** The pipeline generates 50+ features from 31 raw columns: cyclical time encoding, log/sqrt/Box-Cox amount transforms, per-row V-column statistics, Isolation Forest anomaly scores, K-Means cluster assignments, and pairwise V-feature interactions. Velocity features (rolling-window aggregates) are zero-filled at inference time and replaced by Redis real-time computation in production.

**Handling class imbalance.** Three complementary strategies: (1) `class_weight='balanced'` for tree-based models and `scale_pos_weight` for XGBoost/LightGBM, (2) SMOTE oversampling of the minority class, and — critically — (3) SMOTE applied **inside each CV fold**, not before the split. Applying SMOTE before cross-validation leaks synthetic samples into validation folds and inflates metrics; this is the most common mistake flagged by Kaggle gold-medal notebooks on this dataset.

**Ensemble design.** Five models (Random Forest, XGBoost, LightGBM, AdaBoost, Logistic Regression) are trained independently. The ensemble uses AUC-weighted soft voting: each model's fraud probability is weighted by its 5-fold OOF AUC, then averaged. The optimal decision threshold (0.68) was found by sweeping [0.1, 0.9] to maximize F1 on the full dataset.

**Why recall is prioritized over precision.** In fraud detection, a false negative (missing a fraudulent transaction) costs far more than a false positive (blocking a legitimate one, which can be reversed). The ensemble achieves 100% recall at the cost of lower precision (58%). The threshold can be adjusted to trade off precision vs. recall based on business requirements.

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
docker-compose up -d

# Services:
# API Docs:    http://localhost:8000/docs
# Dashboard:   http://localhost:8501
# Health:      http://localhost:8000/health
```

### Option 2: Local Development

```bash
# Setup
python -m venv fraud_env
source fraud_env/bin/activate  # Windows: fraud_env\Scripts\activate
pip install -r requirements.txt

# Download data
# Place creditcard.csv from Kaggle into data/raw/

# Train models
python src/models/train.py --n-samples 50000

# Start API
uvicorn src.api.fraud_api:app --reload

# Start dashboard (new terminal)
streamlit run src/monitoring/dashboard.py
```

---

## API Usage

### Single Prediction

```python
import requests

transaction = {
    "Time": 50000, "Amount": 149.62,
    "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
    # ... V5-V28
    "transaction_id": "TXN_001"
}

response = requests.post("http://localhost:8000/predict", json=transaction)
result = response.json()

print(f"Fraud Probability: {result['fraud_probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
print("Top Risk Factors:")
for f in result.get("top_risk_factors", []):
    print(f"  {f['feature']}: SHAP={f['impact']:.4f} ({f['direction']})")
```

### Batch Prediction

```python
batch = {"transactions": [tx1, tx2, tx3]}  # Up to 1000
response = requests.post("http://localhost:8000/predict/batch", json=batch)
```

### Model Version Management

```bash
# List all versions
curl http://localhost:8000/monitoring/versions

# Roll back to a previous version
curl -X POST "http://localhost:8000/reload-models?version=v20260501_120000&role=production"

# Enable shadow deployment
curl -X POST "http://localhost:8000/reload-models?version=v20260508_120000&role=shadow"
```

---

## Project Structure

```
fraud-mlops-platform/
├── data/
│   ├── raw/creditcard.csv          # Real credit card fraud dataset
│   └── predictions.db              # Prediction audit log (SQLite)
├── models/
│   ├── feature_engineering_pipeline.pkl   # Active production pipeline
│   ├── fraud_detection_ensemble.pkl       # Active production ensemble
│   ├── reference_data.csv                 # Drift detection baseline
│   ├── version_manifest.json              # Version registry
│   └── archive/                           # Historical model versions
├── src/
│   ├── data_processing/
│   │   ├── generate_data.py         # Synthetic data generator + real data loader
│   │   ├── data_loader.py           # Real dataset loader
│   │   ├── feature_engineering.py   # Online/offline feature pipeline
│   │   ├── outlier_handler.py       # IQR-based outlier removal
│   │   └── online_features.py       # Redis sliding window features
│   ├── models/
│   │   ├── fraud_detector.py        # 5 model classes + Ensemble
│   │   ├── explainer.py             # SHAP FraudExplainer
│   │   ├── version_manager.py       # Version manifest + rollback
│   │   └── train.py                 # Training script with MLflow
│   ├── api/
│   │   ├── fraud_api.py             # FastAPI application
│   │   └── auth.py                  # Redis rate limiter
│   └── monitoring/
│       ├── dashboard.py             # Streamlit dashboard
│       └── model_monitor.py         # PSI/KS drift detection
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_models.py
│   ├── test_api.py
│   └── load_test.py                 # Locust load test
├── docker/
│   └── Dockerfile                   # Multi-stage build
├── docker-compose.yml
└── requirements.txt
```

---

## Model Training

```bash
# Full training on all 284k transactions
python src/models/train.py

# Train on 50k sample with model comparison
python src/models/train.py --n-samples 50000 --compare-models

# Train with 5-fold CV (no data leakage)
python src/models/train.py --n-samples 50000 --cv-folds 5

# Disable MLflow tracking
python src/models/train.py --no-mlflow
```

## Load Testing

```bash
# Start API first, then:
locust -f tests/load_test.py --host=http://localhost:8000
# Open http://localhost:8089 in browser
```

## Technology Stack

| Category | Tools |
|---|---|
| ML Models | XGBoost, LightGBM, scikit-learn (RF, AdaBoost, LR) |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI + Pydantic v2 + uvicorn |
| Dashboard | Streamlit + Plotly |
| Monitoring | PSI, KS-test, prediction distribution tracking |
| Versioning | Custom manifest-based with archive |
| Infrastructure | Docker, Docker Compose, Redis |
| Testing | pytest, Locust |

## License

MIT
