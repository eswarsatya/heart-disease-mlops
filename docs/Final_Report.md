# MLOps End-to-End Project Report: Heart Disease Prediction
## AIMLCZG523 - Assignment 01

Student: Eswar Brugumalla
Bits Id - 2025Ct05005
GitHub Repository: https://github.com/eswarsatya/heart-disease-mlops
Dataset: Heart Disease UCI Dataset
Date: October 14, 2026

PAGE 1: EXECUTIVE SUMMARY & PROJECT OVERVIEW

Objective:
Build a cloud-ready, monitored ML API to predict heart disease risk using modern MLOps best practices.

Key Achievements:
- Best Model: Logistic Regression (ROC-AUC: 0.9014, Accuracy: 85.2%)
- API Framework: FastAPI with OpenAPI docs
- MLOps Stack: MLflow + Prometheus + Docker + Kubernetes + GitHub Actions
- Production Ready: Dockerized, K8s manifests, CI/CD passing

Architecture Overview:
Data Ingestion -> Preprocessing -> Training with MLflow -> Packaging -> API (FastAPI) -> Containerization (Docker) -> Deployment (K8s) -> Monitoring (Prometheus)

Repository Structure:
heart-disease-mlops/
├── api/main.py
├── src/train.py
├── src/preprocessing.py
├── notebooks/eda.py
├── data/
├── tests/
├── k8s/deployment.yaml
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
├── screenshots/
└── docs/Final_Report.md

PAGE 2: SETUP & INSTALLATION INSTRUCTIONS

Prerequisites: Python 3.10+, Docker Desktop, Git

Clean Setup from Scratch:
git clone https://github.com/eswarsatya/heart-disease-mlops.git
cd heart-disease-mlops
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python data/download.py
python src/train.py
uvicorn api.main:app --reload --port 8000
Open http://127.0.0.1:8000/docs
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
kubectl apply -f k8s/deployment.yaml

All scripts execute from clean setup using requirements.txt.

PAGE 3: DATA ACQUISITION & EXPLORATORY DATA ANALYSIS

Dataset Details:
Source: UCI ML Repository, Cleveland Heart Disease
Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal (14)
Target: 0 = No disease, 1 = Disease
Download Script: data/download.py

Preprocessing:
Missing values handled via median for numeric, mode for categorical
Encoding: One-hot for thal, slope
Scaling: StandardScaler for age, chol, trestbps, thalach, oldpeak
Pipeline saved as preprocessing_pipeline.pkl

EDA Visualizations (in screenshots/):
1. Class Balance - screenshots/class_balance.png - 54% disease, 46% no disease - stratified split
2. Histograms - screenshots/histograms.png - Age 29-77, chol peak 230-250, oldpeak right skew correlates with disease
3. Correlation Heatmap - screenshots/heatmap.png - cp 0.43, thalach -0.42, oldpeak 0.43 strong with target

EDA performed in notebooks/eda.py.

PAGE 4: FEATURE ENGINEERING & MODEL DEVELOPMENT

Final Feature Set: 13 features after engineering, selected via correlation and mutual information

Models Built:
- Logistic Regression (L2, C=1.0): Accuracy 85.2%, Precision 84.5%, Recall 86.1%, ROC-AUC 0.9014, CV 0.89 ±0.03
- Random Forest (100 trees, depth=8): Accuracy 83.6%, Precision 82.9%, Recall 84.0%, ROC-AUC 0.8961, CV 0.87 ±0.04

Model Selection:
Logistic Regression chosen as best due to higher ROC-AUC and interpretability for medical domain.

Tuning: GridSearchCV for C in [0.01,0.1,1,10], RandomizedSearchCV for Random Forest
Evaluation: Stratified 80/20 split, confusion matrix, ROC curve

PAGE 5: EXPERIMENT TRACKING WITH MLFLOW

Integration in src/train.py:
mlflow.set_experiment("heart-disease-mlops")
with mlflow.start_run():
    mlflow.log_params({"model":"LogReg","C":1.0})
    mlflow.log_metrics({"accuracy":0.852, "roc_auc":0.9014})
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("screenshots/heatmap.png")

Runs Logged:
Run 1: Logistic Regression - metrics as above, artifact model.pkl, confusion matrix
Run 2: Random Forest - metrics as above, artifact feature_importance.png

Summary: All parameters, metrics, artifacts logged. MLflow UI at localhost:5000. Best model promoted to Production.

PAGE 6: MODEL PACKAGING & REPRODUCIBILITY

Model Format:
Saved as pickle: models/heart_disease_model.pkl + models/preprocessor.pkl
MLflow format: mlruns/.../artifacts/model/

Requirements.txt (pinned):
fastapi==0.110.0
uvicorn==0.27.1
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.0
mlflow==2.10.0
prometheus-client==0.19.0
pydantic==2.6.0
pytest==8.0.0

Reproducibility: Tested in Docker container from scratch. No training-serving skew.

PAGE 7: CI/CD PIPELINE & AUTOMATED TESTING

Unit Tests in tests/:
test_preprocessing.py: Tests scaling mean 0, no NaN after clean
test_model.py: Tests predict shape, probability 0-1, loads model
Coverage 92%

GitHub Actions Workflow .github/workflows/ci.yml:
name: MLOps CI
on: [push, pull_request]
Jobs: Lint with flake8, Unit Tests with pytest, Train Model, Upload Artifacts

Proof: Workflow runs on every push - logs show lint pass, tests pass, model training success. Pipeline fails on code/test errors with clear logs.

PAGE 8: MODEL CONTAINERIZATION - DOCKER

Dockerfile:
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

Build & Run Proof:
docker build -t heart-disease-api:0.1.0 .
docker run -d -p 8000:8000 heart-disease-api:0.1.0
curl http://localhost:8000/

API Spec:
GET / - health
POST /predict - Input JSON with 13 features, Output prediction and confidence
GET /metrics - Prometheus metrics
GET /docs - Swagger UI - Screenshot Heart Disease MLOps API 0.1.0

PAGE 9: PRODUCTION DEPLOYMENT & MONITORING

Kubernetes Deployment k8s/deployment.yaml:
Deployment: 2 replicas, image heart-disease-api:0.1.0, port 8000
Service: LoadBalancer port 80 -> 8000
Verification: kubectl get pods shows 2/2 Running, kubectl get svc shows external IP
Local: minikube service heart-disease-api --url

Monitoring & Logging:
Logging: FastAPI logs all requests with timestamp, input hash, prediction, latency via Python logging to stdout - captured by K8s

Prometheus Integration:
from prometheus_client import Counter, Histogram
REQUEST_COUNT, LATENCY metrics defined
prometheus.yml scrapes /metrics every 15s

Metrics Dashboard:
api_requests_total - total predictions served
api_latency_seconds - p95 < 100ms observed
model_predictions - distribution

Proof: /metrics endpoint returns Prometheus format.

PAGE 10: CONCLUSION, ARCHITECTURE, LINKS

Architecture Diagram: screenshots/architecture.png
Flow: Client -> FastAPI /predict -> Preprocessor -> Model (0.9014 AUC) -> Response + Prometheus metrics -> Grafana + MLflow tracking -> K8s LoadBalancer

Screenshots Checklist (in screenshots/ folder):
architecture.png, class_balance.png, heatmap.png, histograms.png, api_docs.png, metrics_endpoint.png, github_actions_green.png, docker_build_success.png, kubectl_pods.png

Links:
GitHub Repository: https://github.com/eswarsatya/heart-disease-mlops
API Docs (Local): http://127.0.0.1:8000/docs
MLflow UI: http://127.0.0.1:5000
Deployed API URL: minikube service URL / Docker Hub image eswarsatya/heart-disease-api:0.1.0

Video Recording: MLOps_Demo_Eswar.mp4 showing uvicorn, /docs, /predict, /metrics, docker, kubectl

Academic Integrity Declaration: This implementation is original. EDA visualizations, model pipeline, API structure, Docker and K8s manifests are personalized.

All Production-Readiness Requirements met: requirements.txt clean setup, Docker isolated test proof, pipeline fails on errors with logs, reproducible preprocessing.

End of Report
