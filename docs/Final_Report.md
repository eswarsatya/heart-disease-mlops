# Heart Disease MLOps - Final Report (50 Marks)

## 1. Setup Instructions
pip install -r requirements.txt
python data/download.py
python src/train.py
uvicorn api.main:app --reload
API: http://127.0.0.1:8000/docs
Metrics: http://127.0.0.1:8000/metrics

## 2. Dataset
Heart Disease UCI, 303 rows, 14 features, Target 0/1

## 3. EDA [5 marks]
Class Balance 54% disease - screenshots/class_balance.png
Histograms - screenshots/histograms.png
Heatmap cp, thalach correlated - screenshots/heatmap.png
Missing values 0

## 4. Model [8 marks]
Logistic Regression Accuracy 0.85 ROC-AUC 0.9014 Best
Random Forest Accuracy 0.83 ROC-AUC 0.8961
Cross-validation 5-fold mean AUC 0.89

## 5. Experiment Tracking [5 marks]
MLflow used, params, metrics, model.pkl logged, Best run lr_auc_0.9014

## 6. Packaging [7 marks]
Saved models/model.pkl, requirements.txt complete, reproducible

## 7. CI/CD [8 marks]
GitHub Actions ci.yml, Lint Test Train, Both tests PASSED

## 8. Docker [5 marks]
Dockerfile exists, POST /predict returns prediction confidence

## 9. Kubernetes [7 marks]
k8s/deployment.yaml 2 replicas LoadBalancer

## 10. Monitoring [3 marks]
Prometheus /metrics, api_requests_total, latency 0.035s
Sample: prediction 0 confidence 0.79 latency 0.035 risk Low

## 11. Architecture
screenshots/architecture.png Flow User FastAPI Docker K8s Prometheus

## 12. Deliverables
Screenshots 4 images, API URL, GitHub link, Video OBS

Conclusion: End-to-end MLOps 0.90 ROC-AUC complete
