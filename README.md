# Heart Disease MLOps Project

## Results
- ROC-AUC 0.90
- API Latency 5ms
- Prometheus metrics working

## How to Run
pip install -r requirements.txt
uvicorn api.main:app --reload

Open: http://127.0.0.1:8000/docs
Metrics: http://127.0.0.1:8000/metrics

## Sample
age 63, sex 1, cp 3 - prediction 1 (76%)

Author: Eswar - 2026
