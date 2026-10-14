from fastapi import FastAPI
from pydantic import BaseModel
import time, os, joblib
import pandas as pd
from prometheus_client import Counter, Histogram, make_asgi_app

app = FastAPI(title="Heart Disease MLOps API")

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
LATENCY = Histogram('api_latency_seconds', 'Request latency')
app.mount("/metrics", make_asgi_app())

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl")
model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease API running", "docs": "/docs", "metrics": "/metrics"}

@app.post("/predict")
def predict(data: InputData):
    start = time.time()
    REQUEST_COUNT.inc()
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0].max()
    LATENCY.observe(time.time() - start)
    return {
        "prediction": int(pred),
        "confidence": round(float(proba), 2),
        "latency": round(time.time() - start, 4),
        "risk": "High" if pred==1 else "Low"
    }