import mlflow, joblib, os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from src.preprocessing import get_preprocessor, load_data

os.makedirs("api", exist_ok=True)
X, y = load_data()
mlflow.set_experiment("Heart-Disease-MLOps")

for name, model in [("LogisticRegression", LogisticRegression(max_iter=1000)), ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42))]:
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([('pre', get_preprocessor()), ('model', model)])
        scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
        pipe.fit(X,y)
        mlflow.log_param("model", name)
        mlflow.log_metric("roc_auc", scores.mean())
        joblib.dump(pipe, f"api/{name}.pkl")
        print(f"{name} ROC-AUC: {scores.mean():.4f}")

pipe = Pipeline([('pre', get_preprocessor()), ('model', RandomForestClassifier())])
pipe.fit(X,y)
joblib.dump(pipe, "api/model.pkl")
print("Saved api/model.pkl")