import pandas as pd
def test_csv_exists():
    df = pd.read_csv("data/heart.csv")
    assert df.shape[0] > 200
def test_model_load():
    import joblib
    m = joblib.load("api/model.pkl")
    assert m is not None