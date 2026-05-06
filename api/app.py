import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.config import MODEL_PATH, FEATURES, DEFAULT_THRESHOLD
from api.schemas import CreditApplication, PredictionOut

app = FastAPI(title="Credit Risk Scoring API")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        model = None

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOut)
def predict(applicant: CreditApplication):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training to generate artifacts/model.joblib",
        )

    X = pd.DataFrame([[getattr(applicant, f) for f in FEATURES]], columns=FEATURES)
    prob = float(model.predict_proba(X)[:, 1][0])
    decision = "ACCEPT" if prob < DEFAULT_THRESHOLD else "REJECT"
    return PredictionOut(prob_default=prob, decision=decision, threshold=DEFAULT_THRESHOLD)