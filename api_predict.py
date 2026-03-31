from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

app = FastAPI(title="Supplier Risk Prediction API")

MODEL_DIR = "ml_models"

# -------------------------------------------------
# Load models at startup
# -------------------------------------------------
try:
    late_clf = pickle.load(open(f"{MODEL_DIR}/late_delivery_clf.pkl", "rb"))
    delay_reg = pickle.load(open(f"{MODEL_DIR}/delay_days_reg.pkl", "rb"))
    price_clf = pickle.load(open(f"{MODEL_DIR}/price_inc_clf.pkl", "rb"))
    quality_reg = pickle.load(open(f"{MODEL_DIR}/quality_reg.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

# -------------------------------------------------
# EXACT feature list used during training
# -------------------------------------------------
FEATURE_COLS = [
    "avg_delay",
    "delay_std",
    "late_rate",
    "avg_quality",
    "price_rate",
    "total_spend",
    "avg_qty"
]

# -------------------------------------------------
# Request schema (forces validation)
# -------------------------------------------------
class PredictRequest(BaseModel):
    avg_delay: float
    delay_std: float
    late_rate: float
    avg_quality: float
    price_rate: float
    total_spend: float
    avg_qty: float


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        # Convert request → DataFrame
        data = pd.DataFrame([req.dict()])

        # Ensure correct feature order
        X = data[FEATURE_COLS]

        return {
            "late_delivery_probability": float(late_clf.predict_proba(X)[0][1]),
            "expected_delay_days": float(delay_reg.predict(X)[0]),
            "price_increase_probability": float(price_clf.predict_proba(X)[0][1]),
            "predicted_quality_score": float(quality_reg.predict(X)[0])
        }

    except Exception as e:
        # Return readable error instead of silent 500
        raise HTTPException(status_code=500, detail=str(e))