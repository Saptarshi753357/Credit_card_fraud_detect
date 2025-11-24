import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conint, confloat
import pandas as pd

# Paths (relative to this file when running uvicorn from backend/)
MODEL_PATH = os.getenv("MODEL_PATH", "./models/fraud_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "./models/scaler.pkl")
TEMPLATE_PATH = os.getenv("TEMPLATE_PATH", "./template_cols.json")

app = FastAPI(title="Fraud Detection (local)")

# Allow local frontend dev origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_artifacts():
    global model, scaler, template_cols, numerical_cols, categorical_cols

    # load model
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

    # load scaler
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load scaler at {SCALER_PATH}: {e}")

    # load template cols
    if os.path.exists(TEMPLATE_PATH):
        import json
        with open(TEMPLATE_PATH, "r") as fh:
            template_cols = json.load(fh)
    else:
        template_cols = None

    # define numeric/categorical columns used in training
    numerical_cols = ['amt','city_pop','trans_hour','trans_day_of_week','time_diff','distance_km']
    categorical_cols = ['category','gender','state']


class PredictIn(BaseModel):
    amt: confloat(ge=0)
    city_pop: confloat(ge=0)
    trans_hour: conint(ge=0, le=23)
    trans_day_of_week: conint(ge=0, le=6)
    time_diff: confloat(ge=0)
    distance_km: confloat(ge=0)
    category: str = Field(default="unknown")
    gender: str = Field(default="U")
    state: str = Field(default="XX")


class PredictOut(BaseModel):
    is_fraud: int
    probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    # DataFrame from payload
    X = pd.DataFrame([payload.dict()])

    # One-hot encode categorical cols (same approach as training)
    X = pd.get_dummies(X, columns=categorical_cols)

    # Align to template columns saved during training
    if template_cols:
        for c in template_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[template_cols]

    # Scale numerical columns
    try:
        X[numerical_cols] = scaler.transform(X[numerical_cols])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scaling inputs: {e}")

    proba = float(model.predict_proba(X)[:, 1][0])
    pred = int(model.predict(X)[0])

    return PredictOut(is_fraud=pred, probability=round(proba, 6))