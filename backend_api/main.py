from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import json

# ---------- Load model, scaler, and column info ----------

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# with open("columns.json", "r") as f:
#     FEATURE_COLUMNS = json.load(f)

FEATURE_COLUMNS = ["amt", "city_pop", "trans_hour", "trans_day_of_week", "time_diff", "distance_km", "category_food_dining", "category_gas_transport", "category_grocery_net", "category_grocery_pos", "category_health_fitness", "category_home", "category_kids_pets", "category_misc_net", "category_misc_pos", "category_personal_care", "category_shopping_net", "category_shopping_pos", "category_travel", "gender_M", "state_AL", "state_AR", "state_AZ", "state_CA", "state_CO", "state_CT", "state_DC", "state_FL", "state_GA", "state_HI", "state_IA", "state_ID", "state_IL", "state_IN", "state_KS", "state_KY", "state_LA", "state_MA", "state_MD", "state_ME", "state_MI", "state_MN", "state_MO", "state_MS", "state_MT", "state_NC", "state_ND", "state_NE", "state_NH", "state_NJ", "state_NM", "state_NV", "state_NY", "state_OH", "state_OK", "state_OR", "state_PA", "state_RI", "state_SC", "state_SD", "state_TN", "state_TX", "state_UT", "state_VA", "state_VT", "state_WA", "state_WI", "state_WV", "state_WY"]

# with open("numerical_cols.json", "r") as f:
#     NUMERIC_COLS = json.load(f)
NUMERIC_COLS  = ["amt", "city_pop", "trans_hour", "trans_day_of_week", "time_diff", "distance_km"]

# These are the categorical columns used in training
CATEGORICAL_COLS = ["category", "gender", "state"]

# ---------- FastAPI app ----------

app = FastAPI(title="Fraud Detection API")

# Allow Next.js (frontend) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request body model ----------

class TransactionInput(BaseModel):
    amt: float
    city_pop: float
    trans_hour: int
    trans_day_of_week: int
    time_diff: float
    distance_km: float
    category: str
    gender: str
    state: str

# ---------- Helper function to prepare features ----------

def build_features(data: TransactionInput) -> pd.DataFrame:
    # Convert input to DataFrame
    d = data.dict()
    df = pd.DataFrame([d])

    # One-hot encode categorical variables (same as training)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Add any missing columns from training as 0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model knows
    df = df[FEATURE_COLUMNS]

    # Scale numerical columns
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    return df

# ---------- Routes ----------

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running 🚀"}

@app.post("/predict")
def predict_fraud(tx: TransactionInput):
    print(tx)
    features = build_features(tx)
    proba = model.predict_proba(features)[0, 1]
    pred = int(model.predict(features)[0])

    return {
        "fraud": bool(pred == 1),
        "prediction": pred,
        "score": float(proba * 100.0)  # send 0–100 risk score
    }


# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import pandas as pd
# import pickle

# # ---------- Load model & scaler ----------

# # Load best ensemble model (RandomForest / AdaBoost / XGBoost)
# with open("fraud_model_ensemble.pkl", "rb") as f:
#     model = pickle.load(f)

# # Load scaler used during training
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # Exact feature names used when fitting the model
# FEATURE_COLUMNS = list(getattr(model, "feature_names_in_", []))

# # Exact feature names used when fitting the scaler
# SCALER_FEATURES = list(getattr(scaler, "feature_names_in_", []))

# # Categorical columns you accept from frontend
# CATEGORICAL_COLS = ["category", "gender"]

# # ---------- FastAPI app ----------

# app = FastAPI(title="Fraud Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------- Request body model ----------

# class TransactionInput(BaseModel):
#     amt: float
#     city_pop: float
#     trans_hour: int
#     trans_day_of_week: int
#     time_diff: float
#     distance_km: float
#     category: str
#     gender: str

# # ---------- Helper: build feature dataframe ----------

# def build_features(data: TransactionInput) -> pd.DataFrame:
#     # Base dataframe from request
#     d = data.dict()
#     df = pd.DataFrame([d])

#     # One-hot encode categorical vars
#     df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

#     # ---- Add missing model feature columns in ONE GO (no fragmentation) ----
#     missing_model_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
#     if missing_model_cols:
#         zeros_model = pd.DataFrame(0, index=df.index, columns=missing_model_cols)
#         df = pd.concat([df, zeros_model], axis=1)

#     # Keep only model’s feature columns
#     df = df[FEATURE_COLUMNS]

#     # ---- Scale using the exact columns scaler was trained on ----
#     if SCALER_FEATURES:
#         missing_scaler_cols = [c for c in SCALER_FEATURES if c not in df.columns]
#         if missing_scaler_cols:
#             zeros_scaler = pd.DataFrame(0.0, index=df.index, columns=missing_scaler_cols)
#             df = pd.concat([df, zeros_scaler], axis=1)

#         df[SCALER_FEATURES] = scaler.transform(df[SCALER_FEATURES])

#     return df

# # ---------- Routes ----------

# @app.get("/")
# def root():
#     return {"message": "Fraud Detection API is running 🚀"}

# @app.post("/predict")
# def predict_fraud(tx: TransactionInput):
#     features = build_features(tx)

#     proba = model.predict_proba(features)[0, 1]
#     pred = int(model.predict(features)[0])

#     response = {
#         "fraud": bool(pred == 1),
#         "prediction": pred,
#         "score": float(proba * 100.0)  # 0–100 risk score
#     }

#     # Debug log – you’ll see this in the terminal
#     print("Prediction response:", response)

#     return response
