import pandas as pd
import numpy as np
import requests
import pickle
import os
from requests.auth import HTTPBasicAuth
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# =========================================================
# SAP ODATA CONFIG
# =========================================================
SAP_ODATA_HOST = "http://g09insahpp01.g09.fujitsu.local:8000"
SAP_ODATA_SERVICE_PATH = "/sap/opu/odata/SAP/ZMM_PO_ML_DATA_SRV"
SAP_ODATA_ENTITY_SET = "DATA_SET"
SAP_USERNAME = "vanarsem"
SAP_PASSWORD = "India!2345"
SAP_CLIENT = "100"

MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# FETCH DATA
# =========================================================
print("Fetching data from SAP OData...")

url = f"{SAP_ODATA_HOST}{SAP_ODATA_SERVICE_PATH}/{SAP_ODATA_ENTITY_SET}"
params = {"$format": "json", "sap-client": SAP_CLIENT}

r = requests.get(
    url,
    auth=HTTPBasicAuth(SAP_USERNAME, SAP_PASSWORD),
    params=params,
    verify=False
)
r.raise_for_status()

df = pd.DataFrame(r.json()["d"]["results"])
print(f"Records fetched: {len(df)}")

# =========================================================
# DATA NORMALIZATION
# =========================================================
df["PODate"] = pd.to_datetime(df["PODate"], dayfirst=True, errors="coerce")

for c in ["DelayDays", "QualityScore", "PriceIncreaseFlag", "Spend", "Quantity"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df["Status"] = (
    df["Status"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "late": 1,
        "delayed": 1,
        "on time": 0,
        "ontime": 0,
        "on-time": 0
    })
    .fillna(0)
)

# =========================================================
# FEATURE ENGINEERING (HISTORICAL AGGREGATION)
# =========================================================
features = (
    df
    .groupby(["Vendor", "Material", "Plant", "Purchasinggroup"])
    .agg(
        avg_delay=("DelayDays", "mean"),
        delay_std=("DelayDays", "std"),
        late_rate=("Status", "mean"),
        avg_quality=("QualityScore", "mean"),
        price_rate=("PriceIncreaseFlag", "mean"),
        total_spend=("Spend", "sum"),
        avg_qty=("Quantity", "mean")
    )
    .fillna(0)
    .reset_index()
)

# =========================================================
# LABELS (HISTORICAL RISK)
# =========================================================
labels = (
    df
    .groupby(["Vendor", "Material", "Plant", "Purchasinggroup"])
    .agg(
        late_risk=("Status", "max"),              # Has ever been late
        delay_days=("DelayDays", "mean"),
        price_risk=("PriceIncreaseFlag", "max"),
        quality_score=("QualityScore", "mean")
    )
    .reset_index()
)

dataset = features.merge(
    labels,
    on=["Vendor", "Material", "Plant", "Purchasinggroup"],
    how="inner"
)

print(f"Training samples: {len(dataset)}")

if dataset.empty:
    raise RuntimeError("No training data available after aggregation.")

# =========================================================
# MODEL FEATURES
# =========================================================
X_cols = [
    "avg_delay",
    "delay_std",
    "late_rate",
    "avg_quality",
    "price_rate",
    "total_spend",
    "avg_qty"
]

X = dataset[X_cols]

# =========================================================
# TRAIN MODELS
# =========================================================
print("Training ML models...")

late_clf = RandomForestClassifier(n_estimators=200, random_state=42)
late_clf.fit(X, dataset["late_risk"])
pickle.dump(late_clf, open(f"{MODEL_DIR}/late_delivery_clf.pkl", "wb"))

delay_reg = RandomForestRegressor(n_estimators=200, random_state=42)
delay_reg.fit(X, dataset["delay_days"])
pickle.dump(delay_reg, open(f"{MODEL_DIR}/delay_days_reg.pkl", "wb"))

price_clf = RandomForestClassifier(n_estimators=200, random_state=42)
price_clf.fit(X, dataset["price_risk"])
pickle.dump(price_clf, open(f"{MODEL_DIR}/price_inc_clf.pkl", "wb"))

quality_reg = RandomForestRegressor(n_estimators=200, random_state=42)
quality_reg.fit(X, dataset["quality_score"])
pickle.dump(quality_reg, open(f"{MODEL_DIR}/quality_reg.pkl", "wb"))

print("✅ Training complete. Models saved to ml_models/")