import os
import io
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "pv_degradation_model.pkl")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = FastAPI(title="Solar & Battery Fault Detection System")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

MODELS = None
EVENT_LOG = [] # Acts as our Fault Events Database
CONSECUTIVE_FAULTS = 0 # For the Live Persistence Check
PERSISTENCE_THRESHOLD = 3

@app.on_event("startup")
def load_models():
    global MODELS
    if os.path.exists(MODEL_PATH):
        MODELS = joblib.load(MODEL_PATH)
        print(f"[OK] Loaded ML model.")

class Reading(BaseModel):
    string_id: int = Field(1, ge=1, le=2)
    voltage: float
    current: float
    irradiance: float
    temperature: float
    battery_voltage: float
    battery_current: float
    battery_temp: float

def build_features(v, i, irr, t):
    power = v * i
    pr = power / (irr + 1e-6)
    res = v / (i + 1e-6)
    return np.array([[v, i, irr, t, power, pr, res]], dtype=float)

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/logs")
def get_logs():
    return JSONResponse(EVENT_LOG[::-1])

@app.post("/api/predict")
def predict(reading: Reading):
    global CONSECUTIVE_FAULTS
    if MODELS is None: raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Solar ML Prediction
    model = MODELS[f"s{reading.string_id}"]
    X = build_features(reading.voltage, reading.current, reading.irradiance, reading.temperature)
    prob = float(model.predict_proba(X)[0][1])
    ml_status = "Degraded" if prob >= 0.4 else "Normal"
    
    # 2. Battery Threshold Rules (Chapter 1 Objective 3)
    battery_status = "Normal"
    if reading.battery_voltage < 10.5:
        battery_status = "Fault (Low Voltage)"
    elif reading.battery_temp > 55.0:
        battery_status = "Fault (Overheating)"
        
    # 3. Persistence Check Logic (Chapter 1 Objective 5)
    is_abnormal = (ml_status == "Degraded" or battery_status != "Normal")
    
    if is_abnormal:
        CONSECUTIVE_FAULTS += 1
    else:
        CONSECUTIVE_FAULTS = 0 # Reset if normal
        
    confirmed_fault = False
    if CONSECUTIVE_FAULTS >= PERSISTENCE_THRESHOLD:
        confirmed_fault = True
        
        # Log to Database if it's a newly confirmed fault
        if not EVENT_LOG or EVENT_LOG[-1].get("timestamp") != datetime.now().strftime("%H:%M:%S"):
            EVENT_LOG.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "string_id": reading.string_id,
                "status": "Confirmed Fault",
                "ml_status": ml_status,
                "battery_status": battery_status,
                "voltage": round(reading.voltage, 2),
                "battery_v": round(reading.battery_voltage, 2)
            })

    return {
        "ml_status": ml_status, 
        "battery_status": battery_status,
        "confirmed_fault": confirmed_fault,
        "consecutive": CONSECUTIVE_FAULTS,
        "prob_degraded": prob
    }

@app.post("/api/stream")
async def stream_csv(string_id: int = 1, file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), sep=";")
    
    V_col, I_col, T_col, label_col = f"V{string_id}", f"I{string_id}", f"T{string_id}", f"Label_S{string_id}"
    df = df[df[label_col].isin(["Normal", "Degradation"])].reset_index(drop=True)
    
    model = MODELS[f"s{string_id}"]
    X = pd.DataFrame({V_col: df[V_col], I_col: df[I_col], "Irr": df["Irr"], T_col: df[T_col]})
    X["Power"] = X[V_col] * X[I_col]
    X["PerformanceRatio"] = X["Power"] / (X["Irr"] + 1e-6)
    X["Resistance"] = X[V_col] / (X[I_col] + 1e-6)
    X = X[[V_col, I_col, "Irr", T_col, "Power", "PerformanceRatio", "Resistance"]]

    predicted = model.predict(X).astype(int)
    actual = (df[label_col] == "Degradation").astype(int).values

    n_normal = int((actual == 0).sum())
    false_alarms = int(((predicted == 1) & (actual == 0)).sum())
    far = (false_alarms / n_normal * 100) if n_normal else 0.0

    first_fault_idx = next((i for i, a in enumerate(actual) if a == 1), None)
    detected, delay_rows = False, None
    if first_fault_idx is not None:
        consecutive = 0
        for i in range(first_fault_idx, len(predicted)):
            if predicted[i] == 1:
                consecutive += 1
                if consecutive >= 3: 
                    delay_rows = int(i - first_fault_idx)
                    detected = True
                    break
            else: consecutive = 0

    delay_seconds = (delay_rows * 60) if delay_rows is not None else None

    return {
        "false_alarm_rate_pct": round(far, 4),
        "detected": detected,
        "detection_delay_seconds": delay_seconds,
        "persistence_threshold_rows": 3
    }