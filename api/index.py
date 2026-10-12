from pathlib import Path
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from detection import DataValidationError, evaluate_predictions, load_activity_data, run_detection

app = FastAPI(title="InsiderWatch AI API")
ROOT = Path(__file__).resolve().parent.parent

def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, list): return [safe(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.bool_,)): return bool(v)
    if pd.isna(v): return None
    if hasattr(v, "item"): return safe(v.item())
    return v

def report(frame, contamination):
    result = run_detection(frame, contamination=contamination)
    data = {
      "summary": {"total_events": int(len(result)), "total_users": int(result.user_id.nunique()),
       "model_anomalies": int(result.is_anomaly.sum()), "baseline_flags": int(result.baseline_flag.sum()),
       "investigation_queue": int(result.investigation_flag.sum()), "contamination": contamination},
      "users": sorted(str(x) for x in result.user_id.dropna().unique()),
      "events": safe(result.replace({np.nan: None}).to_dict(orient="records"))
    }
    if "synthetic_label" in result and result.synthetic_label.notna().all():
        data["evaluation"] = safe(evaluate_predictions(result.synthetic_label.astype(int), result.is_anomaly.astype(int)))
    else: data["evaluation"] = None
    return JSONResponse(content=safe(data))

@app.get("/api/health")
def health(): return {"status":"ok","service":"InsiderWatch AI"}

@app.get("/api/analyze")
def analyze(contamination: float = Query(0.03, ge=0.01, le=0.20)):
    try: return report(load_activity_data(ROOT / "activity_data.csv"), contamination)
    except (DataValidationError, ValueError, FileNotFoundError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/analyze-upload")
async def analyze_upload(file: UploadFile = File(...), contamination: float = Query(0.03, ge=0.01, le=0.20)):
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(status_code=400, detail="Upload a CSV file.")
    raw = await file.read()
    if not raw: raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
    if len(raw) > 4000000: raise HTTPException(status_code=413, detail="CSV must be under 4 MB.")
    try: return report(load_activity_data(io.BytesIO(raw)), contamination)
    except (DataValidationError, ValueError, pd.errors.ParserError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
