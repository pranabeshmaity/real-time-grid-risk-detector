from fastapi import APIRouter, HTTPException
from typing import List, Optional
from datetime import datetime
import numpy as np
import sys
import os

# Add the correct path
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../ml_pipeline/models'))
sys.path.insert(0, models_path)

from ultimate_predictor import UltimatePredictor
from app.models.schemas import PMUData

router = APIRouter()

# Initialize UGIM predictor
ugim_predictor = None

def get_ugim_predictor():
    global ugim_predictor
    if ugim_predictor is None:
        ugim_predictor = UltimatePredictor()
    return ugim_predictor

def extract_features(data):
    voltages = np.array(data.get('voltages', [1.0]*118))
    frequencies = np.array(data.get('frequencies', [50.0]*118))
    powers = np.array(data.get('powers', [0.0]*118))
    
    features = np.array([
        np.mean(voltages), np.std(voltages), np.min(voltages), np.max(voltages), np.ptp(voltages),
        np.mean(frequencies), np.std(frequencies), np.mean(np.abs(frequencies - 50)),
        np.std(np.diff(frequencies)) if len(frequencies) > 1 else 0,
        np.mean(powers), np.std(powers), np.sum(powers),
        1.0 - min(1.0, np.std(voltages)),
        1.0 - min(1.0, np.std(frequencies)),
        float(np.sum(np.abs(np.fft.fft(frequencies))[:10]))
    ])
    
    feature_matrix = np.tile(features, (118, 1))
    return feature_matrix.reshape(1, 1, 118, 15)

def get_alert_level(risk):
    if risk > 0.7: return "CRITICAL"
    elif risk > 0.4: return "WARNING"
    return "NORMAL"

@router.post("/single")
async def predict_single(data: PMUData):
    try:
        predictor = get_ugim_predictor()
        features = extract_features(data.dict())
        result = predictor.predict(features)
        
        return {
            'risk_score': result.oscillation_risk,
            'blackout_probability': result.blackout_probability,
            'voltage_collapse_margin': result.voltage_collapse_margin,
            'frequency_nadir': result.frequency_nadir,
            'rocof': result.rocof,
            'oscillation_mode': result.oscillation_mode,
            'time_to_instability': result.time_to_instability,
            'affected_zones': result.affected_zones,
            'recommended_actions': result.recommended_actions,
            'uncertainty': result.uncertainty,
            'alert_level': get_alert_level(result.oscillation_risk),
            'confidence': 1 - result.uncertainty.get('total', 0.1),
            'timestamp': result.timestamp,
            'model_version': result.model_version,
            'model_type': 'Ultimate Grid Intelligence Model (10 Predictions)'
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest")
async def get_latest_prediction():
    return {"message": "Make a POST request to /single first"}

@router.get("/statistics")
async def get_prediction_statistics():
    return {"message": "Statistics available after predictions"}
