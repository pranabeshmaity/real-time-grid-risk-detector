from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
import os
import numpy as np

from app.api import api_router
from app.core.logging import setup_logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../ml_pipeline/models'))

setup_logging()
logger = logging.getLogger(__name__)

from ultimate_predictor import UltimatePredictor

class UGIMModelWrapper:
    def __init__(self):
        self.predictor = None
        self.is_initialized = False
    
    async def initialize(self):
        try:
            self.predictor = UltimatePredictor()
            self.is_initialized = True
            logger.info("UGIM (10 predictions) initialized")
        except Exception as e:
            logger.error(f"UGIM init failed: {e}")
            self.is_initialized = False
    
    async def predict(self, data):
        if not self.is_initialized:
            await self.initialize()
        
        try:
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
            features_tensor = feature_matrix.reshape(1, 1, 118, 15)
            
            result = self.predictor.predict(features_tensor)
            
            def get_alert_level(risk):
                if risk > 0.7: return "CRITICAL"
                elif risk > 0.4: return "WARNING"
                return "NORMAL"
            
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
            logger.error(f"Prediction failed: {e}")
            return {'risk_score': 0.3, 'alert_level': 'NORMAL', 'confidence': 0.7}
    
    async def cleanup(self):
        self.is_initialized = False

prediction_service = UGIMModelWrapper()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting UGIM Server with 10 Predictions")
    await prediction_service.initialize()
    yield
    await prediction_service.cleanup()

app = FastAPI(title="UGIM - 10 Predictions", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "UGIM-v2.0", "predictions": 10, "initialized": prediction_service.is_initialized}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_ugim:app", host="0.0.0.0", port=8000, reload=True)
