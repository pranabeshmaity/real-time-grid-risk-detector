import sys
import os

PROJECT_ROOT = '/Users/pranabeshmaity/Desktop/grid-oscillation-ecosystem'
MODELS_PATH = os.path.join(PROJECT_ROOT, 'ml_pipeline/models')
sys.path.insert(0, MODELS_PATH)

import numpy as np
import logging
from datetime import datetime
from collections import deque
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.predictor = None
        self.is_initialized = False
        self.prediction_buffer = deque(maxlen=10000)
        self.statistics = {
            'total_predictions': 0,
            'avg_risk': 0.0,
            'alerts_issued': 0
        }
        
    async def initialize(self):
        try:
            from advanced_predictor import GridOscillationPredictor
            self.predictor = GridOscillationPredictor()
            self.is_initialized = True
            logger.info("Advanced prediction service initialized with Physics-Informed GNN")
        except Exception as e:
            logger.error(f"Failed to load advanced model: {e}")
            self.is_initialized = True
            logger.info("Using fallback prediction service")
    
    def _extract_features(self, data: Dict[str, Any]) -> np.ndarray:
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
    
    def _get_alert_level(self, risk_score: float) -> str:
        if risk_score > 0.7:
            return "CRITICAL"
        elif risk_score > 0.4:
            return "WARNING"
        return "NORMAL"
    
    async def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.predictor:
                features = self._extract_features(data)
                prediction = self.predictor.predict(features)
                
                result = {
                    'risk_score': prediction['risk_score'],
                    'alert_level': self._get_alert_level(prediction['risk_score']),
                    'oscillation_mode': prediction['oscillation_mode'],
                    'confidence': prediction['confidence'],
                    'mode_probabilities': prediction.get('mode_probabilities', []),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': 'PI-GNN-v1.0',
                    'model_type': 'Physics-Informed Graph Neural Network'
                }
            else:
                result = self._fallback_prediction(data)
            
            self.statistics['total_predictions'] += 1
            n = self.statistics['total_predictions']
            self.statistics['avg_risk'] = (
                (self.statistics['avg_risk'] * (n-1) + result['risk_score']) / n
            )
            if result['alert_level'] in ['CRITICAL', 'WARNING']:
                self.statistics['alerts_issued'] += 1
            
            self.prediction_buffer.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback_prediction(data)
    
    def _fallback_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        voltages = np.array(data.get('voltages', [1.0]*118))
        frequencies = np.array(data.get('frequencies', [50.0]*118))
        
        risk = min(1.0, max(0.0, 
            0.3 + 0.5 * (1 - np.mean(voltages)) + 
            0.3 * np.std(frequencies) / 0.2
        ))
        
        return {
            'risk_score': float(risk),
            'alert_level': self._get_alert_level(risk),
            'oscillation_mode': 0,
            'confidence': 0.7,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'fallback-v1.0',
            'model_type': 'Statistical Fallback'
        }
    
    async def store_prediction(self, prediction: Dict[str, Any]):
        logger.debug(f"Storing prediction: {prediction.get('risk_score')}")
        self.prediction_buffer.append(prediction)
    
    async def get_history(self, limit: int = 100, offset: int = 0, start_time=None, end_time=None):
        predictions = list(self.prediction_buffer)[-limit:]
        return predictions[::-1]
    
    async def get_latest(self):
        if self.prediction_buffer:
            return self.prediction_buffer[-1]
        return None
    
    async def get_statistics(self, hours: int = 24):
        return {
            **self.statistics,
            'buffer_size': len(self.prediction_buffer),
            'is_ready': self.is_initialized
        }
    
    def is_ready(self) -> bool:
        return self.is_initialized
    
    async def cleanup(self):
        self.is_initialized = False
        self.prediction_buffer.clear()
        logger.info("Prediction service cleaned up")
