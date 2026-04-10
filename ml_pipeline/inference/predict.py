"""
Inference script for UGIM model
"""
import sys
sys.path.append('ml_pipeline/models')
from ultimate_predictor import UltimatePredictor
import numpy as np

class ModelInference:
    def __init__(self, model_path='ml_pipeline/saved_models/ugim_trained.pt'):
        self.predictor = UltimatePredictor(model_path)
    
    def predict_risk(self, voltages, frequencies):
        features = np.array([np.mean(voltages), np.std(voltages)] + 
                           [np.mean(frequencies), np.std(frequencies)])
        result = self.predictor.predict(features)
        return result.oscillation_risk

# Usage
if __name__ == "__main__":
    inference = ModelInference()
    print("Inference engine ready")
