import pytest
import sys
sys.path.append('backend')
from app.services.predictor import PredictionService

@pytest.mark.asyncio
async def test_prediction_service():
    service = PredictionService()
    await service.initialize()
    
    data = {
        "voltages": [1.0] * 118,
        "frequencies": [50.0] * 118,
        "powers": [0.0] * 118
    }
    
    result = await service.predict(data)
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 1
    assert result["model_version"] == "PI-GNN-v1.0"
