import pytest
import requests
import time

def test_full_prediction_flow():
    # Test complete prediction flow
    data = {
        "voltages": [1.0] * 118,
        "frequencies": [50.0] * 118,
        "powers": [0.0] * 118
    }
    
    response = requests.post(
        "http://localhost:8000/api/v1/predictions/single",
        json=data
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "risk_score" in result
    print(f"Prediction successful: Risk = {result['risk_score']*100:.1f}%")
