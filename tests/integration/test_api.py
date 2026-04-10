import pytest
from fastapi.testclient import TestClient
import sys
sys.path.append('backend')
from app.main_realtime import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_realtime_status():
    response = client.get("/api/v1/realtime/status")
    assert response.status_code == 200
    assert "risk" in response.json()
