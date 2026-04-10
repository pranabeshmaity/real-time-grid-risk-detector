from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class PMUData(BaseModel):
    voltages: List[float]
    frequencies: List[float]
    powers: Optional[List[float]] = []
    timestamp: Optional[datetime] = None

class PredictionResponse(BaseModel):
    risk_score: float
    alert_level: str
    oscillation_mode: int
    confidence: float
    timestamp: str
    model_version: str

class BatchPredictionRequest(BaseModel):
    data: List[PMUData]

class SystemMetrics(BaseModel):
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    timestamp: datetime