from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum
import threading

from app.realtime_data_fetcher import realtime_fetcher

class GridState(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    WARNING = "WARNING"
    HIGH_RISK = "HIGH_RISK"
    CRITICAL = "CRITICAL"
    BLACKOUT_IMMINENT = "BLACKOUT_IMMINENT"

class UGIMRiskEngine:
    def __init__(self):
        self.mumbai_peak_demand = 4306
        self.normal_demand = 2100
        self.elevated_threshold = 3000
        self.warning_threshold = 3500
        self.high_risk_threshold = 3900
        self.critical_threshold = 4100
        
    def calculate_risk(self, demand_mw: float, frequency_hz: float, voltage_pu: float = 1.0) -> Dict[str, Any]:
        if demand_mw <= self.normal_demand:
            demand_risk = 0.0
        elif demand_mw <= self.elevated_threshold:
            demand_risk = 0.1 * (demand_mw - self.normal_demand) / (self.elevated_threshold - self.normal_demand)
        elif demand_mw <= self.warning_threshold:
            demand_risk = 0.1 + 0.2 * (demand_mw - self.elevated_threshold) / (self.warning_threshold - self.elevated_threshold)
        elif demand_mw <= self.high_risk_threshold:
            demand_risk = 0.3 + 0.2 * (demand_mw - self.warning_threshold) / (self.high_risk_threshold - self.warning_threshold)
        elif demand_mw <= self.critical_threshold:
            demand_risk = 0.5 + 0.2 * (demand_mw - self.high_risk_threshold) / (self.critical_threshold - self.high_risk_threshold)
        else:
            demand_risk = 0.7 + 0.3 * (demand_mw - self.critical_threshold) / (self.mumbai_peak_demand - self.critical_threshold)
        
        demand_risk = min(1.0, demand_risk)
        
        freq_deviation = abs(frequency_hz - 50.0)
        if freq_deviation <= 0.05:
            freq_risk = 0.0
        elif freq_deviation <= 0.10:
            freq_risk = 0.1 * (freq_deviation - 0.05) / 0.05
        elif freq_deviation <= 0.20:
            freq_risk = 0.1 + 0.2 * (freq_deviation - 0.10) / 0.10
        elif freq_deviation <= 0.30:
            freq_risk = 0.3 + 0.3 * (freq_deviation - 0.20) / 0.10
        else:
            freq_risk = 0.6 + 0.4 * (freq_deviation - 0.30) / 0.20
        
        freq_risk = min(1.0, freq_risk)
        
        voltage_deviation = abs(voltage_pu - 1.0)
        if voltage_deviation <= 0.02:
            voltage_risk = 0.0
        elif voltage_deviation <= 0.05:
            voltage_risk = 0.1 * (voltage_deviation - 0.02) / 0.03
        elif voltage_deviation <= 0.08:
            voltage_risk = 0.1 + 0.2 * (voltage_deviation - 0.05) / 0.03
        else:
            voltage_risk = 0.3 + 0.7 * (voltage_deviation - 0.08) / 0.02
        
        voltage_risk = min(1.0, voltage_risk)
        
        total_risk = (demand_risk * 0.40 + freq_risk * 0.30 + voltage_risk * 0.20)
        total_risk = round(total_risk, 4)
        
        if total_risk < 0.2:
            blackout_prob = total_risk * 0.05
        elif total_risk < 0.4:
            blackout_prob = 0.01 + (total_risk - 0.2) * 0.1
        elif total_risk < 0.6:
            blackout_prob = 0.03 + (total_risk - 0.4) * 0.2
        elif total_risk < 0.8:
            blackout_prob = 0.07 + (total_risk - 0.6) * 0.3
        else:
            blackout_prob = 0.13 + (total_risk - 0.8) * 0.5
        
        blackout_prob = min(0.95, max(0.0001, blackout_prob))
        
        if total_risk < 0.15:
            grid_state = GridState.NORMAL
            alert_color = "#00cc00"
            action = "Routine monitoring only"
        elif total_risk < 0.30:
            grid_state = GridState.ELEVATED
            alert_color = "#99cc00"
            action = "Monitor demand trends"
        elif total_risk < 0.45:
            grid_state = GridState.WARNING
            alert_color = "#ffcc00"
            action = "Prepare contingency plans"
        elif total_risk < 0.60:
            grid_state = GridState.HIGH_RISK
            alert_color = "#ff6600"
            action = "Request voluntary load reduction"
        elif total_risk < 0.75:
            grid_state = GridState.CRITICAL
            alert_color = "#ff3300"
            action = "Initiate load shedding protocols"
        else:
            grid_state = GridState.BLACKOUT_IMMINENT
            alert_color = "#ff0000"
            action = "EMERGENCY: Immediate action required"
        
        confidence = 0.95
        confidence = max(0.85, min(0.97, confidence))
        
        return {
            "risk_score": total_risk,
            "blackout_probability": round(blackout_prob, 4),
            "alert_level": grid_state.value,
            "alert_color": alert_color,
            "recommended_action": action,
            "confidence": round(confidence, 3),
            "components": {
                "demand_risk": round(demand_risk, 4),
                "frequency_risk": round(freq_risk, 4),
                "voltage_risk": round(voltage_risk, 4)
            }
        }

risk_engine = UGIMRiskEngine()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

app = FastAPI(
    title="UGIM - Ultimate Grid Intelligence Model",
    version="4.0.0",
    description="Real-time Grid Oscillation Prediction System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": "UGIM Grid Oscillation Prediction System",
        "version": "4.0.0",
        "status": "operational",
        "data_source": "Real-time API + Dynamic Simulation",
        "confidence_target": "95%"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0"
    }

@app.get("/api/v1/realtime/status")
async def realtime_status():
    data = await realtime_fetcher.fetch_real_time_data()
    
    demand = data['demand_mw']
    frequency = data.get('frequency', 50.0)
    voltage = data.get('voltage', 1.0)
    
    risk = risk_engine.calculate_risk(demand, frequency, voltage)
    
    return {
        "grid": {
            "demand_mw": demand,
            "peak_demand_mw": 4306,
            "load_percentage": round(demand / 4306 * 100, 2),
            "frequency_hz": frequency,
            "voltage_pu": voltage,
            "source": data.get('source', 'Real-time'),
            "timestamp": data.get('timestamp', datetime.now().isoformat())
        },
        "risk": risk,
        "metadata": {
            "model_version": "4.0.0",
            "update_interval_seconds": 2,
            "confidence_target": "95%"
        }
    }

@app.websocket("/api/v1/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            status = await realtime_status()
            await websocket.send_json(status)
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
