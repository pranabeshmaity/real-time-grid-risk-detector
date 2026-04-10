from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uvicorn
import numpy as np
import random
from datetime import datetime
from typing import List
import json

app = FastAPI(title="UGIM Grid Intelligence", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

def calculate_risk(demand_mw: float) -> dict:
    load_factor = demand_mw / 4306
    
    if load_factor < 0.55:
        risk = 0.05 + (load_factor - 0.3) * 0.15
        alert = "NORMAL"
        color = "#00cc00"
        action = "Routine monitoring"
    elif load_factor < 0.70:
        risk = 0.08 + (load_factor - 0.55) * 0.4
        alert = "NORMAL"
        color = "#90cc00"
        action = "Monitor demand trends"
    elif load_factor < 0.85:
        risk = 0.14 + (load_factor - 0.70) * 0.8
        alert = "WARNING"
        color = "#ffcc00"
        action = "Prepare contingency plans"
    elif load_factor < 0.95:
        risk = 0.26 + (load_factor - 0.85) * 1.8
        alert = "HIGH_RISK"
        color = "#ff6600"
        action = "Request load reduction"
    else:
        risk = 0.44 + (load_factor - 0.95) * 5.0
        alert = "CRITICAL"
        color = "#ff0000"
        action = "Immediate action required"
    
    risk = min(0.95, max(0.02, risk))
    confidence = 0.95
    blackout = risk * 0.08
    
    return {
        "risk_score": round(risk, 4),
        "blackout_probability": round(blackout, 4),
        "alert_level": alert,
        "alert_color": color,
        "recommended_action": action,
        "confidence": confidence
    }

def get_grid_data():
    current_hour = datetime.now().hour
    
    if 6 <= current_hour < 10:
        demand = 1800 + (current_hour - 6) * 200
    elif 10 <= current_hour < 13:
        demand = 2600 + (current_hour - 10) * 150
    elif 17 <= current_hour < 20:
        demand = 2600 + (current_hour - 17) * 280
    elif 20 <= current_hour < 23:
        demand = 3440 - (current_hour - 20) * 150
    else:
        demand = 2100
    
    demand = demand + random.uniform(-25, 25)
    demand = max(1400, min(4306, demand))
    
    frequency = 50.0 - (demand / 4306 - 0.5) * 0.15
    frequency = frequency + random.uniform(-0.03, 0.03)
    frequency = max(49.8, min(50.2, frequency))
    
    voltage = 1.0 - (demand / 4306 - 0.5) * 0.05
    voltage = voltage + random.uniform(-0.008, 0.008)
    voltage = max(0.95, min(1.05, voltage))
    
    return round(demand, 1), round(frequency, 3), round(voltage, 3)

@app.get("/")
async def root():
    return {"name": "UGIM Grid Intelligence", "version": "7.0.0", "status": "operational"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/realtime/status")
async def get_status():
    demand, frequency, voltage = get_grid_data()
    risk = calculate_risk(demand)
    return {
        "grid": {
            "demand_mw": demand,
            "peak_demand_mw": 4306,
            "load_percentage": round(demand / 4306 * 100, 1),
            "frequency_hz": frequency,
            "voltage_pu": voltage,
            "timestamp": datetime.now().isoformat()
        },
        "risk": risk,
        "metadata": {"model_version": "7.0.0", "confidence_target": "95%"}
    }

@app.websocket("/api/v1/realtime/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            demand, frequency, voltage = get_grid_data()
            risk = calculate_risk(demand)
            await websocket.send_json({
                "grid": {
                    "demand_mw": demand,
                    "frequency_hz": frequency,
                    "voltage_pu": voltage,
                    "load_percentage": round(demand / 4306 * 100, 1)
                },
                "risk": risk
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
