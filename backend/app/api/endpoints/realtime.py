"""
Real-time API Endpoints for Maharashtra Grid
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any
import asyncio

from app.realtime.data_fetcher import realtime_fetcher
from app.realtime.risk_calculator import risk_calculator
from app.realtime.websocket_manager import realtime_ws_manager

router = APIRouter()

@router.get("/status")
async def get_realtime_status() -> Dict[str, Any]:
    """Get current real-time grid status"""
    data = await realtime_fetcher.fetch_realtime()
    risk = risk_calculator.calculate(data)
    
    return {
        'grid': data,
        'risk': risk,
        'timestamp': __import__('time').time()
    }

@router.get("/history")
async def get_history(minutes: int = 30) -> Dict[str, Any]:
    """Get historical data for charting"""
    history = realtime_fetcher.get_historical_data(minutes=minutes)
    
    return {
        'history': history,
        'points': len(history),
        'minutes_requested': minutes
    }

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await realtime_ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        realtime_ws_manager.disconnect(websocket)
