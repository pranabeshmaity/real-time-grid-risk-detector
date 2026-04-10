"""
WebSocket Manager for Real-time Data Streaming
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class RealtimeWebSocketManager:
    """Manage WebSocket connections for real-time data streaming"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.broadcast_task = None
        self.is_running = False
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        
        # Start broadcast loop if not running
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self._broadcast_loop())
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")
        
        if not self.active_connections:
            self.is_running = False
    
    async def _broadcast_loop(self):
        """Broadcast real-time data to all connected clients"""
        from app.realtime.data_fetcher import realtime_fetcher
        from app.realtime.risk_calculator import risk_calculator
        
        while self.is_running and self.active_connections:
            try:
                # Fetch latest real-time data
                data = await realtime_fetcher.fetch_realtime()
                
                # Calculate advanced risk
                risk_data = risk_calculator.calculate(data)
                
                # Combine for broadcast
                broadcast_data = {
                    'type': 'realtime_update',
                    'timestamp': data.get('last_update', __import__('time').time()),
                    'grid': data,
                    'risk': risk_data
                }
                
                # Send to all connected clients
                for connection in list(self.active_connections):
                    try:
                        await connection.send_json(broadcast_data)
                    except:
                        self.disconnect(connection)
                
                await asyncio.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(5)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all clients"""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except:
                self.disconnect(connection)

# Singleton
realtime_ws_manager = RealtimeWebSocketManager()
