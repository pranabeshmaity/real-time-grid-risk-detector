from fastapi import WebSocket
from typing import List, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_metadata: Dict[WebSocket, Dict] = {}
        self.is_running = True
        
    async def connect(self, websocket: WebSocket, client_info: Dict = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            'connected_at': datetime.now().isoformat(),
            'client_info': client_info or {},
            'message_count': 0
        }
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]['message_count'] += 1
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.disconnect(websocket)
            
    async def broadcast(self, message: Dict):
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                if connection in self.connection_metadata:
                    self.connection_metadata[connection]['message_count'] += 1
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_prediction(self, prediction: Dict):
        message = {
            'type': 'prediction',
            'data': prediction,
            'timestamp': datetime.now().isoformat()
        }
        await self.broadcast(message)
    
    def get_connection_stats(self) -> Dict:
        return {
            'total_connections': len(self.active_connections),
            'connections': [
                {
                    'connected_at': meta['connected_at'],
                    'message_count': meta['message_count']
                }
                for meta in self.connection_metadata.values()
            ]
        }
manager = ConnectionManager()