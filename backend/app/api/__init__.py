from fastapi import APIRouter
from app.api.endpoints import predictions, data, system, websocket

api_router = APIRouter()

api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
api_router.include_router(data.router, prefix="/data", tags=["Data"])
api_router.include_router(system.router, prefix="/system", tags=["System"])
api_router.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])
from app.api.endpoints.realtime import router as realtime_router
api_router.include_router(realtime_router, prefix="/realtime", tags=["Real-time"])
