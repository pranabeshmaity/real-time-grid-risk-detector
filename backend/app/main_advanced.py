from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
import os

from app.api import api_router
from app.core.logging import setup_logging
from app.services.advanced_predictor_service import AdvancedPredictionService

# Override the default predictor in the main module
import app.main
app.main.prediction_service = AdvancedPredictionService()

setup_logging()
logger = logging.getLogger(__name__)

prediction_service = AdvancedPredictionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Advanced Grid Oscillation Prediction System")
    await prediction_service.initialize()
    # Also set the global instance
    import app.main
    app.main.prediction_service = prediction_service
    yield
    await prediction_service.cleanup()

app = FastAPI(
    title="Grid Oscillation System - Advanced AI", 
    version="2.0.0", 
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model": "Physics-Informed GNN", 
        "initialized": prediction_service.is_initialized
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_advanced:app", host="0.0.0.0", port=8000, reload=True)
