from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio

from app.api import api_router
from app.core.logging import setup_logging
from app.services.real_predictor import real_predictor
from app.services.scheduled_updater import updater

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Real-Time Grid Oscillation System")
    
    await real_predictor.initialize()
    asyncio.create_task(updater.start())
    
    yield
    
    await updater.stop()
    await real_predictor.cleanup()

app = FastAPI(
    title="Grid Oscillation Prediction System - Real Data",
    version="1.0.0",
    docs_url="/api/docs",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "data_source": "Indian Grid (CEA/NPP)",
        "last_update": updater.last_prediction.get('timestamp_real') if updater.last_prediction and hasattr(updater.last_prediction, 'get') else None
    }

@app.get("/api/v1/predictions/latest")
async def get_latest():
    prediction = updater.get_latest_prediction()
    if prediction:
        return prediction
    return {"message": "Fetching first update..."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main_real:app", host="0.0.0.0", port=8000, reload=True)
