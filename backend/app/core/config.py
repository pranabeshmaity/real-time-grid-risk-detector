from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    APP_NAME: str = "UGIM Grid Intelligence Platform"
    APP_VERSION: str = "7.0.0"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    CORS_ORIGINS: List[str] = ["*"]
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/grid_db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    MODEL_PATH: str = "ml_pipeline/models/production_model.pt"
    DATA_DIR: str = "ml_pipeline/data"
    
    RETRAIN_INTERVAL_HOURS: int = 24
    MIN_SAMPLES_FOR_RETRAIN: int = 100
    
    WEBSOCKET_UPDATE_INTERVAL: float = 1.0
    
    class Config:
        env_file = ".env"

settings = Settings()
