import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from app.services.real_data_fetcher import real_data_fetcher
from app.services.real_predictor import real_predictor

logger = logging.getLogger(__name__)

class ScheduledDataUpdater:
    def __init__(self, update_interval_seconds: int = 300):
        self.update_interval = update_interval_seconds
        self.running = False
        self.current_data = None
        self.last_prediction = None
        
    async def start(self):
        self.running = True
        logger.info(f"Starting scheduled updates every {self.update_interval} seconds")
        
        while self.running:
            try:
                self.current_data = real_data_fetcher.fetch_from_cea_api()
                self.last_prediction = await real_predictor.predict_with_real_data()
                logger.info(f"Update completed at {datetime.now().isoformat()}")
            except Exception as e:
                logger.error(f"Update failed: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def stop(self):
        self.running = False
    
    def get_latest_prediction(self):
        return self.last_prediction
    
    def get_latest_data(self):
        return self.current_data

updater = ScheduledDataUpdater(update_interval_seconds=300)
