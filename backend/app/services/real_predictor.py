from app.services.predictor import PredictionService
from app.services.real_data_fetcher import real_data_fetcher
import logging

logger = logging.getLogger(__name__)

class RealDataPredictionService(PredictionService):
    """Prediction service using real Indian grid data"""
    
    async def predict_with_real_data(self) -> Dict[str, Any]:
        """Fetch real data and make prediction"""
        # Fetch from best available source
        # Priority: CEA API -> SLDC Live -> NPP -> Simulated
        real_data = real_data_fetcher.fetch_from_cea_api()
        
        # Make prediction
        prediction = await self.predict(real_data)
        
        # Add real data metadata
        prediction['data_source'] = real_data['metadata']['source']
        prediction['timestamp_real'] = real_data['metadata']['timestamp']
        prediction['grid_frequency'] = '50Hz (Indian Standard)'
        
        return prediction

real_predictor = RealDataPredictionService()
