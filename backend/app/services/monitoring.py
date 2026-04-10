import psutil
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.collecting = False
        self.metrics_history = []
    
    async def start(self):
        self.collecting = True
        logger.info("Metrics collector started")
    
    async def stop(self):
        self.collecting = False
        logger.info("Metrics collector stopped")
    
    async def get_metrics(self) -> Dict[str, Any]:
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'connections': len(psutil.net_connections()),
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_summary(self) -> Dict[str, Any]:
        return await self.get_metrics()
