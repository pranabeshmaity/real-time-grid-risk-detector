import redis
import json
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = None
        self.redis_url = redis_url
    
    async def connect(self):
        try:
            self.redis_client = redis.from_url(self.redis_url)
            logger.info("Cache service connected")
        except Exception as e:
            logger.error(f"Cache connection failed: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        if self.redis_client:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        if self.redis_client:
            self.redis_client.setex(key, ttl, json.dumps(value))
    
    async def delete(self, key: str):
        if self.redis_client:
            self.redis_client.delete(key)
    
    async def clear(self):
        if self.redis_client:
            self.redis_client.flushdb()
    
    async def disconnect(self):
        if self.redis_client:
            self.redis_client.close()
