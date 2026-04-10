import aiohttp
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class RealtimeDataFetcher:
    def __init__(self):
        self.last_demand = None
        self.last_frequency = None
        self.last_update = 0
        
    async def fetch_real_time_data(self) -> Dict[str, Any]:
        sources = [
            self._fetch_sldc_live,
            self._fetch_npp_api,
            self._fetch_cea_data
        ]
        
        for source in sources:
            try:
                data = await source()
                if data and data.get('demand_mw', 0) > 0:
                    self.last_demand = data['demand_mw']
                    self.last_frequency = data.get('frequency', 50.0)
                    self.last_update = datetime.now().timestamp()
                    return data
            except Exception as e:
                logger.warning(f"Source failed: {e}")
                continue
        
        return self._generate_dynamic_simulation()
    
    async def _fetch_sldc_live(self) -> Optional[Dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://sldc-live.vercel.app/api/current', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            'demand_mw': float(data.get('demand', 0)),
                            'frequency': float(data.get('frequency', 50.0)),
                            'source': 'SLDC Live',
                            'timestamp': datetime.now().isoformat()
                        }
        except:
            return None
    
    async def _fetch_npp_api(self) -> Optional[Dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://npp.gov.in/api/western_region', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {
                            'demand_mw': float(data.get('load_mw', 0)) * 0.3,
                            'frequency': 50.0,
                            'source': 'NPP API',
                            'timestamp': datetime.now().isoformat()
                        }
        except:
            return None
    
    async def _fetch_cea_data(self) -> Optional[Dict]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://cea.nic.in/api/regional_load.php', timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        western_load = float(data.get('western_region_mw', 25000))
                        return {
                            'demand_mw': western_load * 0.28,
                            'frequency': 50.0,
                            'source': 'CEA Data',
                            'timestamp': datetime.now().isoformat()
                        }
        except:
            return None
    
    def _generate_dynamic_simulation(self) -> Dict:
        current_hour = datetime.now().hour
        current_minute = datetime.now().minute
        current_second = datetime.now().second
        
        time_seconds = current_hour * 3600 + current_minute * 60 + current_second
        
        if 6 <= current_hour < 10:
            demand_base = 1800 + (current_hour - 6) * 200
        elif 10 <= current_hour < 13:
            demand_base = 2600 + (current_hour - 10) * 200
        elif 17 <= current_hour < 20:
            demand_base = 2600 + (current_hour - 17) * 300
        elif 20 <= current_hour < 23:
            demand_base = 3500 - (current_hour - 20) * 200
        elif current_hour < 6:
            demand_base = 1800 - (current_hour - 23) * 100 if current_hour < 23 else 1400
        else:
            demand_base = 2100
        
        minute_variation = np.sin(time_seconds / 300) * 50
        random_variation = np.random.normal(0, 15)
        demand = demand_base + minute_variation + random_variation
        demand = max(1400, min(4306, demand))
        
        frequency_base = 50.0 - (demand / 4306 - 0.5) * 0.15
        frequency_variation = np.sin(time_seconds / 60) * 0.02
        frequency = frequency_base + frequency_variation + np.random.normal(0, 0.01)
        frequency = max(49.7, min(50.3, frequency))
        
        voltage = 1.0 - (demand / 4306 - 0.5) * 0.05
        voltage = voltage + np.sin(time_seconds / 120) * 0.005
        voltage = max(0.95, min(1.05, voltage))
        
        return {
            'demand_mw': round(demand, 1),
            'frequency': round(frequency, 3),
            'voltage': round(voltage, 3),
            'source': 'Dynamic Real-time Simulation',
            'timestamp': datetime.now().isoformat()
        }

realtime_fetcher = RealtimeDataFetcher()
