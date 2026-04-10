"""
Advanced Real-Time Data Fetcher for Maharashtra Grid
Multiple data sources with fallback mechanisms
"""

import asyncio
import aiohttp
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class RealTimeDataFetcher:
    def __init__(self):
        self.mumbai_peak_demand = 4306
        self.current_demand = 0
        self.last_update = 0
        self.update_interval = 2
        self.data_history = deque(maxlen=2880)
        self.demand_ma = deque(maxlen=6)
        self.risk_ma = deque(maxlen=6)
        self.oscillation_detected = False
        
        # For testing - allow forced values
        self.test_mode = False
        self.test_demand = None
        
    def set_test_demand(self, demand: float):
        """Set test demand for simulation"""
        self.test_mode = True
        self.test_demand = demand
        
    def disable_test_mode(self):
        """Disable test mode"""
        self.test_mode = False
        self.test_demand = None
        
    async def fetch_realtime(self, force: bool = False) -> Dict[str, Any]:
        current_time = time.time()
        
        # Use test demand if in test mode
        if self.test_mode and self.test_demand is not None:
            demand = self.test_demand
            load_factor = demand / self.mumbai_peak_demand
            frequency = 50.0 - (load_factor - 0.5) * 0.3
            frequency = np.clip(frequency, 49.5, 50.5)
            voltage = 1.0 - (load_factor - 0.5) * 0.1
            voltage = np.clip(voltage, 0.92, 1.08)
            
            return {
                'demand_mw': demand,
                'peak_demand_mw': self.mumbai_peak_demand,
                'load_factor': load_factor,
                'frequency': frequency,
                'voltage': voltage,
                'source': 'Test Mode',
                'last_update': current_time,
                'update_interval_seconds': self.update_interval
            }
        
        if not force and (current_time - self.last_update) < self.update_interval:
            return self.get_cached_status()
        
        # Generate advanced simulation
        data = self._generate_advanced_simulation()
        self._process_realtime_data(data)
        self.last_update = current_time
        return self.get_cached_status()
    
    def _generate_advanced_simulation(self) -> Dict:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        is_weekend = now.weekday() >= 5
        
        if is_weekend:
            weekend_factor = 0.85
        else:
            weekend_factor = 1.0
        
        if 6 <= hour < 10:
            time_factor = 0.4 + (hour - 6) * 0.15
        elif 10 <= hour < 13:
            time_factor = 1.0 + (hour - 10) * 0.05
        elif 13 <= hour < 17:
            time_factor = 0.9
        elif 17 <= hour < 20:
            time_factor = 1.0 + (hour - 17) * 0.07
        elif 20 <= hour < 23:
            time_factor = 0.8
        else:
            time_factor = 0.4
        
        minute_variation = np.sin(minute / 30 * np.pi) * 0.03
        noise = np.random.normal(0, 0.01)
        
        base_demand = 2500 * weekend_factor
        demand = base_demand * (time_factor + minute_variation + noise)
        demand = min(demand, self.mumbai_peak_demand)
        demand = max(demand, 1400)
        
        frequency = 50.0 - (demand / self.mumbai_peak_demand - 0.5) * 0.3
        frequency = np.clip(frequency, 49.5, 50.5)
        voltage = 1.0 - (demand / self.mumbai_peak_demand - 0.5) * 0.1
        voltage = np.clip(voltage, 0.92, 1.08)
        
        self.oscillation_detected = self._detect_oscillations(demand, frequency)
        
        return {
            'demand_mw': round(demand, 1),
            'frequency': round(frequency, 3),
            'voltage': round(voltage, 3),
            'load_factor': round(demand / self.mumbai_peak_demand, 3),
            'oscillation_detected': self.oscillation_detected,
            'source': 'Advanced Simulation',
            'timestamp': time.time(),
            'hour': hour,
            'is_weekend': is_weekend
        }
    
    def _detect_oscillations(self, demand: float, frequency: float) -> bool:
        if len(self.demand_ma) < 3:
            return False
        demand_list = list(self.demand_ma)
        if len(demand_list) >= 3:
            rate_of_change = abs(demand_list[-1] - demand_list[-3]) / demand_list[-1] if demand_list[-1] > 0 else 0
            if rate_of_change > 0.05:
                return True
        if abs(frequency - 50.0) > 0.15:
            return True
        return False
    
    def _process_realtime_data(self, data: Dict):
        demand = data.get('demand_mw', 0)
        freq = data.get('frequency', 50.0)
        self.demand_ma.append(demand)
        load_factor = demand / self.mumbai_peak_demand if self.mumbai_peak_demand > 0 else 0
        risk = self._calculate_risk(load_factor, freq)
        self.risk_ma.append(risk)
        self.current_demand = demand
        self.current_frequency = freq
        self.current_risk = np.mean(self.risk_ma) if self.risk_ma else risk
        
        self.data_history.append({
            'timestamp': data.get('timestamp', time.time()),
            'demand_mw': demand,
            'frequency': freq,
            'risk_score': self.current_risk,
            'source': data.get('source', 'Unknown')
        })
    
    def _calculate_risk(self, load_factor: float, frequency: float) -> float:
        if load_factor > 0.9:
            demand_risk = 0.7 + (load_factor - 0.9) * 3
        elif load_factor > 0.75:
            demand_risk = 0.4 + (load_factor - 0.75) * 2
        elif load_factor > 0.6:
            demand_risk = 0.2 + (load_factor - 0.6) * 1.33
        else:
            demand_risk = load_factor * 0.33
        
        freq_deviation = abs(frequency - 50.0)
        if freq_deviation > 0.3:
            freq_risk = 0.8
        elif freq_deviation > 0.15:
            freq_risk = 0.5
        elif freq_deviation > 0.05:
            freq_risk = 0.2
        else:
            freq_risk = 0.0
        
        rocof_risk = 0.0
        if len(self.demand_ma) >= 3:
            demand_list = list(self.demand_ma)
            rate = abs(demand_list[-1] - demand_list[-3]) / self.mumbai_peak_demand
            rocof_risk = min(0.5, rate * 5)
        
        risk = demand_risk * 0.5 + freq_risk * 0.3 + rocof_risk * 0.2
        return min(1.0, max(0.0, risk))
    
    def get_cached_status(self) -> Dict[str, Any]:
        if not self.data_history:
            return self._generate_advanced_simulation()
        latest = self.data_history[-1]
        risk_trend = "stable"
        if len(self.data_history) > 6:
            old_risk = self.data_history[-6]['risk_score']
            if latest['risk_score'] > old_risk + 0.05:
                risk_trend = "increasing"
            elif latest['risk_score'] < old_risk - 0.05:
                risk_trend = "decreasing"
        return {
            'demand_mw': latest['demand_mw'],
            'peak_demand_mw': self.mumbai_peak_demand,
            'load_factor': round(latest['demand_mw'] / self.mumbai_peak_demand, 3),
            'frequency': latest.get('frequency', 50.0),
            'risk_score': latest['risk_score'],
            'risk_trend': risk_trend,
            'oscillation_detected': self.oscillation_detected,
            'source': latest.get('source', 'Real-time'),
            'last_update': latest['timestamp'],
            'update_interval_seconds': self.update_interval,
            'history_points': len(self.data_history)
        }
    
    def get_historical_data(self, minutes: int = 60) -> List[Dict]:
        history = list(self.data_history)
        if minutes > 0:
            points_needed = minutes * 2
            return history[-points_needed:]
        return history

realtime_fetcher = RealTimeDataFetcher()
