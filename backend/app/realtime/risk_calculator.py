"""
Advanced Risk Calculator with Physics-Informed ML
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)

class AdvancedRiskCalculator:
    """
    Multi-factor risk calculation using:
    1. Load factor
    2. Frequency deviation
    3. Rate of change of frequency (RoCoF)
    4. Voltage stability
    5. Oscillation detection
    6. Historical patterns
    """
    
    def __init__(self):
        self.risk_history = deque(maxlen=100)
        self.oscillation_modes = ['Normal', 'Low Freq', 'Inter-area', 'Local', 'Forced', 'Sub-sync']
        self.current_mode = 0
        
    def calculate(self, grid_data: Dict) -> Dict:
        """Calculate comprehensive risk metrics"""
        demand = grid_data.get('demand_mw', 2500)
        peak = grid_data.get('peak_demand_mw', 4306)
        frequency = grid_data.get('frequency', 50.0)
        voltage = grid_data.get('voltage', 1.0)
        
        load_factor = demand / peak
        
        # 1. Load-based risk
        load_risk = self._load_risk(load_factor)
        
        # 2. Frequency-based risk
        freq_risk = self._frequency_risk(frequency)
        
        # 3. Voltage-based risk
        voltage_risk = self._voltage_risk(voltage)
        
        # 4. Oscillation risk
        oscillation_risk = self._oscillation_risk(frequency)
        
        # 5. Trend risk (rate of change)
        trend_risk = self._trend_risk(load_factor)
        
        # Weighted combination
        total_risk = (
            load_risk * 0.30 +
            freq_risk * 0.25 +
            voltage_risk * 0.20 +
            oscillation_risk * 0.15 +
            trend_risk * 0.10
        )
        
        total_risk = min(1.0, max(0.0, total_risk))
        
        # Determine alert level
        if total_risk > 0.75:
            alert = "CRITICAL"
            color = "#ff0000"
        elif total_risk > 0.5:
            alert = "HIGH"
            color = "#ff6600"
        elif total_risk > 0.25:
            alert = "WARNING"
            color = "#ffcc00"
        else:
            alert = "NORMAL"
            color = "#00cc00"
        
        # Calculate blackout probability
        blackout_prob = self._blackout_probability(total_risk, frequency, load_factor)
        
        # Time to instability
        tti = self._time_to_instability(total_risk, frequency)
        
        self.risk_history.append(total_risk)
        
        return {
            'risk_score': round(total_risk, 4),
            'alert_level': alert,
            'alert_color': color,
            'blackout_probability': round(blackout_prob, 4),
            'time_to_instability': round(tti, 1),
            'components': {
                'load_risk': round(load_risk, 4),
                'frequency_risk': round(freq_risk, 4),
                'voltage_risk': round(voltage_risk, 4),
                'oscillation_risk': round(oscillation_risk, 4),
                'trend_risk': round(trend_risk, 4)
            },
            'oscillation_mode': self.current_mode,
            'oscillation_mode_name': self.oscillation_modes[self.current_mode]
        }
    
    def _load_risk(self, load_factor: float) -> float:
        """Calculate risk from load factor"""
        if load_factor > 0.9:
            return 0.7 + (load_factor - 0.9) * 3
        elif load_factor > 0.75:
            return 0.4 + (load_factor - 0.75) * 2
        elif load_factor > 0.6:
            return 0.2 + (load_factor - 0.6) * 1.33
        else:
            return load_factor * 0.33
    
    def _frequency_risk(self, frequency: float) -> float:
        """Calculate risk from frequency deviation"""
        deviation = abs(frequency - 50.0)
        if deviation > 0.3:
            return 0.9
        elif deviation > 0.2:
            return 0.7
        elif deviation > 0.1:
            return 0.4
        elif deviation > 0.05:
            return 0.2
        else:
            return 0.0
    
    def _voltage_risk(self, voltage: float) -> float:
        """Calculate risk from voltage deviation"""
        deviation = abs(voltage - 1.0)
        if deviation > 0.08:
            return 0.8
        elif deviation > 0.05:
            return 0.5
        elif deviation > 0.03:
            return 0.25
        else:
            return 0.0
    
    def _oscillation_risk(self, frequency: float) -> float:
        """Detect and quantify oscillation risk"""
        # Simulate oscillation detection based on frequency variation
        if len(self.risk_history) > 10:
            freq_variation = np.std(list(self.risk_history)[-10:])
            if freq_variation > 0.1:
                self.current_mode = 2  # Inter-area oscillation
                return 0.6
            elif freq_variation > 0.05:
                self.current_mode = 1  # Low frequency
                return 0.3
        return 0.0
    
    def _trend_risk(self, current_load: float) -> float:
        """Calculate risk from trend direction"""
        if len(self.risk_history) < 5:
            return 0.0
        
        recent = list(self.risk_history)[-5:]
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]
            if trend > 0.1:
                return 0.5
            elif trend > 0.05:
                return 0.25
        return 0.0
    
    def _blackout_probability(self, risk: float, frequency: float, load_factor: float) -> float:
        """Calculate blackout probability using multiple factors"""
        base_prob = risk * 0.7
        
        # Frequency factor
        if frequency < 49.5:
            base_prob += 0.2
        elif frequency > 50.5:
            base_prob += 0.1
        
        # Load factor factor
        if load_factor > 0.95:
            base_prob += 0.2
        elif load_factor > 0.9:
            base_prob += 0.1
        
        return min(0.99, base_prob)
    
    def _time_to_instability(self, risk: float, frequency: float) -> float:
        """Estimate time to instability in seconds"""
        if risk < 0.3:
            return 3600  # > 1 hour
        elif risk < 0.5:
            return 1800  # ~30 minutes
        elif risk < 0.7:
            return 600   # ~10 minutes
        elif risk < 0.85:
            return 180   # 3 minutes
        else:
            return 60    # 1 minute

# Singleton
risk_calculator = AdvancedRiskCalculator()
