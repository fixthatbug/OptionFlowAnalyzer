# analysis_modules/alert_manager.py
"""
Alert management system for real-time signals
"""

from datetime import datetime
from typing import List, Tuple, Optional, Dict, Callable, Any
import threading
import queue

class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self, log_callback: Callable = None):
        self.log = log_callback or print
        self.alerts_queue = queue.Queue()
        self.alert_history = []
        
    def process_signal(self, signal):
        """Process incoming signal for alerts"""
        try:
            # Create alert based on signal urgency
            if signal.urgency_score >= 85:
                alert_level = "CRITICAL"
            elif signal.urgency_score >= 70:
                alert_level = "HIGH"
            else:
                alert_level = "MEDIUM"
            
            alert = {
                'timestamp': datetime.now(),
                'level': alert_level,
                'ticker': signal.ticker,
                'symbol': signal.option_symbol,
                'message': f"{signal.signal_type.value}: {signal.trade_recommendation}",
                'urgency': signal.urgency_score
            }
            
            self.alerts_queue.put(alert)
            self.alert_history.append(alert)
            
            # Log critical alerts
            if alert_level == "CRITICAL":
                self.log(f"🚨 CRITICAL ALERT: {alert['message']}")
            
        except Exception as e:
            self.log(f"Error processing alert: {e}")
    
    def get_recent_alerts(self, minutes: int = 15) -> list:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alert_history if alert['timestamp'] >= cutoff]