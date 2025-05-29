# real_time_alpha_monitor.py
"""
Real-time monitoring system for continuous alpha extraction
"""

import threading
import queue
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import pandas as pd
import numpy as np
from alpha_extractor import AlphaExtractor, AlphaSignal, SignalType

class RealTimeAlphaMonitor:
    """Monitors option flow in real-time and generates alpha signals"""
    
    def __init__(self, ticker: str, log_callback: Callable):
        self.ticker = ticker
        self.log = log_callback
        self.is_monitoring = False
        self.monitor_thread = None
        self.signal_queue = queue.Queue()
        self.trade_buffer = []
        self.last_process_time = datetime.now()
        self.alpha_extractor = None
        self.historical_signals = []
        
        # Performance tracking
        self.signal_performance = {}
        
        # Alert thresholds
        self.URGENT_SIGNAL_THRESHOLD = 85
        self.SMART_MONEY_THRESHOLD = 80
        self.BUFFER_PROCESS_INTERVAL = timedelta(seconds=5)
        
    def start_monitoring(self, initial_data: pd.DataFrame):
        """Start real-time monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.log(f"Starting real-time alpha monitoring for {self.ticker}")
        
        # Initialize alpha extractor with current price
        if not initial_data.empty:
            current_price = initial_data['Underlying_Price'].iloc[-1]
            self.alpha_extractor = AlphaExtractor(self.ticker, current_price)
            
            # Process initial data for historical context
            initial_signals = self.alpha_extractor.process_trade_flow(initial_data)
            self.historical_signals.extend(initial_signals)
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        self.log(f"Stopped alpha monitoring for {self.ticker}")
    
    def add_new_trades(self, new_trades: pd.DataFrame):
        """Add new trades to the buffer for processing"""
        if not self.is_monitoring or new_trades.empty:
            return
        
        # Add to buffer
        self.trade_buffer.append(new_trades)
        
        # Check for urgent signals immediately
        self._check_urgent_signals(new_trades)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Process buffered trades periodically
                if (datetime.now() - self.last_process_time) >= self.BUFFER_PROCESS_INTERVAL:
                    self._process_trade_buffer()
                    self.last_process_time = datetime.now()
                
                # Check signal performance
                self._update_signal_performance()
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                self.log(f"Error in monitoring loop: {e}", is_error=True)
    
    def _process_trade_buffer(self):
        """Process accumulated trades in buffer"""
        if not self.trade_buffer:
            return
        
        # Combine all buffered trades
        combined_df = pd.concat(self.trade_buffer, ignore_index=True)
        self.trade_buffer.clear()
        
        if self.alpha_extractor and not combined_df.empty:
            # Update underlying price if available
            if 'Underlying_Price' in combined_df.columns:
                self.alpha_extractor.underlying_price = combined_df['Underlying_Price'].iloc[-1]
            
            # Extract signals
            new_signals = self.alpha_extractor.process_trade_flow(combined_df)
            
            # Process each signal
            for signal in new_signals:
                self._process_new_signal(signal)
    
    def _check_urgent_signals(self, trades: pd.DataFrame):
        """Check for urgent signals that need immediate attention"""
        if trades.empty or not self.alpha_extractor:
            return
        
        # Quick checks for urgent patterns
        for _, trade in trades.iterrows():
            # Large block at ask
            if (trade['TradeQuantity'] >= 100 and 
                'Ask' in trade['Aggressor'] and
                trade['NotionalValue'] >= 25000):
                
                self.log(f"🚨 URGENT: Large block detected - {trade['StandardOptionSymbol']} "
                        f"{trade['TradeQuantity']} @ ${trade['Trade_Price']:.2f}")
            
            # Sweep pattern starting
            symbol_recent = trades[trades['StandardOptionSymbol'] == trade['StandardOptionSymbol']]
            if len(symbol_recent) >= 3:
                time_span = (symbol_recent.iloc[-1]['Time_dt'] - symbol_recent.iloc[0]['Time_dt']).total_seconds()
                if time_span <= 2:  # Multiple trades within 2 seconds
                    self.log(f"⚡ SWEEP ALERT: Potential sweep in {trade['StandardOptionSymbol']}")
    
    def _process_new_signal(self, signal: AlphaSignal):
        """Process a new alpha signal"""
        # Log the signal
        self._log_signal(signal)
        
        # Add to queue for UI update
        self.signal_queue.put(signal)
        
        # Track for performance
        self.signal_performance[signal.timestamp] = {
            'signal': signal,
            'entry_price': signal.entry_price,
             # Continuing real_time_alpha_monitor.py
            'current_price': signal.entry_price,
            'pnl': 0.0,
            'status': 'ACTIVE'
        }
        
        # Send alerts for high-priority signals
        if signal.urgency_score >= self.URGENT_SIGNAL_THRESHOLD:
            self._send_alert(signal, "URGENT")
        elif signal.smart_money_score >= self.SMART_MONEY_THRESHOLD:
            self._send_alert(signal, "SMART MONEY")
    
    def _log_signal(self, signal: AlphaSignal):
        """Log signal with appropriate formatting"""
        emoji_map = {
            SignalType.SWEEP_ORDER: "⚡",
            SignalType.BLOCK_TRADE: "🏢",
            SignalType.SMART_MONEY_ACCUMULATION: "🧠",
            SignalType.UNUSUAL_VOLUME: "📊",
            SignalType.VOLATILITY_ARBITRAGE: "💹",
            SignalType.GAMMA_SQUEEZE: "🎯",
        }
        
        emoji = emoji_map.get(signal.signal_type, "📌")
        
        self.log(
            f"{emoji} {signal.signal_type.value} - {signal.direction}\n"
            f"   {signal.option_symbol} @ ${signal.entry_price:.2f}\n"
            f"   Confidence: {signal.confidence:.0f}% | Smart Money: {signal.smart_money_score:.0f}\n"
            f"   {signal.trade_recommendation}",
            self.ticker
        )
    
    def _send_alert(self, signal: AlphaSignal, alert_type: str):
        """Send alert for high-priority signals"""
        # This would integrate with notification system
        # For now, just enhanced logging
        self.log(
            f"🚨 {alert_type} ALERT 🚨\n{signal.trade_recommendation}\n"
            f"Notional: ${signal.notional_value:,.0f}",
            self.ticker
        )
    
    def _update_signal_performance(self):
        """Update performance tracking for active signals"""
        # This would connect to real-time price feed
        # For now, it's a placeholder for the architecture
        pass
    
    def get_active_signals(self) -> list[AlphaSignal]:
        """Get currently active signals"""
        active = []
        while not self.signal_queue.empty():
            try:
                signal = self.signal_queue.get_nowait()
                active.append(signal)
            except queue.Empty:
                break
        return active
    
    def get_performance_summary(self) -> dict:
        """Get performance summary of signals"""
        total_signals = len(self.signal_performance)
        if total_signals == 0:
            return {'total_signals': 0}
        
        # Calculate win rate, avg profit, etc.
        winning = sum(1 for s in self.signal_performance.values() if s['pnl'] > 0)
        total_pnl = sum(s['pnl'] for s in self.signal_performance.values())
        
        return {
            'total_signals': total_signals,
            'win_rate': (winning / total_signals * 100) if total_signals > 0 else 0,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total_signals if total_signals > 0 else 0,
            'active_signals': sum(1 for s in self.signal_performance.values() if s['status'] == 'ACTIVE')
        }