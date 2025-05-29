# rtd_integration.py
"""
Integration functions to connect RTD with analysis engine and signal detection
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
import threading
import queue

try:
    from rtd_handler import RTDDataHandler, RTDSignalProcessor
    from alpha_extractor import AlphaExtractor, AlphaSignal
    from analysis_engine import AnalysisEngine
    import config
    import alpha_config
except ImportError as e:
    print(f"Import warning in rtd_integration.py: {e}")
    # Create dummy classes to prevent import errors
    class RTDDataHandler:
        def __init__(self, *args, **kwargs): pass
    class RTDSignalProcessor:
        def __init__(self, *args, **kwargs): pass
    class AlphaExtractor:
        def __init__(self, *args, **kwargs): pass
    class AnalysisEngine:
        def __init__(self, *args, **kwargs): pass


class RTDAnalysisIntegrator:
    """Integrates RTD data with analysis engine for enhanced signal detection"""
    
    def __init__(self, ticker: str, log_callback: Callable = None):
        self.ticker = ticker
        self.log = log_callback or self._default_log
        
        # Core components
        self.rtd_handler = None
        self.rtd_processor = None
        self.analysis_engine = None
        self.alpha_extractor = None
        
        # Integration state
        self.is_integrated = False
        self.integration_thread = None
        self.stop_integration = False
        
        # Signal enhancement
        self.enhanced_signals = []
        self.signal_callbacks = []
        
        # Performance tracking
        self.rtd_signal_count = 0
        self.enhanced_signal_count = 0
        self.integration_start_time = None
    
    def _default_log(self, message: str, is_error: bool = False):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[RTD_INTEGRATION] [{timestamp}] {message}")
    
    def initialize_integration(self, analysis_results: dict, cleaned_df: pd.DataFrame) -> bool:
        """Initialize RTD integration with analysis results"""
        
        try:
            self.log(f"Initializing RTD integration for {self.ticker}")
            
            # Initialize RTD handler
            self.rtd_handler = RTDDataHandler(self.log)
            if not self.rtd_handler.initialize_rtd_connection():
                self.log("Failed to initialize RTD connection", is_error=True)
                return False
            
            # Identify symbols with signals
            signal_symbols = self.rtd_handler.identify_signal_based_symbols(cleaned_df, analysis_results)
            
            if not signal_symbols:
                self.log("No symbols with signals found for RTD monitoring")
                return False
            
            # Set up RTD formulas
            if not self.rtd_handler.setup_rtd_formulas(signal_symbols):
                self.log("Failed to set up RTD formulas", is_error=True)
                return False
            
            # Initialize analysis components
            self.analysis_engine = AnalysisEngine(self.ticker, self.log)
            
            # Get current underlying price for alpha extractor
            current_price = self._get_current_underlying_price(cleaned_df)
            self.alpha_extractor = AlphaExtractor(self.ticker, current_price)
            
            # Initialize RTD processor
            self.rtd_processor = RTDSignalProcessor(self.rtd_handler, self.log)
            
            # Set up RTD update callback
            self.rtd_handler.add_update_callback(self._on_rtd_update)
            
            self.log(f"RTD integration initialized for {len(signal_symbols)} symbols")
            return True
            
        except Exception as e:
            self.log(f"Error initializing RTD integration: {e}", is_error=True)
            return False
    
    def start_integration(self) -> bool:
        """Start integrated RTD and analysis monitoring"""
        
        if not self.rtd_handler:
            self.log("RTD handler not initialized", is_error=True)
            return False
        
        try:
            # Start RTD monitoring
            if not self.rtd_handler.start_rtd_monitoring():
                self.log("Failed to start RTD monitoring", is_error=True)
                return False
            
            # Start integration processing thread
            self.is_integrated = True
            self.stop_integration = False
            self.integration_start_time = datetime.now()
            
            self.integration_thread = threading.Thread(
                target=self._integration_loop, 
                daemon=True
            )
            self.integration_thread.start()
            
            self.log("RTD analysis integration started successfully")
            return True
            
        except Exception as e:
            self.log(f"Error starting integration: {e}", is_error=True)
            return False
    
    def stop_integration(self):
        """Stop RTD integration"""
        
        self.stop_integration = True
        self.is_integrated = False
        
        if self.rtd_handler:
            self.rtd_handler.stop_rtd_monitoring()
        
        if self.integration_thread:
            self.integration_thread.join(timeout=5)
        
        self.log("RTD integration stopped")
    
    def _integration_loop(self):
        """Main integration processing loop"""
        
        while self.is_integrated and not self.stop_integration:
            try:
                # Process any enhanced signals
                self._process_enhanced_signals()
                
                # Clean up old data periodically
                if datetime.now().minute % 10 == 0:  # Every 10 minutes
                    self._cleanup_old_data()
                
                # Sleep for processing interval
                time_to_sleep = getattr(alpha_config, 'ALPHA_BUFFER_PROCESS_INTERVAL', 5)
                threading.Event().wait(time_to_sleep)
                
            except Exception as e:
                self.log(f"Error in integration loop: {e}", is_error=True)
                threading.Event().wait(5)  # Wait longer on error
    
    def _on_rtd_update(self, rtd_data: dict):
        """Callback for RTD data updates"""
        
        try:
            # Process RTD data for signals
            rtd_signals = self.rtd_processor.process_rtd_update(rtd_data)
            self.rtd_signal_count += len(rtd_signals)
            
            # Enhance signals with analysis
            enhanced_signals = self._enhance_signals_with_rtd(rtd_signals, rtd_data)
            
            # Store enhanced signals
            self.enhanced_signals.extend(enhanced_signals)
            self.enhanced_signal_count += len(enhanced_signals)
            
            # Trigger callbacks for enhanced signals
            urgent_threshold = getattr(alpha_config, 'ALPHA_URGENT_SIGNAL_THRESHOLD', 85)
            for signal in enhanced_signals:
                if signal.urgency_score >= urgent_threshold:
                    self._trigger_signal_callbacks(signal)
                    
                    # Write urgent signals to Excel
                    if self.rtd_handler:
                        self.rtd_handler.write_alpha_signal(signal)
            
            # Log significant activity
            if len(enhanced_signals) > 0:
                urgent_count = sum(1 for s in enhanced_signals 
                                 if s.urgency_score >= urgent_threshold)
                if urgent_count > 0:
                    self.log(f"Generated {len(enhanced_signals)} enhanced signals ({urgent_count} urgent)")
            
        except Exception as e:
            self.log(f"Error in RTD update callback: {e}", is_error=True)
    
    def _enhance_signals_with_rtd(self, rtd_signals: list[dict], rtd_data: dict) -> list:
        """Enhance RTD signals with additional analysis"""
        
        enhanced_signals = []
        
        for rtd_signal in rtd_signals:
            try:
                # Get RTD data for this symbol
                symbol = rtd_signal['symbol']
                symbol_rtd_data = rtd_data.get(symbol, {})
                
                # Create enhanced alpha signal
                enhanced_signal = self._create_enhanced_alpha_signal(rtd_signal, symbol_rtd_data)
                
                if enhanced_signal:
                    # Apply additional analysis filters
                    if self._passes_analysis_filters(enhanced_signal, symbol_rtd_data):
                        enhanced_signals.append(enhanced_signal)
                
            except Exception as e:
                self.log(f"Error enhancing signal: {e}", is_error=True)
        
        return enhanced_signals
    
    def _create_enhanced_alpha_signal(self, rtd_signal: dict, rtd_data: dict):
        """Create enhanced alpha signal from RTD signal"""
        
        # This is a simplified implementation to prevent syntax errors
        # In a full implementation, this would create a proper AlphaSignal object
        return {
            'timestamp': datetime.now(),
            'signal_type': 'RTD_ENHANCED',
            'symbol': rtd_signal.get('symbol', ''),
            'urgency_score': rtd_signal.get('severity_score', 50),
            'confidence': 70,  # Default confidence
            'smart_money_score': 65,  # Default score
            'trade_recommendation': f"RTD Signal: {rtd_signal.get('message', 'Unknown')}"
        }
    
    def _passes_analysis_filters(self, signal, rtd_data: dict) -> bool:
        """Check if enhanced signal passes analysis filters"""
        
        # Basic filtering logic
        min_confidence = getattr(alpha_config, 'ALPHA_MIN_CONFIDENCE', 60)
        if signal.get('confidence', 0) < min_confidence:
            return False
        
        # Data quality filter
        if not self._has_good_rtd_data_quality(rtd_data):
            return False
        
        return True
    
    def _has_good_rtd_data_quality(self, rtd_data: dict) -> bool:
        """Check if RTD data has good quality"""
        
        # Check for essential data points
        required_fields = ['last_price', 'bid', 'ask']
        has_required = all(rtd_data.get(field) for field in required_fields)
        
        return has_required
    
    def _process_enhanced_signals(self):
        """Process and clean up enhanced signals"""
        
        # Remove old signals
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.enhanced_signals = [
            signal for signal in self.enhanced_signals 
            if signal.get('timestamp', datetime.now()) >= cutoff_time
        ]
    
    def _cleanup_old_data(self):
        """Clean up old data and signals"""
        
        try:
            # Clean up RTD processor data
            if self.rtd_processor:
                self.rtd_processor.clear_old_signals(hours=4)
            
            # Clean up enhanced signals (keep last 2 hours)
            cutoff_time = datetime.now() - timedelta(hours=2)
            initial_count = len(self.enhanced_signals)
            
            self.enhanced_signals = [
                signal for signal in self.enhanced_signals 
                if signal.get('timestamp', datetime.now()) >= cutoff_time
            ]
            
            cleaned_count = initial_count - len(self.enhanced_signals)
            if cleaned_count > 0:
                self.log(f"Cleaned up {cleaned_count} old enhanced signals")
            
        except Exception as e:
            self.log(f"Error in cleanup: {e}", is_error=True)
    
    def _trigger_signal_callbacks(self, signal):
        """Trigger callbacks for enhanced signals"""
        
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                self.log(f"Error in signal callback: {e}", is_error=True)
    
    def _get_current_underlying_price(self, df: pd.DataFrame) -> float:
        """Get current underlying price from dataframe"""
        
        if 'Underlying_Price' in df.columns and not df.empty:
            return df['Underlying_Price'].iloc[-1]
        
        return 100.0  # Default fallback
    
    # Public methods for external access
    def add_signal_callback(self, callback: Callable):
        """Add callback for enhanced signals"""
        self.signal_callbacks.append(callback)
    
    def remove_signal_callback(self, callback: Callable):
        """Remove signal callback"""
        if callback in self.signal_callbacks:
            self.signal_callbacks.remove(callback)
    
    def get_recent_enhanced_signals(self, minutes: int = 15) -> list:
        """Get recent enhanced signals"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [signal for signal in self.enhanced_signals 
                if signal.get('timestamp', datetime.now()) >= cutoff_time]
    
    def get_integration_summary(self) -> dict:
        """Get integration status summary"""
        
        summary = {
            'is_integrated': self.is_integrated,
            'ticker': self.ticker,
            'rtd_signals_total': self.rtd_signal_count,
            'enhanced_signals_total': self.enhanced_signal_count,
            'recent_enhanced_signals': len(self.get_recent_enhanced_signals(15)),
            'rtd_summary': None,
            'integration_start_time': self.integration_start_time
        }
        
        if self.rtd_handler:
            summary['rtd_summary'] = self.rtd_handler.get_rtd_summary()
        
        return summary
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics"""
        
        if not self.integration_start_time:
            return {}
        
        runtime = datetime.now() - self.integration_start_time
        runtime_hours = runtime.total_seconds() / 3600
        
        recent_signals = self.get_recent_enhanced_signals(60)  # Last hour
        urgent_threshold = getattr(alpha_config, 'ALPHA_URGENT_SIGNAL_THRESHOLD', 85)
        urgent_signals = [s for s in recent_signals if s.get('urgency_score', 0) >= urgent_threshold]
        
        return {
            'runtime_hours': runtime_hours,
            'rtd_signals_per_hour': self.rtd_signal_count / runtime_hours if runtime_hours > 0 else 0,
            'enhanced_signals_per_hour': self.enhanced_signal_count / runtime_hours if runtime_hours > 0 else 0,
            'recent_signals_last_hour': len(recent_signals),
            'urgent_signals_last_hour': len(urgent_signals),
            'enhancement_rate': (self.enhanced_signal_count / max(1, self.rtd_signal_count)) * 100,
            'avg_confidence': sum(s.get('confidence', 0) for s in recent_signals) / len(recent_signals) if recent_signals else 0,
            'avg_urgency': sum(s.get('urgency_score', 0) for s in recent_signals) / len(recent_signals) if recent_signals else 0
        }
    
    def cleanup(self):
        """Clean up integration resources"""
        
        self.stop_integration()
        
        if self.rtd_handler:
            self.rtd_handler.cleanup()
        
        self.log("RTD integration cleanup completed")


# Integration helper functions

def setup_comprehensive_rtd_integration(ticker: str, analysis_results: dict, 
                                       cleaned_df: pd.DataFrame, 
                                       log_callback: Callable) -> Optional[RTDAnalysisIntegrator]:
    """Set up comprehensive RTD integration"""
    
    try:
        # Create integrator
        integrator = RTDAnalysisIntegrator(ticker, log_callback)
        
        # Initialize integration
        if integrator.initialize_integration(analysis_results, cleaned_df):
            log_callback(f"RTD integration initialized for {ticker}")
            
            # Start integration
            if integrator.start_integration():
                log_callback(f"RTD integration started for {ticker}")
                return integrator
            else:
                log_callback(f"Failed to start RTD integration for {ticker}", is_error=True)
        else:
            log_callback(f"Failed to initialize RTD integration for {ticker}", is_error=True)
        
        return None
        
    except Exception as e:
        log_callback(f"Error setting up RTD integration: {e}", is_error=True)
        return None


def generate_rtd_integration_report(integrator: RTDAnalysisIntegrator) -> str:
    """Generate comprehensive RTD integration report"""
    
    if not integrator:
        return "RTD integration not available"
    
    report = "=" * 70 + "\n"
    report += "RTD INTEGRATION ANALYSIS REPORT\n"
    report += "=" * 70 + "\n\n"
    
    # Integration summary
    summary = integrator.get_integration_summary()
    
    report += f"Ticker: {summary['ticker']}\n"
    report += f"Integration Status: {'ACTIVE' if summary['is_integrated'] else 'INACTIVE'}\n"
    
    if summary['integration_start_time']:
        runtime = datetime.now() - summary['integration_start_time']
        report += f"Runtime: {runtime}\n"
    
    report += f"Total RTD Signals: {summary['rtd_signals_total']}\n"
    report += f"Total Enhanced Signals: {summary['enhanced_signals_total']}\n"
    report += f"Recent Enhanced Signals (15min): {summary['recent_enhanced_signals']}\n\n"
    
    # Performance metrics
    metrics = integrator.get_performance_metrics()
    if metrics:
        report += "PERFORMANCE METRICS:\n"
        report += "-" * 25 + "\n"
        report += f"RTD Signals/Hour: {metrics['rtd_signals_per_hour']:.1f}\n"
        report += f"Enhanced Signals/Hour: {metrics['enhanced_signals_per_hour']:.1f}\n"
        report += f"Enhancement Rate: {metrics['enhancement_rate']:.1f}%\n"
        report += f"Average Confidence: {metrics['avg_confidence']:.1f}%\n"
        report += f"Average Urgency: {metrics['avg_urgency']:.1f}\n"
        report += f"Urgent Signals (Last Hour): {metrics['urgent_signals_last_hour']}\n\n"
    
    # RTD summary
    rtd_summary = summary.get('rtd_summary')
    if rtd_summary:
        report += "RTD MONITORING STATUS:\n"
        report += "-" * 25 + "\n"
        report += f"Symbols Monitored: {rtd_summary.get('symbols_count', 0)}\n"
        report += f"Data Points: {rtd_summary.get('data_points', 0)}\n"
        
        if rtd_summary.get('last_update'):
            report += f"Last Update: {rtd_summary['last_update'].strftime('%H:%M:%S')}\n"
        
        if rtd_summary.get('priority_symbols'):
            report += f"Priority Symbols: {', '.join(rtd_summary['priority_symbols'][:5])}\n"
        
        report += "\n"
    
    # Recent enhanced signals
    recent_signals = integrator.get_recent_enhanced_signals(30)  # Last 30 minutes
    if recent_signals:
        report += "RECENT ENHANCED SIGNALS (30min):\n"
        report += "-" * 35 + "\n"
        
        for i, signal in enumerate(recent_signals[:10], 1):  # Top 10
            timestamp = signal.get('timestamp', datetime.now()).strftime("%H:%M:%S")
            signal_type = signal.get('signal_type', 'Unknown')
            symbol = signal.get('symbol', 'Unknown')
            confidence = signal.get('confidence', 0)
            urgency = signal.get('urgency_score', 0)
            recommendation = signal.get('trade_recommendation', 'No recommendation')
            
            report += f"{i:2d}. [{timestamp}] {signal_type}\n"
            report += f"    {symbol} | Confidence: {confidence:.0f}% | Urgency: {urgency:.0f}\n"
            report += f"    {recommendation}\n\n"
    
    report += "=" * 70 + "\n"
    
    return report