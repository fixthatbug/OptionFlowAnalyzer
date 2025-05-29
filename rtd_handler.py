# rtd_handler.py
"""
Real-Time Data (RTD) handler for fetching live option data from Excel/TOS
"""

import pandas as pd
import numpy as np
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import win32com.client
import pythoncom
import config
import alpha_config

class RTDDataHandler:
    """Handles RTD connections and live data streaming"""
    
    def __init__(self, log_callback: Callable = None):
        self.log = log_callback or self._default_log
        self.rtd_server = None
        self.excel_app = None
        self.workbook = None
        self.rtd_sheet = None
        self.alpha_sheet = None
        
        # RTD data storage
        self.rtd_data = {}
        self.last_update_time = {}
        self.rtd_symbols = set()
        self.update_callbacks = []
        
        # RTD monitoring
        self.is_monitoring = False
        self.monitor_thread = None
        self.data_queue = queue.Queue()
        
        # Signal-based symbol selection
        self.signal_based_symbols = []
        self.symbol_priorities = {}
        
    def _default_log(self, message: str, ticker: str = None, is_error: bool = False):
        """Default logging function"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[RTD] [{timestamp}] {message}")
    
    def initialize_rtd_connection(self) -> bool:
        """Initialize RTD connection with Excel"""
        try:
            # Initialize COM for this thread
            pythoncom.CoInitialize()
            
            # Connect to Excel
            self.excel_app = win32com.client.Dispatch("Excel.Application")
            self.excel_app.Visible = True
            self.excel_app.DisplayAlerts = False
            
            # Open or create RTD workbook
            workbook_path = os.path.join(os.getcwd(), config.EXCEL_WORKBOOK_NAME)
            
            try:
                self.workbook = self.excel_app.Workbooks.Open(workbook_path)
                self.log("Opened existing RTD workbook")
            except:
                self.workbook = self.excel_app.Workbooks.Add()
                self.workbook.SaveAs(workbook_path)
                self.log("Created new RTD workbook")
            
            # Set up worksheets
            self._setup_rtd_worksheets()
            
            self.log("RTD connection initialized successfully")
            return True
            
        except Exception as e:
            self.log(f"Failed to initialize RTD connection: {e}", is_error=True)
            return False
    
    def _setup_rtd_worksheets(self):
        """Set up RTD and Alpha worksheets"""
        # RTD Data sheet
        try:
            self.rtd_sheet = self.workbook.Worksheets(config.EXCEL_RTD_SHEET_NAME)
        except:
            self.rtd_sheet = self.workbook.Worksheets.Add()
            self.rtd_sheet.Name = config.EXCEL_RTD_SHEET_NAME
        
        # Alpha Signals sheet
        try:
            self.alpha_sheet = self.workbook.Worksheets(config.EXCEL_ALPHA_SHEET_NAME)
        except:
            self.alpha_sheet = self.workbook.Worksheets.Add()
            self.alpha_sheet.Name = config.EXCEL_ALPHA_SHEET_NAME
        
        # Set up headers for RTD sheet
        rtd_headers = [
            "Symbol", "Last Price", "Bid", "Ask", "Volume", "Open Interest",
            "IV", "Delta", "Gamma", "Theta", "Vega", "Underlying Price",
            "Strike", "Expiration", "Type", "Last Update", "Signal Priority"
        ]
        
        for i, header in enumerate(rtd_headers, 1):
            self.rtd_sheet.Cells(1, i).Value = header
            self.rtd_sheet.Cells(1, i).Font.Bold = True
        
        # Set up headers for Alpha sheet
        alpha_headers = [
            "Timestamp", "Signal Type", "Symbol", "Direction", "Confidence",
            "Urgency", "Smart Money Score", "Entry Price", "Target", "Stop",
            "Notional", "Recommendation", "Status"
        ]
        
        for i, header in enumerate(alpha_headers, 1):
            self.alpha_sheet.Cells(1, i).Value = header
            self.alpha_sheet.Cells(1, i).Font.Bold = True
    
    def identify_signal_based_symbols(self, df: pd.DataFrame, analysis_results: dict) -> list[str]:
        """Identify option symbols with potential signals for RTD monitoring"""
        
        signal_symbols = []
        symbol_scores = {}
        
        # Extract symbols from alpha signals
        alpha_signals = analysis_results.get('alpha_signals', [])
        for signal in alpha_signals:
            symbol = signal.option_symbol
            if symbol:
                priority_score = (signal.confidence * signal.urgency_score * signal.smart_money_score) / 10000
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + priority_score
                signal_symbols.append(symbol)
        
        # Extract symbols from unusual activity
        unusual_activity = analysis_results.get('unusual_activity', {})
        
        # Volume unusual symbols
        vol_unusual = unusual_activity.get('volume_unusual', [])
        for item in vol_unusual:
            symbol = item.get('symbol')
            if symbol:
                score = item.get('volume_ratio', 1) * 10
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
                signal_symbols.append(symbol)
        
        # Large trade symbols
        size_unusual = unusual_activity.get('size_unusual', [])
        for item in size_unusual:
            symbol = item.get('symbol')
            if symbol:
                score = item.get('percentile', 50)
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
                signal_symbols.append(symbol)
        
        # Extract symbols from flow analysis significant flows
        flow_analysis = analysis_results.get('flow_analysis', {})
        significant_flows = flow_analysis.get('significant_flows', [])
        for flow in significant_flows:
            symbol = flow.get('symbol')
            if symbol:
                score = flow.get('significance_score', 0)
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
                signal_symbols.append(symbol)
        
        # Extract symbols from pattern analysis
        pattern_analysis = analysis_results.get('pattern_analysis', {})
        detected_patterns = pattern_analysis.get('detected_patterns', {})
        
        # Block trades
        block_trades = detected_patterns.get('block_trades', {})
        for trade in block_trades.get('trades', []):
            symbol = trade.get('symbol')
            if symbol:
                score = trade.get('institutional_score', 0)
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
                signal_symbols.append(symbol)
        
        # Sweep orders
        sweep_orders = detected_patterns.get('sweep_orders', {})
        for sweep in sweep_orders.get('sweeps', []):
            symbol = sweep.get('symbol')
            if symbol:
                score = sweep.get('urgency_score', 0)
                symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
                signal_symbols.append(symbol)
        
        # Remove duplicates and sort by priority score
        unique_symbols = list(set(signal_symbols))
        
        # Sort by score (highest first)
        sorted_symbols = sorted(unique_symbols, 
                              key=lambda x: symbol_scores.get(x, 0), 
                              reverse=True)
        
        # Limit to max RTD symbols
        final_symbols = sorted_symbols[:config.MAX_OPTIONS_FOR_RTD]
        
        # Store priorities
        self.symbol_priorities = {symbol: symbol_scores.get(symbol, 0) for symbol in final_symbols}
        
        self.log(f"Identified {len(final_symbols)} symbols with signals for RTD monitoring")
        self.log(f"Top symbols: {final_symbols[:5]}")
        
        return final_symbols
    
    def setup_rtd_formulas(self, symbols: list[str]) -> bool:
        """Set up RTD formulas in Excel for specified symbols"""
        
        if not self.rtd_sheet:
            self.log("RTD sheet not available", is_error=True)
            return False
        
        try:
            # Clear existing data (keep headers)
            if len(symbols) > 0:
                last_row = self.rtd_sheet.UsedRange.Rows.Count
                if last_row > 1:
                    clear_range = f"A2:Q{last_row}"
                    self.rtd_sheet.Range(clear_range).Clear()
            
            # Set up RTD formulas for each symbol
            for i, symbol in enumerate(symbols, 2):  # Start from row 2
                self._setup_symbol_rtd_formulas(symbol, i)
            
            self.rtd_symbols = set(symbols)
            self.log(f"Set up RTD formulas for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.log(f"Error setting up RTD formulas: {e}", is_error=True)
            return False
    
    def _setup_symbol_rtd_formulas(self, symbol: str, row: int):
        """Set up RTD formulas for a single symbol"""
        
        try:
            # Symbol
            self.rtd_sheet.Cells(row, 1).Value = symbol
            
            # RTD formulas for TOS (adjust server name as needed)
            tos_server = "TOS.RTD"  # Adjust based on your TOS RTD server
            
            # Last Price
            formula = f'=RTD("{tos_server}","","{symbol}","LAST")'
            self.rtd_sheet.Cells(row, 2).Formula = formula
            
            # Bid
            formula = f'=RTD("{tos_server}","","{symbol}","BID")'
            self.rtd_sheet.Cells(row, 3).Formula = formula
            
            # Ask
            formula = f'=RTD("{tos_server}","","{symbol}","ASK")'
            self.rtd_sheet.Cells(row, 4).Formula = formula
            
            # Volume
            formula = f'=RTD("{tos_server}","","{symbol}","VOLUME")'
            self.rtd_sheet.Cells(row, 5).Formula = formula
            
            # Open Interest
            formula = f'=RTD("{tos_server}","","{symbol}","OPEN_INTEREST")'
            self.rtd_sheet.Cells(row, 6).Formula = formula
            
            # Greeks
            formula = f'=RTD("{tos_server}","","{symbol}","IMPLIED_VOLATILITY")'
            self.rtd_sheet.Cells(row, 7).Formula = formula
            
            formula = f'=RTD("{tos_server}","","{symbol}","DELTA")'
            self.rtd_sheet.Cells(row, 8).Formula = formula
            
            formula = f'=RTD("{tos_server}","","{symbol}","GAMMA")'
            self.rtd_sheet.Cells(row, 9).Formula = formula
            
            formula = f'=RTD("{tos_server}","","{symbol}","THETA")'
            self.rtd_sheet.Cells(row, 10).Formula = formula
            
            formula = f'=RTD("{tos_server}","","{symbol}","VEGA")'
            self.rtd_sheet.Cells(row, 11).Formula = formula
            
            # Underlying price
            underlying = self._extract_underlying_from_symbol(symbol)
            formula = f'=RTD("{tos_server}","","{underlying}","LAST")'
            self.rtd_sheet.Cells(row, 12).Formula = formula
            
            # Parse option details
            option_details = self._parse_option_symbol(symbol)
            self.rtd_sheet.Cells(row, 13).Value = option_details.get('strike', '')
            self.rtd_sheet.Cells(row, 14).Value = option_details.get('expiration', '')
            self.rtd_sheet.Cells(row, 15).Value = option_details.get('type', '')
            
            # Last update timestamp
            self.rtd_sheet.Cells(row, 16).Value = datetime.now().strftime("%H:%M:%S")
            
            # Signal priority
            priority = self.symbol_priorities.get(symbol, 0)
            self.rtd_sheet.Cells(row, 17).Value = f"{priority:.1f}"
            
        except Exception as e:
            self.log(f"Error setting up RTD formulas for {symbol}: {e}", is_error=True)
    
    def start_rtd_monitoring(self) -> bool:
        """Start monitoring RTD data updates"""
        
        if self.is_monitoring:
            self.log("RTD monitoring already active")
            return True
        
        try:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._rtd_monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            self.log("Started RTD monitoring")
            return True
            
        except Exception as e:
            self.log(f"Failed to start RTD monitoring: {e}", is_error=True)
            self.is_monitoring = False
            return False
    
    def stop_rtd_monitoring(self):
        """Stop RTD monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        self.log("Stopped RTD monitoring")
    
    def _rtd_monitoring_loop(self):
        """Main RTD monitoring loop"""
        
        while self.is_monitoring:
            try:
                # Read current RTD data
                rtd_data = self._read_rtd_data()
                
                if rtd_data:
                    # Update internal storage
                    self._update_rtd_storage(rtd_data)
                    
                    # Trigger callbacks
                    for callback in self.update_callbacks:
                        try:
                            callback(rtd_data)
                        except Exception as e:
                            self.log(f"Error in RTD callback: {e}", is_error=True)
                    
                    # Check for alerts
                    self._check_rtd_alerts(rtd_data)
                
                # Sleep for update interval
                time.sleep(1)  # Update every second
                
            except Exception as e:
                self.log(f"Error in RTD monitoring loop: {e}", is_error=True)
                time.sleep(5)  # Wait longer on error
    
    def _read_rtd_data(self) -> Optional[pd.DataFrame]:
        """Read current RTD data from Excel"""
        
        if not self.rtd_sheet:
            return None
        
        try:
            # Get used range
            used_range = self.rtd_sheet.UsedRange
            
            if used_range.Rows.Count <= 1:  # Only headers
                return None
            
            # Read data
            data = used_range.Value
            
            if not data or len(data) <= 1:
                return None
            
            # Convert to DataFrame
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            df['Last_Update_Time'] = datetime.now()
            
            return df
            
        except Exception as e:
            self.log(f"Error reading RTD data: {e}", is_error=True)
            return None
    
    def _update_rtd_storage(self, rtd_data: pd.DataFrame):
        """Update internal RTD data storage"""
        
        for _, row in rtd_data.iterrows():
            symbol = row.get('Symbol')
            if symbol:
                self.rtd_data[symbol] = {
                    'last_price': self._safe_float(row.get('Last Price')),
                    'bid': self._safe_float(row.get('Bid')),
                    'ask': self._safe_float(row.get('Ask')),
                    'volume': self._safe_int(row.get('Volume')),
                    'open_interest': self._safe_int(row.get('Open Interest')),
                    'iv': self._safe_float(row.get('IV')),
                    'delta': self._safe_float(row.get('Delta')),
                    'gamma': self._safe_float(row.get('Gamma')),
                    'theta': self._safe_float(row.get('Theta')),
                    'vega': self._safe_float(row.get('Vega')),
                    'underlying_price': self._safe_float(row.get('Underlying Price')),
                    'strike': self._safe_float(row.get('Strike')),
                    'expiration': row.get('Expiration'),
                    'option_type': row.get('Type'),
                    'signal_priority': self._safe_float(row.get('Signal Priority')),
                    'timestamp': datetime.now()
                }
                
                self.last_update_time[symbol] = datetime.now()
    
    def _check_rtd_alerts(self, rtd_data: pd.DataFrame):
        """Check RTD data for alert conditions"""
        
        alerts = []
        
        for _, row in rtd_data.iterrows():
            symbol = row.get('Symbol')
            if not symbol:
                continue
            
            # Price movement alerts
            last_price = self._safe_float(row.get('Last Price'))
            bid = self._safe_float(row.get('Bid'))
            ask = self._safe_float(row.get('Ask'))
            
            if last_price and bid and ask:
                spread = ask - bid
                mid_price = (bid + ask) / 2
                
                # Wide spread alert
                if mid_price > 0:
                    spread_pct = (spread / mid_price) * 100
                    if spread_pct > 20:  # 20% spread
                        alerts.append({
                            'type': 'wide_spread',
                            'symbol': symbol,
                            'spread_pct': spread_pct,
                            'message': f"Wide spread on {symbol}: {spread_pct:.1f}%"
                        })
                
                # Price outside bid-ask
                if last_price < bid * 0.95 or last_price > ask * 1.05:
                    alerts.append({
                        'type': 'price_anomaly',
                        'symbol': symbol,
                        'last_price': last_price,
                        'bid': bid,
                        'ask': ask,
                        'message': f"Price anomaly on {symbol}: {last_price} vs {bid}-{ask}"
                    })
            
            # Volume alerts
            volume = self._safe_int(row.get('Volume'))
            if volume and volume > 1000:  # High volume threshold
                alerts.append({
                    'type': 'high_volume',
                    'symbol': symbol,
                    'volume': volume,
                    'message': f"High volume on {symbol}: {volume:,}"
                })
            
            # IV alerts
            iv = self._safe_float(row.get('IV'))
            if iv:
                if iv > 1.0:  # 100% IV
                    alerts.append({
                        'type': 'high_iv',
                        'symbol': symbol,
                        'iv': iv,
                        'message': f"High IV on {symbol}: {iv:.1%}"
                    })
                elif iv < 0.05:  # 5% IV
                    alerts.append({
                        'type': 'low_iv',
                        'symbol': symbol,
                        'iv': iv,
                        'message': f"Low IV on {symbol}: {iv:.1%}"
                    })
        
        # Log alerts
        for alert in alerts:
            self.log(f"🚨 RTD Alert: {alert['message']}")
    
    def get_current_rtd_data(self, symbol: str = None) -> dict:
        """Get current RTD data for symbol or all symbols"""
        
        if symbol:
            return self.rtd_data.get(symbol, {})
        else:
            return self.rtd_data.copy()
    
    def get_rtd_summary(self) -> dict:
        """Get summary of RTD monitoring status"""
        
        summary = {
            'is_monitoring': self.is_monitoring,
            'symbols_count': len(self.rtd_symbols),
            'data_points': len(self.rtd_data),
            'last_update': max(self.last_update_time.values()) if self.last_update_time else None,
            'active_symbols': list(self.rtd_symbols),
            'priority_symbols': []
        }
        
        # Get top priority symbols
        if self.symbol_priorities:
            sorted_priorities = sorted(self.symbol_priorities.items(), 
                                     key=lambda x: x[1], reverse=True)
            summary['priority_symbols'] = [symbol for symbol, _ in sorted_priorities[:5]]
        
        return summary
    
    def add_update_callback(self, callback: Callable):
        """Add callback function for RTD updates"""
        self.update_callbacks.append(callback)
    
    def remove_update_callback(self, callback: Callable):
        """Remove callback function"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def write_alpha_signal(self, signal):
        """Write alpha signal to Excel sheet"""
        
        if not self.alpha_sheet:
            return
        
        try:
            # Find next empty row
            last_row = self.alpha_sheet.UsedRange.Rows.Count + 1
            
            # Write signal data
            self.alpha_sheet.Cells(last_row, 1).Value = signal.timestamp.strftime("%H:%M:%S")
            self.alpha_sheet.Cells(last_row, 2).Value = signal.signal_type.value
            self.alpha_sheet.Cells(last_row, 3).Value = signal.option_symbol
            self.alpha_sheet.Cells(last_row, 4).Value = signal.direction
            self.alpha_sheet.Cells(last_row, 5).Value = f"{signal.confidence:.1f}%"
            self.alpha_sheet.Cells(last_row, 6).Value = f"{signal.urgency_score:.1f}"
            self.alpha_sheet.Cells(last_row, 7).Value = f"{signal.smart_money_score:.1f}"
            self.alpha_sheet.Cells(last_row, 8).Value = f"${signal.entry_price:.2f}"
            self.alpha_sheet.Cells(last_row, 9).Value = f"${signal.target_price:.2f}" if signal.target_price else ""
            self.alpha_sheet.Cells(last_row, 10).Value = f"${signal.stop_price:.2f}" if signal.stop_price else ""
            self.alpha_sheet.Cells(last_row, 11).Value = f"${signal.notional_value:,.0f}"
            self.alpha_sheet.Cells(last_row, 12).Value = signal.trade_recommendation
            self.alpha_sheet.Cells(last_row, 13).Value = "ACTIVE"
            
            # Color code by urgency
            if signal.urgency_score >= alpha_config.ALPHA_URGENT_SIGNAL_THRESHOLD:
                # Red for urgent
                for col in range(1, 14):
                    self.alpha_sheet.Cells(last_row, col).Interior.Color = 0x0000FF  # Red
            elif signal.direction == "BULLISH":
                # Green for bullish
                for col in range(1, 14):
                    self.alpha_sheet.Cells(last_row, col).Interior.Color = 0x00FF00  # Green
            elif signal.direction == "BEARISH":
                # Light red for bearish
                for col in range(1, 14):
                    self.alpha_sheet.Cells(last_row, col).Interior.Color = 0x8080FF  # Light red
            
        except Exception as e:
            self.log(f"Error writing alpha signal to Excel: {e}", is_error=True)
    
    def cleanup(self):
        """Clean up RTD resources"""
        
        self.stop_rtd_monitoring()
        
        try:
            if self.workbook:
                self.workbook.Save()
            
            if self.excel_app:
                self.excel_app.Quit()
        except:
            pass
        
        try:
            pythoncom.CoUninitialize()
        except:
            pass
        
        self.log("RTD cleanup completed")
    
    # Helper methods
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert value to float"""
        try:
            if value is None or value == "":
                return None
            return float(value)
        except:
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert value to int"""
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except:
            return None
    
    def _extract_underlying_from_symbol(self, symbol: str) -> str:
        """Extract underlying ticker from option symbol"""
        # Remove leading dots
        symbol = symbol.lstrip('.')
        
        # Find where numbers start
        for i, char in enumerate(symbol):
            if char.isdigit():
                return symbol[:i]
        
        return symbol[:3]  # Fallback to first 3 characters
    
    def _parse_option_symbol(self, symbol: str) -> dict:
        """Parse option symbol to extract details"""
        details = {'strike': '', 'expiration': '', 'type': ''}
        
        try:
            # This is a simplified parser - adjust based on your symbol format
            import re
            
            # Remove leading dots
            symbol = symbol.lstrip('.')
            
            # Try to match standard option symbol format
            # Example: SPY240315C00500000
            match = re.match(r'([A-Z]+)(\d{6})([CP])(\d{8})', symbol.replace(' ', ''))
            
            if match:
                ticker, date_str, call_put, strike_str = match.groups()
                
                # Parse expiration date
                exp_date = datetime.strptime(date_str, '%y%m%d')
                details['expiration'] = exp_date.strftime('%m/%d/%y')
                
                # Parse strike
                strike = int(strike_str) / 1000
                details['strike'] = strike
                
                # Parse type
                details['type'] = 'Call' if call_put == 'C' else 'Put'
        
        except Exception as e:
            self.log(f"Error parsing option symbol {symbol}: {e}")
        
        return details


def fetch_and_display_rtd_data_for_tab(symbols: list[str], ticker: str, 
                                     display_widget, log_func: Callable,
                                     completion_callback: Callable = None):
    """Fetch RTD data and display in UI widget"""
    
    def rtd_fetch_thread():
        try:
            log_func(f"Starting RTD fetch for {len(symbols)} symbols...", ticker)
            
            # Initialize RTD handler
            rtd_handler = RTDDataHandler(log_func)
            
            if not rtd_handler.initialize_rtd_connection():
                log_func("Failed to initialize RTD connection", ticker, is_error=True)
                if completion_callback:
                    completion_callback(None)
                return
            
            # Set up RTD formulas
            if not rtd_handler.setup_rtd_formulas(symbols):
                log_func("Failed to set up RTD formulas", ticker, is_error=True)
                if completion_callback:
                    completion_callback(None)
                return
            
            log_func("RTD formulas set up, waiting for data...", ticker)
            
            # Wait for data to populate
            time.sleep(3)
            
            # Read data
            rtd_data = rtd_handler._read_rtd_data()
            
            if rtd_data is not None and not rtd_data.empty:
                # Display data in widget
                if display_widget:
                    display_rtd_results(rtd_data, display_widget, log_func)
                
                log_func(f"RTD data fetched successfully for {len(rtd_data)} symbols", ticker)
                
                if completion_callback:
                    completion_callback(rtd_data)
            else:
                log_func("No RTD data received", ticker, is_error=True)
                if completion_callback:
                    completion_callback(None)
            
            # Cleanup
            rtd_handler.cleanup()
            
        except Exception as e:
            log_func(f"Error in RTD fetch thread: {e}", ticker, is_error=True)
            if completion_callback:
                completion_callback(None)
    
    # Start RTD fetch in separate thread
    rtd_thread = threading.Thread(target=rtd_fetch_thread, daemon=True)
    rtd_thread.start()


def display_rtd_results(rtd_data: pd.DataFrame, display_widget, log_func: Callable):
    """Display RTD results in UI widget"""
    
    try:
        # Enable widget for writing
        current_state = display_widget.cget('state')
        if current_state == 'disabled':
            display_widget.config(state='normal')
        
        # Clear existing content
        display_widget.delete('1.0', 'end')
        
        # Format and display data
        output = "=== REAL-TIME DATA RESULTS ===\n\n"
        output += f"Last Updated: {datetime.now().strftime('%H:%M:%S')}\n"
        output += f"Symbols Monitored: {len(rtd_data)}\n\n"
        
        # Sort by signal priority
        rtd_data_sorted = rtd_data.sort_values('Signal Priority', ascending=False, na_position='last')
        
        for i, (_, row) in enumerate(rtd_data_sorted.iterrows(), 1):
            symbol = row.get('Symbol', 'Unknown')
            last_price = row.get('Last Price', 0)
            bid = row.get('Bid', 0)
            ask = row.get('Ask', 0)
            volume = row.get('Volume', 0)
            iv = row.get('IV', 0)
            delta = row.get('Delta', 0)
            priority = row.get('Signal Priority', 0)
            
            output += f"{i}. {symbol}\n"
            output += f"   Last: ${last_price:.2f} | Bid: ${bid:.2f} | Ask: ${ask:.2f}\n"
            
            if volume:
                output += f"   Volume: {volume:,} | "
            else:
                output += f"   Volume: N/A | "
            
            if iv:
                output += f"IV: {iv:.1%}\n"
            else:
                output += f"IV: N/A\n"
            
            if delta:
                output += f"   Delta: {delta:.3f} | "
            else:
                output += f"   Delta: N/A | "
            
            output += f"Priority: {priority:.1f}\n"
            
            # Add spread analysis
            if bid and ask and bid > 0:
                spread = ask - bid
                mid_price = (bid + ask) / 2
                spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
                output += f"   Spread: ${spread:.2f} ({spread_pct:.1f}%)\n"
            
            output += "\n"
        
        # Add summary statistics
        output += "\n=== SUMMARY STATISTICS ===\n"
        
        # Calculate averages for numeric columns
        numeric_cols = ['Last Price', 'Volume', 'IV', 'Delta', 'Signal Priority']
        for col in numeric_cols:
            if col in rtd_data.columns:
                values = pd.to_numeric(rtd_data[col], errors='coerce').dropna()
                if not values.empty:
                    if col == 'Volume':
                        output += f"Avg {col}: {values.mean():,.0f}\n"
                    elif col == 'IV':
                        output += f"Avg {col}: {values.mean():.1%}\n"
                    elif col == 'Signal Priority':
                        output += f"Avg {col}: {values.mean():.1f}\n"
                    else:
                        output += f"Avg {col}: ${values.mean():.2f}\n"
        
        # Insert formatted text
        display_widget.insert('1.0', output)
        
        # Restore original state
        if current_state == 'disabled':
            display_widget.config(state='disabled')
        
        log_func("RTD results displayed successfully")
        
    except Exception as e:
        log_func(f"Error displaying RTD results: {e}", is_error=True)


class RTDSignalProcessor:
    """Processes RTD data to generate additional signals"""
    
    def __init__(self, rtd_handler: RTDDataHandler, log_callback: Callable = None):
        self.rtd_handler = rtd_handler
        self.log = log_callback or self._default_log
        
        # Signal processing parameters
        self.price_change_threshold = 0.05  # 5% price change
        self.volume_spike_multiplier = 3.0
        self.iv_change_threshold = 0.20  # 20% IV change
        
        # Historical data for comparison
        self.price_history = {}
        self.volume_history = {}
        self.iv_history = {}
        
        # Generated signals
        self.rtd_signals = []
    
    def _default_log(self, message: str, is_error: bool = False):
        print(f"[RTD_PROCESSOR] {message}")
    
    def process_rtd_update(self, rtd_data: dict):
        """Process RTD data update and generate signals"""
        
        new_signals = []
        
        for symbol, data in rtd_data.items():
            try:
                # Update historical data
                self._update_historical_data(symbol, data)
                
                # Check for price movement signals
                price_signals = self._check_price_movements(symbol, data)
                new_signals.extend(price_signals)
                
                # Check for volume spikes
                volume_signals = self._check_volume_spikes(symbol, data)
                new_signals.extend(volume_signals)
                
                # Check for IV changes
                iv_signals = self._check_iv_changes(symbol, data)
                new_signals.extend(iv_signals)
                
                # Check for spread anomalies
                spread_signals = self._check_spread_anomalies(symbol, data)
                new_signals.extend(spread_signals)
                
            except Exception as e:
                self.log(f"Error processing RTD data for {symbol}: {e}", is_error=True)
        
        # Store new signals
        self.rtd_signals.extend(new_signals)
        
        # Log significant signals
        for signal in new_signals:
            if signal['severity'] in ['HIGH', 'CRITICAL']:
                self.log(f"🚨 RTD Signal: {signal['message']}")
        
        return new_signals
    
    def _update_historical_data(self, symbol: str, data: dict):
        """Update historical data for comparison"""
        
        timestamp = datetime.now()
        
        # Price history
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        last_price = data.get('last_price')
        if last_price:
            self.price_history[symbol].append((timestamp, last_price))
            # Keep only recent history (last 100 points)
            self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Volume history
        if symbol not in self.volume_history:
            self.volume_history[symbol] = []
        
        volume = data.get('volume')
        if volume:
            self.volume_history[symbol].append((timestamp, volume))
            self.volume_history[symbol] = self.volume_history[symbol][-100:]
        
        # IV history
        if symbol not in self.iv_history:
            self.iv_history[symbol] = []
        
        iv = data.get('iv')
        if iv:
            self.iv_history[symbol].append((timestamp, iv))
            self.iv_history[symbol] = self.iv_history[symbol][-100:]
    
    def _check_price_movements(self, symbol: str, data: dict) -> list[dict]:
        """Check for significant price movements"""
        
        signals = []
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return signals
        
        current_price = data.get('last_price')
        if not current_price:
            return signals
        
        # Compare with recent prices
        recent_prices = [price for _, price in self.price_history[symbol][-10:]]
        avg_recent_price = sum(recent_prices) / len(recent_prices)
        
        price_change = (current_price - avg_recent_price) / avg_recent_price
        
        if abs(price_change) >= self.price_change_threshold:
            direction = "UP" if price_change > 0 else "DOWN"
            severity = "CRITICAL" if abs(price_change) >= 0.10 else "HIGH"
            
            signals.append({
                'type': 'PRICE_MOVEMENT',
                'symbol': symbol,
                'direction': direction,
                'change_percent': price_change * 100,
                'current_price': current_price,
                'avg_price': avg_recent_price,
                'severity': severity,
                'timestamp': datetime.now(),
                'message': f"{symbol} price moved {direction} {abs(price_change)*100:.1f}% to ${current_price:.2f}"
            })
        
        return signals
    
    def _check_volume_spikes(self, symbol: str, data: dict) -> list[dict]:
        """Check for volume spikes"""
        
        signals = []
        
        if symbol not in self.volume_history or len(self.volume_history[symbol]) < 5:
            return signals
        
        current_volume = data.get('volume')
        if not current_volume:
            return signals
        
        # Calculate average recent volume
        recent_volumes = [vol for _, vol in self.volume_history[symbol][-20:] if vol > 0]
        
        if not recent_volumes:
            return signals
        
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        
        if current_volume >= avg_volume * self.volume_spike_multiplier:
            volume_ratio = current_volume / avg_volume
            severity = "CRITICAL" if volume_ratio >= 5.0 else "HIGH"
            
            signals.append({
                'type': 'VOLUME_SPIKE',
                'symbol': symbol,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'severity': severity,
                'timestamp': datetime.now(),
                'message': f"{symbol} volume spike: {current_volume:,} ({volume_ratio:.1f}x avg)"
            })
        
        return signals
    
    def _check_iv_changes(self, symbol: str, data: dict) -> list[dict]:
        """Check for IV changes"""
        
        signals = []
        
        if symbol not in self.iv_history or len(self.iv_history[symbol]) < 2:
            return signals
        
        current_iv = data.get('iv')
        if not current_iv:
            return signals
        
        # Compare with recent IV
        recent_ivs = [iv for _, iv in self.iv_history[symbol][-10:]]
        avg_iv = sum(recent_ivs) / len(recent_ivs)
        
        iv_change = abs(current_iv - avg_iv) / avg_iv
        
        if iv_change >= self.iv_change_threshold:
            direction = "UP" if current_iv > avg_iv else "DOWN"
            severity = "HIGH" if iv_change >= 0.50 else "MEDIUM"
            
            signals.append({
                'type': 'IV_CHANGE',
                'symbol': symbol,
                'direction': direction,
                'change_percent': iv_change * 100,
                'current_iv': current_iv,
                'avg_iv': avg_iv,
                'severity': severity,
                'timestamp': datetime.now(),
                'message': f"{symbol} IV moved {direction} {iv_change*100:.1f}% to {current_iv:.1%}"
            })
        
        return signals
    
    def _check_spread_anomalies(self, symbol: str, data: dict) -> list[dict]:
        """Check for bid-ask spread anomalies"""
        
        signals = []
        
        bid = data.get('bid')
        ask = data.get('ask')
        
        if not bid or not ask or bid <= 0 or ask <= 0:
            return signals
        
        spread = ask - bid
        mid_price = (bid + ask) / 2
        spread_pct = (spread / mid_price) * 100
        
        # Check for wide spreads
        if spread_pct > 25:  # 25% spread
            severity = "CRITICAL" if spread_pct > 50 else "HIGH"
            
            signals.append({
                'type': 'WIDE_SPREAD',
                'symbol': symbol,
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'spread_percent': spread_pct,
                'severity': severity,
                'timestamp': datetime.now(),
                'message': f"{symbol} wide spread: {spread_pct:.1f}% (${spread:.2f})"
            })
        
        return signals
    
    def get_recent_signals(self, minutes: int = 5) -> list[dict]:
        """Get signals from last N minutes"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [signal for signal in self.rtd_signals 
                if signal['timestamp'] >= cutoff_time]
    
    def get_signals_by_symbol(self, symbol: str) -> list[dict]:
        """Get all signals for a specific symbol"""
        
        return [signal for signal in self.rtd_signals 
                if signal['symbol'] == symbol]
    
    def clear_old_signals(self, hours: int = 24):
        """Clear signals older than specified hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        self.rtd_signals = [signal for signal in self.rtd_signals 
                           if signal['timestamp'] >= cutoff_time]


def integrate_rtd_with_alpha_monitor(rtd_handler: RTDDataHandler, alpha_monitor, ticker: str):
    """Integrate RTD data with alpha monitoring"""
    
    rtd_processor = RTDSignalProcessor(rtd_handler)
    
    def rtd_update_callback(rtd_data):
        """Callback for RTD data updates"""
        
        try:
            # Process RTD data for additional signals
            rtd_signals = rtd_processor.process_rtd_update(rtd_data)
            
            # Convert significant RTD signals to alpha signals
            for rtd_signal in rtd_signals:
                if rtd_signal['severity'] in ['HIGH', 'CRITICAL']:
                    # Create pseudo-alpha signal from RTD data
                    alpha_signal = create_alpha_signal_from_rtd(rtd_signal, ticker)
                    if alpha_signal and alpha_monitor:
                        alpha_monitor.signal_queue.put(alpha_signal)
            
            # Update RTD display if available
            update_rtd_live_display(rtd_data, ticker)
            
        except Exception as e:
            print(f"Error in RTD callback: {e}")
    
    # Add callback to RTD handler
    rtd_handler.add_update_callback(rtd_update_callback)
    
    return rtd_processor


def create_alpha_signal_from_rtd(rtd_signal: dict, ticker: str):
    """Create alpha signal from RTD signal"""
    
    try:
        from alpha_extractor import AlphaSignal, SignalType, SignalStrength
        
        # Map RTD signal types to alpha signal types
        signal_type_map = {
            'PRICE_MOVEMENT': SignalType.UNUSUAL_VOLUME,
            'VOLUME_SPIKE': SignalType.UNUSUAL_VOLUME,
            'IV_CHANGE': SignalType.IV_SPIKE,
            'WIDE_SPREAD': SignalType.UNUSUAL_VOLUME
        }
        
        signal_type = signal_type_map.get(rtd_signal['type'], SignalType.UNUSUAL_VOLUME)
        
        # Determine strength
        severity_map = {
            'LOW': SignalStrength.WEAK,
            'MEDIUM': SignalStrength.MODERATE,
            'HIGH': SignalStrength.STRONG,
            'CRITICAL': SignalStrength.VERY_STRONG
        }
        
        strength = severity_map.get(rtd_signal['severity'], SignalStrength.MODERATE)
        
        # Calculate confidence based on signal type
        confidence = 60  # Base confidence
        if rtd_signal['type'] == 'VOLUME_SPIKE':
            confidence += min(30, rtd_signal.get('volume_ratio', 1) * 5)
        elif rtd_signal['type'] == 'PRICE_MOVEMENT':
            confidence += min(25, abs(rtd_signal.get('change_percent', 0)) * 2)
        elif rtd_signal['type'] == 'IV_CHANGE':
            confidence += min(20, rtd_signal.get('change_percent', 0))
        
        # Determine direction
        direction = "NEUTRAL"
        if rtd_signal['type'] == 'PRICE_MOVEMENT':
            direction = "BULLISH" if rtd_signal.get('direction') == 'UP' else "BEARISH"
        elif rtd_signal['type'] == 'IV_CHANGE':
            if rtd_signal.get('direction') == 'UP':
                direction = "BULLISH"  # Rising IV often bullish for option buyers
        
        # Create alpha signal
        alpha_signal = AlphaSignal(
            timestamp=rtd_signal['timestamp'],
            signal_type=signal_type,
            strength=strength,
            ticker=ticker,
            option_symbol=rtd_signal['symbol'],
            direction=direction,
            confidence=min(95, confidence),
            entry_price=rtd_signal.get('current_price', 0),
            target_price=None,
            stop_price=None,
            notional_value=0,  # RTD doesn't provide notional
            urgency_score=80 if rtd_signal['severity'] == 'CRITICAL' else 60,
            smart_money_score=70,  # Default score
            metadata={
                'source': 'RTD',
                'rtd_signal_type': rtd_signal['type'],
                'rtd_data': rtd_signal
            },
            trade_recommendation=f"RTD Alert: {rtd_signal['message']}"
        )
        
        return alpha_signal
        
    except Exception as e:
        print(f"Error creating alpha signal from RTD: {e}")
        return None


def update_rtd_live_display(rtd_data: dict, ticker: str):
    """Update live RTD display in UI"""
    
    try:
        # This would update the UI display with latest RTD data
        # Implementation depends on your UI framework
        
        # For now, just log key updates
        high_priority_updates = []
        
        for symbol, data in rtd_data.items():
            priority = data.get('signal_priority', 0)
            if priority > 50:  # High priority symbols
                last_price = data.get('last_price', 0)
                volume = data.get('volume', 0)
                iv = data.get('iv', 0)
                
                high_priority_updates.append(f"{symbol}: ${last_price:.2f} Vol:{volume:,} IV:{iv:.1%}")
        
        if high_priority_updates:
            print(f"[RTD UPDATE] {ticker} - " + " | ".join(high_priority_updates[:3]))
            
    except Exception as e:
        print(f"Error updating RTD display: {e}")


# Example usage functions

def setup_rtd_monitoring_for_signals(analysis_results: dict, ticker: str, log_callback: Callable) -> RTDDataHandler:
    """Set up RTD monitoring based on analysis results"""
    
    try:
        # Initialize RTD handler
        rtd_handler = RTDDataHandler(log_callback)
        
        if not rtd_handler.initialize_rtd_connection():
            log_callback("Failed to initialize RTD connection", ticker, is_error=True)
            return None
        
        # Identify symbols with signals
        signal_symbols = rtd_handler.identify_signal_based_symbols(pd.DataFrame(), analysis_results)
        
        if not signal_symbols:
            log_callback("No symbols with signals identified for RTD", ticker)
            return rtd_handler
        
        # Set up RTD formulas
        if rtd_handler.setup_rtd_formulas(signal_symbols):
            log_callback(f"RTD monitoring set up for {len(signal_symbols)} symbols", ticker)
            
            # Start monitoring
            if rtd_handler.start_rtd_monitoring():
                log_callback("RTD monitoring started successfully", ticker)
            else:
                log_callback("Failed to start RTD monitoring", ticker, is_error=True)
        else:
            log_callback("Failed to set up RTD formulas", ticker, is_error=True)
        
        return rtd_handler
        
    except Exception as e:
        log_callback(f"Error setting up RTD monitoring: {e}", ticker, is_error=True)
        return None


def generate_rtd_report(rtd_handler: RTDDataHandler) -> str:
    """Generate RTD monitoring report"""
    
    if not rtd_handler:
        return "RTD handler not available"
    
    report = "=" * 60 + "\n"
    report += "REAL-TIME DATA MONITORING REPORT\n"
    report += "=" * 60 + "\n\n"
    
    # Get summary
    summary = rtd_handler.get_rtd_summary()
    
    report += f"Monitoring Status: {'ACTIVE' if summary['is_monitoring'] else 'INACTIVE'}\n"
    report += f"Symbols Monitored: {summary['symbols_count']}\n"
    report += f"Data Points Available: {summary['data_points']}\n"
    
    if summary['last_update']:
        report += f"Last Update: {summary['last_update'].strftime('%H:%M:%S')}\n"
    
    report += "\n"
    
    # Priority symbols
    if summary['priority_symbols']:
        report += "TOP PRIORITY SYMBOLS:\n"
        report += "-" * 25 + "\n"
        
        for symbol in summary['priority_symbols']:
            data = rtd_handler.get_current_rtd_data(symbol)
            if data:
                last_price = data.get('last_price', 0)
                volume = data.get('volume', 0)
                iv = data.get('iv', 0)
                priority = data.get('signal_priority', 0)
                
                report += f"• {symbol}: ${last_price:.2f} | Vol: {volume:,} | IV: {iv:.1%} | Priority: {priority:.1f}\n"
        
        report += "\n"
    
    # Active symbols
    if summary['active_symbols']:
        report += f"ACTIVE SYMBOLS ({len(summary['active_symbols'])}):\n"
        report += "-" * 20 + "\n"
        
        for i, symbol in enumerate(summary['active_symbols'][:10], 1):  # Show top 10
            report += f"{i:2d}. {symbol}\n"
        
        if len(summary['active_symbols']) > 10:
            report += f"    ... and {len(summary['active_symbols']) - 10} more\n"
    
    report += "\n" + "=" * 60 + "\n"
    
    return report