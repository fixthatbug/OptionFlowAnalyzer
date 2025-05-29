# Enhanced monitoring_manager.py with incremental data processing

import threading
import queue
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import pandas as pd
import json
import os
from pathlib import Path
import tos_data_grabber
import data_utils

class MonitoringThread(threading.Thread):
    """Enhanced thread for monitoring with incremental processing"""
    
    def __init__(self, ticker: str, window_handle: int, coordinates: dict, 
                 interval_ms: int, log_func: Callable):
        super().__init__(daemon=True)
        self.ticker = ticker
        self.window_handle = window_handle
        self.coordinates = coordinates
        self.interval_ms = interval_ms
        self.log_func = log_func
        self.is_running = False
        self.data_queue = queue.Queue()
        
        # Incremental processing state
        self.last_trade_info = self._load_last_trade_info()
        self.daily_file_path = self._get_daily_file_path()
        self.metadata_file = self._get_metadata_file_path()
        self.total_new_trades = 0
        self.extraction_errors = 0
        self.max_errors = 5
        
    def _get_daily_file_path(self) -> str:
        """Get path for daily data file"""
        today = datetime.now().strftime("%Y%m%d")
        data_dir = Path("Daily_Data")
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / f"{self.ticker}_{today}_cleaned.csv")
    
    def _get_metadata_file_path(self) -> str:
        """Get path for metadata file"""
        today = datetime.now().strftime("%Y%m%d")
        metadata_dir = Path("Daily_Data") / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        return str(metadata_dir / f"{self.ticker}_{today}_metadata.json")
    
    def _load_last_trade_info(self) -> dict:
        """Load last trade information from metadata"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return {
                        'last_trade_time': metadata.get('last_trade_time'),
                        'last_trade_hash': metadata.get('last_trade_hash'),
                        'total_trades_processed': metadata.get('total_trades_processed', 0)
                    }
        except Exception as e:
            self.log_func(f"Error loading metadata: {e}", self.ticker)
        
        return {
            'last_trade_time': None,
            'last_trade_hash': None,
            'total_trades_processed': 0
        }
    
    def _save_last_trade_info(self, last_trade_time: str, last_trade_hash: str):
        """Save last trade information to metadata"""
        try:
            metadata = {
                'ticker': self.ticker,
                'last_trade_time': last_trade_time,
                'last_trade_hash': last_trade_hash,
                'total_trades_processed': self.last_trade_info['total_trades_processed'] + self.total_new_trades,
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.log_func(f"Error saving metadata: {e}", self.ticker)
    
    def run(self):
        """Main monitoring loop with incremental processing"""
        self.is_running = True
        self.log_func(f"Monitoring thread started for {self.ticker} (incremental mode)", self.ticker)
        
        # Initial extraction to catch up
        self._perform_initial_extraction()
        
        while self.is_running:
            try:
                # Extract incremental data
                new_data = self._extract_incremental_data()
                
                if new_data and new_data.get('new_trades_count', 0) > 0:
                    # Queue the new data
                    self.data_queue.put(new_data)
                    self.extraction_errors = 0
                    
                    # Update statistics
                    self.total_new_trades += new_data['new_trades_count']
                    
                    self.log_func(
                        f"Extracted {new_data['new_trades_count']} new trades "
                        f"(total today: {self.total_new_trades})", 
                        self.ticker
                    )
                
                # Sleep for interval
                time.sleep(self.interval_ms / 1000.0)
                
            except Exception as e:
                self.log_func(f"Error in monitoring thread: {e}", self.ticker, is_error=True)
                self.extraction_errors += 1
                
                if self.extraction_errors >= self.max_errors:
                    self.log_func(
                        f"Too many errors. Stopping monitoring for {self.ticker}", 
                        self.ticker, is_error=True
                    )
                    self.is_running = False
                
                time.sleep(1)
    
    def _perform_initial_extraction(self):
        """Perform initial extraction to catch up with any missed trades"""
        try:
            self.log_func(f"Performing initial data extraction for {self.ticker}...", self.ticker)
            
            # Extract all current data
            raw_data = tos_data_grabber.copy_time_sales_data(
                self.window_handle, self.coordinates, self.log_func
            )
            
            if raw_data:
                # Process with incremental logic
                cleaned_df, summary = data_utils.process_tos_options_data(
                    raw_data, self.ticker, self.log_func
                )
                
                if not cleaned_df.empty:
                    self.total_new_trades = summary.get('new_trades_processed', 0)
                    
                    # Update last trade info
                    last_trade = cleaned_df.iloc[-1]
                    self.last_trade_info['last_trade_time'] = last_trade['Time']
                    self.last_trade_info['last_trade_hash'] = data_utils._create_tos_trade_hash(last_trade)
                    
                    self.log_func(
                        f"Initial extraction complete: {self.total_new_trades} trades processed", 
                        self.ticker
                    )
                    
                    # Queue initial data for UI update
                    self.data_queue.put({
                        'time_sales_data': raw_data,
                        'cleaned_data': cleaned_df,
                        'new_trades_count': self.total_new_trades,
                        'extraction_time': datetime.now()
                    })
                    
        except Exception as e:
            self.log_func(f"Error in initial extraction: {e}", self.ticker, is_error=True)
    
    def _extract_incremental_data(self) -> Optional[dict]:
        """Extract only new trades since last extraction"""
        try:
            # Ensure window is still valid
            if not tos_data_grabber.is_window_valid(self.window_handle):
                self.log_func(f"Window handle invalid for {self.ticker}", self.ticker, is_error=True)
                return None
            
            # Extract current data
            raw_data = tos_data_grabber.copy_time_sales_data(
                self.window_handle, self.coordinates, self.log_func
            )
            
            if not raw_data:
                return None
            
            # Process incrementally
            cleaned_df, summary = data_utils.process_tos_options_data(
                raw_data, self.ticker, self.log_func
            )
            
            new_trades_count = summary.get('new_trades_processed', 0)
            
            if new_trades_count > 0 and not cleaned_df.empty:
                # Update last trade info
                last_trade = cleaned_df.iloc[-1]
                self.last_trade_info['last_trade_time'] = last_trade['Time']
                self.last_trade_info['last_trade_hash'] = data_utils._create_tos_trade_hash(last_trade)
                self.last_trade_info['total_trades_processed'] += new_trades_count
                
                # Save metadata
                self._save_last_trade_info(
                    self.last_trade_info['last_trade_time'],
                    self.last_trade_info['last_trade_hash']
                )
                
                return {
                    'time_sales_data': raw_data,
                    'cleaned_data': cleaned_df,
                    'new_trades_count': new_trades_count,
                    'extraction_time': datetime.now(),
                    'option_statistics': self._extract_option_statistics()
                }
            
            return {
                'new_trades_count': 0,
                'extraction_time': datetime.now(),
                'option_statistics': self._extract_option_statistics()
            }
            
        except Exception as e:
            self.log_func(f"Error extracting incremental data: {e}", self.ticker, is_error=True)
            return None
    
    def _extract_option_statistics(self) -> Optional[str]:
        """Extract option statistics from ToS window"""
        try:
            if self.coordinates.get("option_statistics"):
                stats_coord = self.coordinates["option_statistics"]
                stats_text = tos_data_grabber.extract_text_from_region(
                    self.window_handle,
                    stats_coord["x"], stats_coord["y"],
                    stats_coord["width"], stats_coord["height"]
                )
                return stats_text
        except Exception as e:
            self.log_func(f"Error extracting option statistics: {e}", self.ticker)
        
        return None
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        self.log_func(f"Stopping monitoring for {self.ticker} ({self.total_new_trades} trades today)", self.ticker)
    
    def get_status(self) -> dict:
        """Get current monitoring status"""
        return {
            'is_running': self.is_running,
            'total_new_trades': self.total_new_trades,
            'last_trade_time': self.last_trade_info.get('last_trade_time'),
            'extraction_errors': self.extraction_errors,
            'queue_size': self.data_queue.qsize()
        }

class MonitoringManager:
    """Enhanced monitoring manager with incremental processing support"""
    
    def __init__(self, log_func: Callable):
        self.log_func = log_func
        self.active_monitors: dict[str, MonitoringThread] = {}
        self.monitoring_interval = 1000  # Default 1 second
        self.daily_data_cache: dict[str, pd.DataFrame] = {}
        
    def start_monitoring(self, ticker: str, window_handle: int, 
                        coordinates: dict) -> bool:
        """Start monitoring for a ticker with incremental processing"""
        if ticker in self.active_monitors:
            self.log_func(f"Monitoring already active for {ticker}", ticker)
            return False
        
        try:
            # Validate window and coordinates
            if not tos_data_grabber.is_window_valid(window_handle):
                self.log_func(f"Invalid window handle for {ticker}", ticker, is_error=True)
                return False
            
            if not coordinates:
                self.log_func(f"No coordinates configured for {ticker}", ticker, is_error=True)
                return False
            
            # Create and start monitoring thread
            monitor_thread = MonitoringThread(
                ticker, window_handle, coordinates,
                self.monitoring_interval, self.log_func
            )
            monitor_thread.start()
            
            self.active_monitors[ticker] = monitor_thread
            
            # Load existing daily data into cache
            self._load_daily_data_to_cache(ticker)
            
            self.log_func(f"Started incremental monitoring for {ticker}", ticker)
            return True
            
        except Exception as e:
            self.log_func(f"Failed to start monitoring for {ticker}: {e}", 
                        ticker, is_error=True)
            return False
    
    def stop_monitoring(self, ticker: str):
        """Stop monitoring for a ticker"""
        if ticker in self.active_monitors:
            monitor = self.active_monitors[ticker]
            monitor.stop()
            monitor.join(timeout=2)
            
            # Get final status
            status = monitor.get_status()
            
            del self.active_monitors[ticker]
            
            # Clear cache
            if ticker in self.daily_data_cache:
                del self.daily_data_cache[ticker]
            
            self.log_func(
                f"Stopped monitoring for {ticker} "
                f"(processed {status['total_new_trades']} trades today)", 
                ticker
            )
    
    def get_new_data(self, ticker: str) -> Optional[dict]:
        """Get new data for ticker since last fetch"""
        if ticker not in self.active_monitors:
            return None
        
        monitor = self.active_monitors[ticker]
        
        # Collect all queued data
        all_new_data = []
        latest_stats = None
        total_new_trades = 0
        
        while not monitor.data_queue.empty():
            try:
                data = monitor.data_queue.get_nowait()
                all_new_data.append(data)
                
                if data.get('option_statistics'):
                    latest_stats = data['option_statistics']
                
                total_new_trades += data.get('new_trades_count', 0)
                
            except queue.Empty:
                break
        
        if not all_new_data:
            return None
        
        # Combine cleaned dataframes
        cleaned_dfs = []
        for data in all_new_data:
            if 'cleaned_data' in data and isinstance(data['cleaned_data'], pd.DataFrame):
                if not data['cleaned_data'].empty:
                    cleaned_dfs.append(data['cleaned_data'])
        
        result = {
            'new_trades_count': total_new_trades,
            'extraction_time': datetime.now()
        }
        
        if cleaned_dfs:
            combined_df = pd.concat(cleaned_dfs, ignore_index=True)
            result['cleaned_data'] = combined_df
            
            # Update cache
            self._update_daily_cache(ticker, combined_df)
            
            # Get raw format for compatibility
            result['time_sales_data'] = self._convert_to_raw_format(combined_df)
        
        if latest_stats:
            result['option_statistics'] = latest_stats
        
        return result
    
    def get_monitoring_status(self, ticker: str) -> dict:
        """Get detailed monitoring status for ticker"""
        if ticker not in self.active_monitors:
            return {"active": False}
        
        monitor = self.active_monitors[ticker]
        status = monitor.get_status()
        
        # Add cache info
        status['cached_trades'] = len(self.daily_data_cache.get(ticker, pd.DataFrame()))
        
        return status
    
    def get_daily_data(self, ticker: str) -> pd.DataFrame:
        """Get all daily data for ticker from cache"""
        return self.daily_data_cache.get(ticker, pd.DataFrame())
    
    def _load_daily_data_to_cache(self, ticker: str):
        """Load existing daily data file to cache"""
        try:
            today = datetime.now().strftime("%Y%m%d")
            daily_file = Path("Daily_Data") / f"{ticker}_{today}_cleaned.csv"
            
            if daily_file.exists():
                df = pd.read_csv(daily_file)
                self.daily_data_cache[ticker] = df
                self.log_func(
                    f"Loaded {len(df)} existing trades for {ticker} to cache", 
                    ticker
                )
        except Exception as e:
            self.log_func(f"Error loading daily data to cache: {e}", ticker)
            self.daily_data_cache[ticker] = pd.DataFrame()
    
    def _update_daily_cache(self, ticker: str, new_data: pd.DataFrame):
        """Update daily cache with new data"""
        if ticker not in self.daily_data_cache:
            self.daily_data_cache[ticker] = new_data
        else:
            # Append new data
            self.daily_data_cache[ticker] = pd.concat(
                [self.daily_data_cache[ticker], new_data], 
                ignore_index=True
            )
    
    def _convert_to_raw_format(self, df: pd.DataFrame) -> str:
        """Convert cleaned dataframe back to raw format for compatibility"""
        if df.empty:
            return ""
        
        lines = []
        for _, row in df.iterrows():
            # Reconstruct tab-delimited line
            parts = [
                str(row.get('Time', '')),
                str(row.get('Option_Description_orig', '')),
                str(int(row.get('TradeQuantity', 0))),
                str(row.get('Trade_Price', '')),
                str(row.get('Exchange', '')),
                f"{row.get('Option_Bid', 'N/A')}x{row.get('Option_Ask', 'N/A')}",
                str(row.get('Delta', 'N/A')),
                f"{row.get('IV', 'N/A')*100:.2f}%" if pd.notna(row.get('IV')) else 'N/A',
                str(row.get('Underlying_Price', '')),
                str(row.get('Condition', ''))
            ]
            lines.append('\t'.join(parts))
        
        return '\n'.join(lines)
    
    def set_monitoring_interval(self, interval_ms: int):
        """Set monitoring interval for future monitors"""
        self.monitoring_interval = max(100, interval_ms)  # Minimum 100ms
    
    def get_all_monitoring_summary(self) -> dict:
        """Get summary of all active monitoring"""
        summary = {
            'active_tickers': list(self.active_monitors.keys()),
            'total_monitors': len(self.active_monitors),
            'monitors': {}
        }
        
        for ticker, monitor in self.active_monitors.items():
            status = monitor.get_status()
            summary['monitors'][ticker] = {
                'is_running': status['is_running'],
                'trades_today': status['total_new_trades'],
                'last_trade': status['last_trade_time'],
                'errors': status['extraction_errors']
            }
        
        return summary