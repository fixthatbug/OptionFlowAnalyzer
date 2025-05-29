# enhanced_tos_data_processor.py
"""
Enhanced TOS Options Time & Sales data processor with incremental updates
Handles raw data extraction, parsing, cleaning, and incremental file updates
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import re
import os
import json
import logging
from pathlib import Path

class TOSDataProcessor:
    """Enhanced processor for TOS Options Time & Sales data with incremental updates"""
    
    def __init__(self, ticker: str, data_directory: str = "Daily_Data", log_callback: Callable = None):
        self.ticker = ticker.upper()
        self.data_directory = data_directory
        self.log = log_callback or self._default_log
        
        # Create directories
        Path(self.data_directory).mkdir(exist_ok=True)
        Path(f"{self.data_directory}/metadata").mkdir(exist_ok=True)
        
        # File paths
        self.daily_file_path = self._get_daily_file_path()
        self.metadata_file = self._get_metadata_file_path()
        
        # Expected columns for cleaned data
        self.output_columns = [
            'Time', 'Option_Description_orig', 'Expiration_Date', 'Strike_Price', 
            'Option_Type', 'TradeQuantity', 'Trade_Price', 'Option_Bid', 
            'Option_Ask', 'Delta', 'IV', 'Underlying_Price', 'Condition', 'Exchange'
        ]
        
        # Load existing data info
        self.last_trade_info = self._load_last_trade_info()
        
        self.log(f"TOSDataProcessor initialized for {ticker}")
    
    def _default_log(self, message: str, is_error: bool = False):
        """Default logging function"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        level = "ERROR" if is_error else "INFO"
        print(f"[TOS_PROCESSOR] [{timestamp}] [{level}] {message}")
    
    def _get_daily_file_path(self) -> str:
        """Get daily file path for ticker"""
        today = datetime.now().strftime("%Y%m%d")
        return f"{self.data_directory}/{self.ticker}_{today}_cleaned.csv"
    
    def _get_metadata_file_path(self) -> str:
        """Get metadata file path"""
        today = datetime.now().strftime("%Y%m%d")
        return f"{self.data_directory}/metadata/{self.ticker}_{today}_metadata.json"
    
    def _load_last_trade_info(self) -> dict:
        """Load information about the last processed trade"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.log(f"Error loading metadata: {e}", is_error=True)
        
        return {
            'last_trade_time': None,
            'last_trade_hash': None,
            'total_trades_processed': 0,
            'last_update': None
        }
    
    def _save_last_trade_info(self, last_trade_time: str, last_trade_hash: str, total_trades: int):
        """Save information about the last processed trade"""
        try:
            metadata = {
                'last_trade_time': last_trade_time,
                'last_trade_hash': last_trade_hash,
                'total_trades_processed': total_trades,
                'last_update': datetime.now().isoformat(),
                'ticker': self.ticker
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.log(f"Error saving metadata: {e}", is_error=True)
    
    def _create_trade_hash(self, row: pd.Series) -> str:
        """Create unique hash for a trade row"""
        # Create hash from time + symbol + qty + price for uniqueness
        hash_string = f"{row['Time']}_{row['Option_Description_orig']}_{row['TradeQuantity']}_{row['Trade_Price']}"
        return str(hash(hash_string))
    
    def parse_raw_tos_data(self, raw_data: str) -> pd.DataFrame:
        """Parse raw TOS data string into structured DataFrame"""
        
        if not raw_data or not raw_data.strip():
            self.log("Empty raw data provided")
            return pd.DataFrame()
        
        self.log("Starting raw data parsing...")
        
        # Split into lines and clean
        lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
        
        if not lines:
            self.log("No valid lines found in raw data")
            return pd.DataFrame()
        
        parsed_rows = []
        
        for line_num, line in enumerate(lines, 1):
            try:
                parsed_row = self._parse_single_line(line)
                if parsed_row:
                    parsed_rows.append(parsed_row)
            except Exception as e:
                self.log(f"Error parsing line {line_num}: {e}", is_error=True)
                continue
        
        if not parsed_rows:
            self.log("No valid trades parsed from raw data")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(parsed_rows)
        
        # Filter out pre-market/invalid data
        df = self._filter_valid_trades(df)
        
        self.log(f"Successfully parsed {len(df)} valid trades from {len(lines)} raw lines")
        
        return df
    
    def _parse_single_line(self, line: str) -> Optional[dict]:
        """Parse a single line of raw TOS data"""
        
        # Split by tabs (TOS uses tab separation)
        parts = line.split('\t')
        
        if len(parts) < 9:  # Minimum required fields
            return None
        
        # Extract basic fields
        time_str = parts[0].strip()
        option_desc = parts[1].strip()
        qty_str = parts[2].strip()
        price_str = parts[3].strip()
        exchange = parts[4].strip()
        market_str = parts[5].strip() if len(parts) > 5 else ""
        delta_str = parts[6].strip() if len(parts) > 6 else ""
        iv_str = parts[7].strip() if len(parts) > 7 else ""
        underlying_str = parts[8].strip() if len(parts) > 8 else ""
        condition = parts[9].strip() if len(parts) > 9 else ""
        
        # Skip invalid/pre-market data
        if qty_str == "0" or price_str in ["N/A", "0", ""]:
            return None
        
        # Parse numeric fields
        try:
            qty = int(qty_str)
            price = float(price_str)
            underlying_price = float(underlying_str) if underlying_str and underlying_str != "N/A" else None
        except ValueError:
            return None
        
        # Parse market data (bid x ask)
        bid_price, ask_price = self._parse_market_data(market_str)
        
        # Parse delta
        delta = self._parse_delta(delta_str)
        
        # Parse IV
        iv = self._parse_iv(iv_str)
        
        # Parse option description
        exp_date, strike_price, option_type = self._parse_option_description(option_desc)
        
        return {
            'Time': time_str,
            'Option_Description_orig': option_desc,
            'Expiration_Date': exp_date,
            'Strike_Price': strike_price,
            'Option_Type': option_type,
            'TradeQuantity': qty,
            'Trade_Price': price,
            'Option_Bid': bid_price,
            'Option_Ask': ask_price,
            'Delta': delta,
            'IV': iv,
            'Underlying_Price': underlying_price,
            'Condition': condition,
            'Exchange': exchange
        }
    
    def _parse_market_data(self, market_str: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse market data string (e.g., '1.68x1.71') into bid and ask"""
        
        if not market_str or market_str == "N/A":
            return None, None
        
        try:
            # Handle format like "1.68x1.71"
            if 'x' in market_str:
                bid_str, ask_str = market_str.split('x')
                bid = float(bid_str.strip()) if bid_str.strip() != "N/A" else None
                ask = float(ask_str.strip()) if ask_str.strip() != "N/A" else None
                return bid, ask
        except:
            pass
        
        return None, None
    
    def _parse_delta(self, delta_str: str) -> Optional[float]:
        """Parse delta string into float"""
        
        if not delta_str or delta_str in ["N/A", ""]:
            return None
        
        try:
            return float(delta_str)
        except ValueError:
            return None
    
    def _parse_iv(self, iv_str: str) -> Optional[float]:
        """Parse IV string (e.g., '35.94%') into float"""
        
        if not iv_str or iv_str in ["N/A", ""]:
            return None
        
        try:
            # Remove % sign if present
            clean_iv = iv_str.replace('%', '').strip()
            iv_value = float(clean_iv)
            
            # Convert percentage to decimal if it looks like a percentage
            if iv_value > 5:  # Assume values > 5 are percentages
                iv_value = iv_value / 100
            
            return iv_value
        except ValueError:
            return None
    
    def _parse_option_description(self, option_desc: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Parse option description (e.g., '30 MAY 25 200 P') into components"""
        
        if not option_desc:
            return None, None, None
        
        try:
            # Pattern for option description: "DAY MONTH YEAR STRIKE TYPE"
            # Examples: "30 MAY 25 200 P", "13 JUN 25 275 C"
            parts = option_desc.split()
            
            if len(parts) < 5:
                return None, None, None
            
            # Extract components
            day = parts[0]
            month = parts[1]
            year = parts[2]
            strike = float(parts[3])
            option_type = "Call" if parts[4].upper() == 'C' else "Put"
            
            # Create expiration date string
            exp_date = f"{day} {month} {year}"
            
            return exp_date, strike, option_type
            
        except (ValueError, IndexError):
            return None, None, None
    
    def _filter_valid_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid/pre-market trades"""
        
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Remove trades with zero quantity
        df = df[df['TradeQuantity'] > 0]
        
        # Remove trades with zero/null prices
        df = df[df['Trade_Price'] > 0]
        
        # Remove trades with null underlying prices
        df = df[df['Underlying_Price'].notna()]
        df = df[df['Underlying_Price'] > 0]
        
        # Remove clearly invalid option descriptions
        df = df[df['Option_Description_orig'].notna()]
        df = df[df['Option_Description_orig'] != '']
        
        filtered_count = len(df)
        
        if initial_count != filtered_count:
            self.log(f"Filtered out {initial_count - filtered_count} invalid trades")
        
        return df
    
    def process_incremental_update(self, raw_data: str) -> Tuple[pd.DataFrame, int]:
        """Process raw data incrementally, only adding new trades"""
        
        # Parse raw data
        new_df = self.parse_raw_tos_data(raw_data)
        
        if new_df.empty:
            self.log("No new valid trades to process")
            return pd.DataFrame(), 0
        
        # If no previous data, process all
        if not self.last_trade_info['last_trade_time']:
            self.log("No previous data found, processing all trades")
            return self._save_all_trades(new_df)
        
        # Find incremental trades
        incremental_df = self._find_incremental_trades(new_df)
        
        if incremental_df.empty:
            self.log("No new trades found since last update")
            return pd.DataFrame(), 0
        
        # Append to existing file
        return self._append_new_trades(incremental_df)
    
    def _find_incremental_trades(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """Find trades that occurred after the last processed trade"""
        
        last_trade_time = self.last_trade_info['last_trade_time']
        last_trade_hash = self.last_trade_info['last_trade_hash']
        
        self.log(f"Looking for trades after {last_trade_time}")
        
        # Convert time strings for comparison
        new_df['Time_for_comparison'] = pd.to_datetime(
            new_df['Time'], format='%H:%M:%S', errors='coerce'
        ).dt.time
        
        last_time = datetime.strptime(last_trade_time, '%H:%M:%S').time()
        
        # Find trades after the last trade time
        mask = new_df['Time_for_comparison'] > last_time
        
        # If there are trades at the exact same time, use hash to avoid duplicates
        same_time_mask = new_df['Time_for_comparison'] == last_time
        if same_time_mask.any():
            # Create hashes for same-time trades
            same_time_df = new_df[same_time_mask].copy()
            same_time_df['trade_hash'] = same_time_df.apply(self._create_trade_hash, axis=1)
            
            # Only include trades with different hashes
            new_same_time = same_time_df[same_time_df['trade_hash'] != last_trade_hash]
            
            # Combine with trades after the time
            after_time_df = new_df[mask]
            incremental_df = pd.concat([new_same_time, after_time_df], ignore_index=True)
        else:
            incremental_df = new_df[mask]
        
        # Clean up temporary column
        if 'Time_for_comparison' in incremental_df.columns:
            incremental_df = incremental_df.drop('Time_for_comparison', axis=1)
        if 'trade_hash' in incremental_df.columns:
            incremental_df = incremental_df.drop('trade_hash', axis=1)
        
        # Sort by time
        incremental_df = incremental_df.sort_values('Time').reset_index(drop=True)
        
        self.log(f"Found {len(incremental_df)} new trades to process")
        
        return incremental_df
    
    def _save_all_trades(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Save all trades to new daily file"""
        
        try:
            # Ensure columns are in correct order
            df_ordered = df[self.output_columns].copy()
            
            # Save to CSV
            df_ordered.to_csv(self.daily_file_path, index=False)
            
            # Update metadata
            if not df_ordered.empty:
                last_trade = df_ordered.iloc[-1]
                last_trade_hash = self._create_trade_hash(last_trade)
                self._save_last_trade_info(
                    last_trade['Time'], 
                    last_trade_hash, 
                    len(df_ordered)
                )
            
            self.log(f"Saved {len(df_ordered)} trades to {self.daily_file_path}")
            
            return df_ordered, len(df_ordered)
            
        except Exception as e:
            self.log(f"Error saving trades: {e}", is_error=True)
            return pd.DataFrame(), 0
    
    def _append_new_trades(self, incremental_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Append new trades to existing daily file"""
        
        try:
            # Ensure columns are in correct order
            incremental_ordered = incremental_df[self.output_columns].copy()
            
            # Load existing data to get total count
            existing_count = 0
            if os.path.exists(self.daily_file_path):
                existing_df = pd.read_csv(self.daily_file_path)
                existing_count = len(existing_df)
            
            # Append to CSV
            incremental_ordered.to_csv(
                self.daily_file_path, 
                mode='a', 
                header=not os.path.exists(self.daily_file_path),
                index=False
            )
            
            # Update metadata
            if not incremental_ordered.empty:
                last_trade = incremental_ordered.iloc[-1]
                last_trade_hash = self._create_trade_hash(last_trade)
                total_trades = existing_count + len(incremental_ordered)
                
                self._save_last_trade_info(
                    last_trade['Time'], 
                    last_trade_hash, 
                    total_trades
                )
            
            self.log(f"Appended {len(incremental_ordered)} new trades to {self.daily_file_path}")
            
            return incremental_ordered, len(incremental_ordered)
            
        except Exception as e:
            self.log(f"Error appending trades: {e}", is_error=True)
            return pd.DataFrame(), 0
    
    def get_current_data_summary(self) -> dict:
        """Get summary of current day's data"""
        
        summary = {
            'ticker': self.ticker,
            'date': datetime.now().strftime("%Y-%m-%d"),
            'file_exists': os.path.exists(self.daily_file_path),
            'total_trades': 0,
            'last_trade_time': None,
            'file_size_mb': 0
        }
        
        try:
            if os.path.exists(self.daily_file_path):
                df = pd.read_csv(self.daily_file_path)
                summary['total_trades'] = len(df)
                
                if not df.empty:
                    summary['last_trade_time'] = df.iloc[-1]['Time']
                
                summary['file_size_mb'] = os.path.getsize(self.daily_file_path) / (1024 * 1024)
            
            # Add metadata info
            summary.update(self.last_trade_info)
            
        except Exception as e:
            self.log(f"Error getting data summary: {e}", is_error=True)
        
        return summary
    
    def validate_data_integrity(self) -> dict:
        """Validate the integrity of processed data"""
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'total_trades': 0,
            'data_quality_score': 0
        }
        
        try:
            if not os.path.exists(self.daily_file_path):
                validation_results['issues'].append("Daily file does not exist")
                validation_results['is_valid'] = False
                return validation_results
            
            df = pd.read_csv(self.daily_file_path)
            validation_results['total_trades'] = len(df)
            
            if df.empty:
                validation_results['issues'].append("No data in file")
                validation_results['is_valid'] = False
                return validation_results
            
            # Check for required columns
            missing_cols = [col for col in self.output_columns if col not in df.columns]
            if missing_cols:
                validation_results['issues'].append(f"Missing columns: {missing_cols}")
                validation_results['is_valid'] = False
            
            # Check for duplicate trades
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                validation_results['issues'].append(f"Found {duplicates} duplicate trades")
            
            # Check time ordering
            if 'Time' in df.columns:
                time_series = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
                if not time_series.is_monotonic_increasing:
                    validation_results['issues'].append("Trades are not in chronological order")
            
            # Check data completeness
            null_counts = df.isnull().sum()
            critical_nulls = null_counts[['Time', 'TradeQuantity', 'Trade_Price']].sum()
            if critical_nulls > 0:
                validation_results['issues'].append(f"Missing critical data: {critical_nulls} null values")
            
            # Calculate quality score
            total_possible_issues = 5
            issues_found = len(validation_results['issues'])
            validation_results['data_quality_score'] = max(0, 100 - (issues_found / total_possible_issues * 100))
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {e}")
            validation_results['is_valid'] = False
        
        return validation_results


# Utility functions for integration with existing system

def process_tos_raw_data(ticker: str, raw_data: str, data_directory: str = "Daily_Data", 
                        log_callback: Callable = None) -> Tuple[pd.DataFrame, dict]:
    """
    Main function to process TOS raw data with incremental updates
    
    Args:
        ticker: Stock ticker symbol
        raw_data: Raw TOS data string
        data_directory: Directory to save processed data
        log_callback: Logging function
    
    Returns:
        Tuple of (processed_dataframe, processing_summary)
    """
    
    processor = TOSDataProcessor(ticker, data_directory, log_callback)
    
    # Process incremental update
    new_trades_df, new_trade_count = processor.process_incremental_update(raw_data)
    
    # Get summary
    summary = processor.get_current_data_summary()
    summary['new_trades_processed'] = new_trade_count
    summary['processing_timestamp'] = datetime.now().isoformat()
    
    # Validate data integrity
    validation = processor.validate_data_integrity()
    summary['validation'] = validation
    
    return new_trades_df, summary


def get_daily_data_for_analysis(ticker: str, data_directory: str = "Daily_Data") -> pd.DataFrame:
    """
    Load complete daily data for analysis
    
    Args:
        ticker: Stock ticker symbol
        data_directory: Directory containing processed data
    
    Returns:
        Complete daily DataFrame for analysis
    """
    
    processor = TOSDataProcessor(ticker, data_directory)
    
    try:
        if os.path.exists(processor.daily_file_path):
            return pd.read_csv(processor.daily_file_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading daily data: {e}")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Example raw data processing
    sample_raw_data = """10:11:57	30 MAY 25 200 P	2	1.70	CBOE	1.68x1.71	-.41	35.94%	201.13		
10:11:57	30 MAY 25 200 P	1	1.70	CBOE	1.68x1.71	-.41	35.94%	201.13		
10:11:57	13 JUN 25 275 C	1	.03	CBOE	.01x.03	.00	54.98%	201.13		"""
    
    # Process the data
    processed_df, summary = process_tos_raw_data("AAPL", sample_raw_data)
    
    print("Processing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    if not processed_df.empty:
        print(f"\nProcessed {len(processed_df)} trades:")
        print(processed_df.to_string())