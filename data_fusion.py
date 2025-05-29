# data_fusion.py
"""
Data fusion engine for combining multiple data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import re

class DataFusionEngine:
    """Fuses data from multiple sources for comprehensive analysis"""
    
    def __init__(self):
        self.fusion_cache = {}
        self.data_sources = {
            'time_sales': None,
            'option_statistics': None,
            'ticker_info': None,
            'market_data': None
        }
        self.last_fusion_time = None
        
    def merge_data_sources(self, time_sales_df: pd.DataFrame, 
                          option_stats: dict, 
                          ticker_info: dict) -> pd.DataFrame:
        """Merge multiple data sources into unified dataset"""
        if time_sales_df.empty:
            return time_sales_df
        
        # Start with time & sales as base
        merged_df = time_sales_df.copy()
        
        # Add option statistics if available
        if option_stats:
            merged_df = self._merge_option_statistics(merged_df, option_stats)
        
        # Add ticker information
        if ticker_info:
            merged_df = self._merge_ticker_info(merged_df, ticker_info)
        
        # Calculate derived metrics
        merged_df = self._calculate_derived_metrics(merged_df)
        
        # Validate data consistency
        merged_df = self._validate_consistency(merged_df)
        
        self.last_fusion_time = datetime.now()
        return merged_df
    
    def _merge_option_statistics(self, df: pd.DataFrame, 
                                stats: dict) -> pd.DataFrame:
        """Merge option statistics into dataframe"""
        # Extract relevant stats
        if 'total_volume' in stats:
            df['session_total_volume'] = stats['total_volume']
        
        if 'open_interest' in stats:
            df['session_open_interest'] = stats['open_interest']
        
        if 'put_call_ratio' in stats:
            df['session_put_call_ratio'] = stats['put_call_ratio']
        
        if 'average_iv' in stats:
            # Use session average IV if individual IV is missing
            df['IV'] = df['IV'].fillna(stats['average_iv'])
        
        # Volume percentile
        if 'session_total_volume' in df.columns:
            df['volume_percentile'] = (
                df['TradeQuantity'].cumsum() / 
                df['session_total_volume'].iloc[0] * 100
            )
        
        return df
    
    def _merge_ticker_info(self, df: pd.DataFrame, 
                          ticker_info: dict) -> pd.DataFrame:
        """Merge ticker-specific information"""
        if 'underlying_price' in ticker_info:
            # Update any missing underlying prices
            df['Underlying_Price'] = df['Underlying_Price'].fillna(
                ticker_info['underlying_price']
            )
        
        if 'ticker' in ticker_info:
            df['Ticker'] = ticker_info['ticker']
        
        if 'market_cap' in ticker_info:
            df['Market_Cap'] = ticker_info['market_cap']
        
        if 'sector' in ticker_info:
            df['Sector'] = ticker_info['sector']
        
        return df
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional metrics from fused data"""
        # Relative volume
        if 'session_total_volume' in df.columns and 'Volume' in df.columns:
            avg_minute_volume = df['session_total_volume'].iloc[0] / 390  # Trading minutes
            df['relative_volume'] = df['Volume'] / avg_minute_volume
        
        # Moneyness precision
        if all(col in df.columns for col in ['Strike_Price_calc', 'Underlying_Price']):
            df['moneyness_ratio'] = df['Strike_Price_calc'] / df['Underlying_Price']
            df['moneyness_pct'] = (df['moneyness_ratio'] - 1) * 100
        
        # Time decay factor
        if 'DTE_calc' in df.columns:
            df['time_decay_factor'] = np.exp(-df['DTE_calc'] / 365)
        
        # Spread quality
        if all(col in df.columns for col in ['Option_Bid', 'Option_Ask']):
            df['spread_pct'] = (
                (df['Option_Ask'] - df['Option_Bid']) / 
                df['Option_Ask'] * 100
            ).fillna(0)
            df['spread_quality'] = pd.cut(
                df['spread_pct'],
                bins=[0, 2, 5, 10, 100],
                labels=['Excellent', 'Good', 'Fair', 'Poor']
            )
        
        # Trade size category
        if 'TradeQuantity' in df.columns:
            df['trade_size_category'] = pd.cut(
                df['TradeQuantity'],
                bins=[0, 10, 50, 100, 500, np.inf],
                # Continuing data_fusion.py
                labels=['Retail', 'Small', 'Medium', 'Large', 'Institutional']
            )
        
        # Option leverage
        if all(col in df.columns for col in ['Trade_Price', 'Underlying_Price', 'Delta']):
            df['option_leverage'] = (
                df['Underlying_Price'] * df['Delta'].abs() / df['Trade_Price']
            ).fillna(0)
        
        return df
    
    def _validate_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data consistency across sources"""
        # Check for impossible values
        if 'IV' in df.columns:
            df.loc[df['IV'] < 0, 'IV'] = np.nan
            df.loc[df['IV'] > 5, 'IV'] = np.nan  # 500% IV is suspicious
        
        if 'Delta' in df.columns:
            df.loc[df['Delta'].abs() > 1, 'Delta'] = np.nan
        
        # Validate price relationships
        if all(col in df.columns for col in ['Trade_Price', 'Option_Bid', 'Option_Ask']):
            # Trade should be between bid and ask (with some tolerance)
            tolerance = 0.05  # 5 cents
            invalid_trades = (
                (df['Trade_Price'] < df['Option_Bid'] - tolerance) |
                (df['Trade_Price'] > df['Option_Ask'] + tolerance)
            )
            if invalid_trades.any():
                print(f"Warning: {invalid_trades.sum()} trades outside bid-ask spread")
        
        return df
    
    def detect_data_quality_issues(self, df: pd.DataFrame) -> dict:
        """Detect potential data quality issues"""
        issues = {
            'missing_data': {},
            'outliers': {},
            'inconsistencies': [],
            'warnings': []
        }
        
        # Check missing data
        for col in df.columns:
            missing_pct = df[col].isna().sum() / len(df) * 100
            if missing_pct > 0:
                issues['missing_data'][col] = f"{missing_pct:.1f}%"
        
        # Check for outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['IV', 'Trade_Price', 'TradeQuantity']:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                outliers = df[(df[col] < q1) | (df[col] > q99)]
                if len(outliers) > 0:
                    issues['outliers'][col] = len(outliers)
        
        # Check timestamp consistency
        if 'Time_dt' in df.columns:
            time_diffs = df['Time_dt'].diff()
            if time_diffs.max() > pd.Timedelta(minutes=5):
                issues['warnings'].append("Large time gaps detected in data")
        
        # Check price consistency
        if 'Underlying_Price' in df.columns:
            price_changes = df['Underlying_Price'].pct_change().abs()
            if (price_changes > 0.05).any():  # 5% change
                issues['warnings'].append("Large underlying price movements detected")
        
        return issues
    
    def create_analysis_ready_dataset(self, raw_data: dict[str, any]) -> pd.DataFrame:
        """Create analysis-ready dataset from raw data sources"""
        # Extract time & sales data
        time_sales_df = raw_data.get('time_sales', pd.DataFrame())
        if time_sales_df.empty:
            return pd.DataFrame()
        
        # Extract option statistics
        option_stats = self._parse_option_statistics(
            raw_data.get('option_statistics', '')
        )
        
        # Extract ticker info
        ticker_info = {
            'ticker': raw_data.get('ticker'),
            'underlying_price': raw_data.get('underlying_price')
        }
        
        # Merge all sources
        merged_df = self.merge_data_sources(
            time_sales_df, option_stats, ticker_info
        )
        
        # Add metadata
        merged_df['data_source'] = 'real_time_monitor'
        merged_df['fusion_timestamp'] = datetime.now()
        
        return merged_df
    
    def _parse_option_statistics(self, stats_text: str) -> dict:
        """Parse option statistics from text"""
        if not stats_text:
            return {}
        
        parsed_stats = {}
        
        # Common patterns in option statistics
        patterns = {
            'total_volume': r'Volume[:\s]+(\d+)',
            'open_interest': r'Open\s+Interest[:\s]+(\d+)',
            'put_call_ratio': r'Put/Call[:\s]+([\d.]+)',
            'average_iv': r'Avg\s+IV[:\s]+([\d.]+)%?',
            'contracts_traded': r'Contracts[:\s]+(\d+)',
            'dollar_volume': r'Dollar\s+Volume[:\s]+\$?([\d,]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, stats_text, re.IGNORECASE)
            if match:
                value = match.group(1).replace(',', '')
                try:
                    if key in ['put_call_ratio', 'average_iv']:
                        parsed_stats[key] = float(value) / 100 if 'iv' in key else float(value)
                    else:
                        parsed_stats[key] = int(value)
                except ValueError:
                    pass
        
        return parsed_stats