# analysis_modules/unusual_activity_detector.py
"""
Detector for unusual options activity (UOA)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
from scipy import stats

class UnusualActivityDetector:
    """Detects unusual options activity patterns"""
    
    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days
        self.historical_cache = {}
        self.detection_thresholds = {
            'volume_multiplier': 3.0,
            'oi_change_pct': 25.0,
            'large_trade_size': 100,
            'premium_threshold': 50000,
            'iv_spike_pct': 20.0
        }
    
    def detect_unusual_activity(self, df: pd.DataFrame, 
                              historical_data: Optional[pd.DataFrame] = None) -> dict:
        """Main detection function for unusual activity"""
        detections = {
            'volume_unusual': self._detect_unusual_volume(df, historical_data),
            'oi_unusual': self._detect_unusual_oi_changes(df, historical_data),
            'size_unusual': self._detect_unusual_trade_sizes(df),
            'premium_unusual': self._detect_unusual_premium(df),
            'iv_unusual': self._detect_unusual_iv(df),
            'pattern_unusual': self._detect_unusual_patterns(df),
            'summary': {}
        }
        
        # Create summary
        detections['summary'] = self._create_uoa_summary(detections)
        
        return detections
    
    def _detect_unusual_volume(self, df: pd.DataFrame, 
                              historical: Optional[pd.DataFrame]) -> list[dict]:
        """Detect unusual volume activity"""
        unusual_volume = []
        
        # Group by option symbol
        volume_by_symbol = df.groupby('StandardOptionSymbol').agg({
            'TradeQuantity': 'sum',
            'NotionalValue': 'sum',
            'Trade_Price': 'mean'
        }).reset_index()
        
        # If we have historical data, compare
        if historical is not None and not historical.empty:
            # Calculate historical average volume
            hist_avg = historical.groupby('StandardOptionSymbol')['TradeQuantity'].mean()
            
            for _, row in volume_by_symbol.iterrows():
                symbol = row['StandardOptionSymbol']
                current_vol = row['TradeQuantity']
                
                if symbol in hist_avg.index:
                    avg_vol = hist_avg.loc[symbol]
                    if pd.api.types.is_scalar(avg_vol) and pd.api.types.is_scalar(current_vol):
                        if avg_vol > 0 and current_vol > (avg_vol * self.detection_thresholds['volume_multiplier']):
                            unusual_volume.append({
                                'symbol': symbol,
                                'current_volume': current_vol,
                                'average_volume': avg_vol,
                                'volume_ratio': current_vol / avg_vol,
                                'notional_value': row['NotionalValue'],
                                'detection_type': 'Historical Comparison'
                            })
        else:
            # Use relative volume within session
            total_volume = volume_by_symbol['TradeQuantity'].sum()
            avg_volume_per_symbol = total_volume / len(volume_by_symbol)
            
            for _, row in volume_by_symbol.iterrows():
                if row['TradeQuantity'] > avg_volume_per_symbol * self.detection_thresholds['volume_multiplier']:
                    unusual_volume.append({
                        'symbol': row['StandardOptionSymbol'],
                        'current_volume': row['TradeQuantity'],
                        'session_average': avg_volume_per_symbol,
                        'volume_ratio': row['TradeQuantity'] / avg_volume_per_symbol,
                        'notional_value': row['NotionalValue'],
                        'detection_type': 'Session Relative'
                    })
        
        return unusual_volume
    
    def _detect_unusual_oi_changes(self, df: pd.DataFrame, 
                                  historical: Optional[pd.DataFrame]) -> list[dict]:
        """Detect unusual open interest changes"""
        unusual_oi = []
        
        # This requires OI data which might come from option statistics
        # For now, we'll look at volume vs typical OI ratios
        volume_by_symbol = df.groupby('StandardOptionSymbol')['TradeQuantity'].sum()
        
        # High volume relative to typical OI suggests unusual activity
        for symbol, volume in volume_by_symbol.items():
            # Estimate if volume is unusually high for OI
            # This is a simplified approach
            if volume > self.detection_thresholds['large_trade_size'] * 10:
                unusual_oi.append({
                    'symbol': symbol,
                    'volume': volume,
                    'estimated_oi_impact': 'High',
                    'detection_type': 'Volume-based Estimate'
                })
        
        return unusual_oi
    
    def _detect_unusual_trade_sizes(self, df: pd.DataFrame) -> list[dict]:
        """Detect unusually large trades"""
        unusual_sizes = []
        
        # Calculate size percentiles
        size_p95 = df['TradeQuantity'].quantile(0.95)
        size_p99 = df['TradeQuantity'].quantile(0.99)
        
        # Find large trades
        large_trades = df[df['TradeQuantity'] >= self.detection_thresholds['large_trade_size']]
        
        for _, trade in large_trades.iterrows():
            size_percentile = stats.percentileofscore(df['TradeQuantity'], trade['TradeQuantity'])
            
            unusual_sizes.append({
                'symbol': trade['StandardOptionSymbol'],
                'time': trade['Time'],
                'size': trade['TradeQuantity'],
                'notional': trade['NotionalValue'],
                'percentile': size_percentile,
                'vs_p95': trade['TradeQuantity'] / size_p95 if size_p95 > 0 else 0,
                'aggressor': trade['Aggressor']
            })
        
        return unusual_sizes
    
    def _detect_unusual_premium(self, df: pd.DataFrame) -> list[dict]:
        """Detect unusual premium spent"""
        unusual_premium = []
        
        # Group by symbol and time window
        df['time_window'] = pd.to_datetime(df['Time_dt']).dt.floor('5min')
        
        premium_by_window = df.groupby(['StandardOptionSymbol', 'time_window']).agg({
            'NotionalValue': 'sum',
            'TradeQuantity': 'sum',
            'Aggressor': lambda x: x.mode()[0] if not x.empty else 'Unknown'
        }).reset_index()
        
        # Find unusual premium
        for _, row in premium_by_window.iterrows():
            if row['NotionalValue'] >= self.detection_thresholds['premium_threshold']:
                unusual_premium.append({
                    'symbol': row['StandardOptionSymbol'],
                    'time_window': row['time_window'],
                    'premium': row['NotionalValue'],
                    'contracts': row['TradeQuantity'],
                    'avg_price': row['NotionalValue'] / row['TradeQuantity'] if row['TradeQuantity'] > 0 else 0,
                    'primary_side': row['Aggressor']
                })
        
        return unusual_premium
    
    def _detect_unusual_iv(self, df: pd.DataFrame) -> list[dict]:
        """Detect unusual implied volatility movements"""
        unusual_iv = []
        
        # Calculate IV changes by symbol
        iv_by_symbol = df.groupby('StandardOptionSymbol').agg({
            'IV': ['first', 'last', 'mean', 'std', 'max', 'min']
        })
        
        iv_by_symbol.columns = ['_'.join(col).strip() for col in iv_by_symbol.columns.values]
        
        for symbol, row in iv_by_symbol.iterrows():
            # Check for IV spikes
            iv_change_pct = (row['IV_last'] - row['IV_first']) / row['IV_first'] * 100 if row['IV_first'] > 0 else 0
            
            if abs(iv_change_pct) >= self.detection_thresholds['iv_spike_pct']:
                unusual_iv.append({
                    'symbol': symbol,
                    'iv_start': row['IV_first'],
                    'iv_end': row['IV_last'],
                    'iv_change_pct': iv_change_pct,
                    'iv_high': row['IV_max'],
                    'iv_low': row['IV_min'],
                    'iv_volatility': row['IV_std']
                })
        
        return unusual_iv
    
    def _detect_unusual_patterns(self, df: pd.DataFrame) -> list[dict]:
        """Detect unusual trading patterns"""
        patterns = []
        
        # Pattern 1: Rapid accumulation
        rapid_accumulation = self._detect_rapid_accumulation(df)
        patterns.extend(rapid_accumulation)
        
        # Pattern 2: Strike clustering
        strike_clustering = self._detect_strike_clustering(df)
        patterns.extend(strike_clustering)
        
        # Pattern 3: Time clustering
        time_clustering = self._detect_time_clustering(df)
        patterns.extend(time_clustering)
        
        return patterns
    
    def _detect_rapid_accumulation(self, df: pd.DataFrame) -> list[dict]:
        """Detect rapid position accumulation"""
        accumulation_patterns = []
        
        # Look for symbols with increasing trade sizes
        for symbol in df['StandardOptionSymbol'].unique():
            symbol_trades = df[df['StandardOptionSymbol'] == symbol].sort_values('Time_dt')
            
            if len(symbol_trades) >= 5:
                # Calculate rolling average of trade sizes
                symbol_trades['size_ma'] = symbol_trades['TradeQuantity'].rolling(3).mean()
                
                # Check if sizes are increasing
                if symbol_trades['size_ma'].is_monotonic_increasing:
                    accumulation_patterns.append({
                        'pattern_type': 'Rapid Accumulation',
                        'symbol': symbol,
                        'trade_count': len(symbol_trades),
                        'start_size': symbol_trades['TradeQuantity'].iloc[0],
                        'end_size': symbol_trades['TradeQuantity'].iloc[-1],
                        'total_volume': symbol_trades['TradeQuantity'].sum()
                    })
        
        return accumulation_patterns
    
    def _detect_strike_clustering(self, df: pd.DataFrame) -> list[dict]:
        """Detect unusual clustering around specific strikes"""
        clustering_patterns = []
        
        # Group by strike price
        if 'Strike_Price_calc' in df.columns:
            strike_volume = df.groupby('Strike_Price_calc').agg({
                'TradeQuantity': 'sum',
                'NotionalValue': 'sum',
                'StandardOptionSymbol': 'nunique'
            }).reset_index()
            
            # Find strikes with unusual concentration
            total_volume = strike_volume['TradeQuantity'].sum()
            for _, row in strike_volume.iterrows():
                concentration = row['TradeQuantity'] / total_volume
                if concentration > 0.2:  # 20% of volume at one strike
                    clustering_patterns.append({
                        'pattern_type': 'Strike Clustering',
                        'strike': row['Strike_Price_calc'],
                        'volume': row['TradeQuantity'],
                        'notional': row['NotionalValue'],
                        'concentration_pct': concentration * 100,
                        'unique_options': row['StandardOptionSymbol']
                    })
        
        return clustering_patterns
    
    def _detect_time_clustering(self, df: pd.DataFrame) -> list[dict]:
        """Detect unusual time-based clustering"""
        time_patterns = []
        
        # Create 1-minute bins
        df['time_bin'] = pd.to_datetime(df['Time_dt']).dt.floor('1min')
        
        # Count trades per minute
        trades_per_minute = df.groupby('time_bin').agg({
            'TradeQuantity': ['count', 'sum'],
            'NotionalValue': 'sum'
        }).reset_index()
        
        trades_per_minute.columns = ['time_bin', 'trade_count', 'volume', 'notional']
        
        # Find unusual bursts
        avg_trades_per_min = trades_per_minute['trade_count'].mean()
        std_trades_per_min = trades_per_minute['trade_count'].std()
        
        for _, row in trades_per_minute.iterrows():
            if row['trade_count'] > avg_trades_per_min + 2 * std_trades_per_min:
                time_patterns.append({
                    'pattern_type': 'Time Clustering',
                    'time': row['time_bin'],
                    'trade_count': row['trade_count'],
                    'volume': row['volume'],
                    'notional': row['notional'],
                    'intensity': row['trade_count'] / avg_trades_per_min
                })
        
        return time_patterns
    
    def _create_uoa_summary(self, detections: dict) -> dict:
        """Create summary of all unusual activity"""
        summary = {
            'total_detections': 0,
            'high_priority_count': 0,
            'detection_types': {},
            'top_symbols': [],
            'alert_level': 'Normal'
        }
        
        # Count detections by type
        for detection_type, items in detections.items():
            if detection_type != 'summary' and isinstance(items, list):
                count = len(items)
                if count > 0:
                    summary['total_detections'] += count
                    summary['detection_types'][detection_type] = count
        
        # Identify high priority
        high_priority_threshold = 3  # Symbols appearing in 3+ detection types
        symbol_counts = {}
        
        for detection_type, items in detections.items():
            if detection_type != 'summary' and isinstance(items, list):
                for item in items:
                    symbol = item.get('symbol')
                    if symbol:
                        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Top symbols
        summary['top_symbols'] = sorted(
            [(sym, count) for sym, count in symbol_counts.items() if count >= 2],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        summary['high_priority_count'] = sum(1 for _, count in symbol_counts.items() if count >= high_priority_threshold)
        
        # Set alert level
        if summary['high_priority_count'] >= 5:
            summary['alert_level'] = 'High'
        elif summary['high_priority_count'] >= 2 or summary['total_detections'] >= 10:
            summary['alert_level'] = 'Elevated'
        elif summary['total_detections'] >= 5:
            summary['alert_level'] = 'Moderate'
        
        return summary