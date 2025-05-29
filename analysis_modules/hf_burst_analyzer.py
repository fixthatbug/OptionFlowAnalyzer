# analysis_modules/hf_burst_analyzer.py
"""
High-frequency burst analyzer for detecting rapid trading patterns and institutional activity
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class HfBurstAnalyzer:
    """Analyzes high-frequency trading bursts and rapid execution patterns"""
    
    def __init__(self):
        self.burst_thresholds = {
            'min_trades_per_burst': 5,
            'max_time_window_seconds': 30,
            'min_volume_per_burst': 100,
            'min_notional_per_burst': 50000,
            'volume_spike_multiplier': 3.0,
            'frequency_threshold': 0.5  # trades per second
        }
        
        self.pattern_cache = {}
        
    def analyze_hf_bursts(self, df: pd.DataFrame) -> dict:
        """Main analysis function for high-frequency bursts"""
        
        if df.empty or len(df) < 5:
            return self._empty_burst_results()
        
        # Ensure data is sorted by time
        if 'Time_dt' in df.columns:
            df = df.sort_values('Time_dt').reset_index(drop=True)
        
        burst_analysis = {
            'detected_bursts': self._detect_trading_bursts(df),
            'burst_statistics': {},
            'institutional_patterns': [],
            'momentum_bursts': [],
            'cross_symbol_bursts': [],
            'temporal_patterns': {},
            'volume_spike_events': [],
            'execution_quality': {},
            'burst_clustering': {},
            'market_impact_analysis': {}
        }
        
        # Calculate statistics for detected bursts
        if burst_analysis['detected_bursts']:
            burst_analysis['burst_statistics'] = self._calculate_burst_statistics(
                burst_analysis['detected_bursts']
            )
            
            # Identify institutional patterns
            burst_analysis['institutional_patterns'] = self._identify_institutional_patterns(
                burst_analysis['detected_bursts'], df
            )
            
            # Analyze momentum bursts
            burst_analysis['momentum_bursts'] = self._analyze_momentum_bursts(
                burst_analysis['detected_bursts'], df
            )
            
            # Cross-symbol burst analysis
            burst_analysis['cross_symbol_bursts'] = self._analyze_cross_symbol_bursts(
                burst_analysis['detected_bursts']
            )
            
            # Temporal pattern analysis
            burst_analysis['temporal_patterns'] = self._analyze_temporal_patterns(
                burst_analysis['detected_bursts']
            )
            
            # Volume spike events
            burst_analysis['volume_spike_events'] = self._detect_volume_spikes(df)
            
            # Execution quality analysis
            burst_analysis['execution_quality'] = self._analyze_execution_quality(
                burst_analysis['detected_bursts'], df
            )
            
            # Burst clustering analysis
            burst_analysis['burst_clustering'] = self._analyze_burst_clustering(
                burst_analysis['detected_bursts']
            )
            
            # Market impact analysis
            burst_analysis['market_impact_analysis'] = self._analyze_market_impact(
                burst_analysis['detected_bursts'], df
            )
        
        return burst_analysis
    
    def _empty_burst_results(self) -> dict:
        """Return empty burst analysis structure"""
        return {
            'detected_bursts': [],
            'burst_statistics': {},
            'institutional_patterns': [],
            'momentum_bursts': [],
            'cross_symbol_bursts': [],
            'temporal_patterns': {},
            'volume_spike_events': [],
            'execution_quality': {},
            'burst_clustering': {},
            'market_impact_analysis': {}
        }
    
    def _detect_trading_bursts(self, df: pd.DataFrame) -> list[dict]:
        """Detect high-frequency trading bursts"""
        
        bursts = []
        
        if 'Time_dt' not in df.columns:
            return bursts
        
        # Convert to datetime if not already
        df['Time_dt'] = pd.to_datetime(df['Time_dt'])
        
        # Group by symbol for individual analysis
        for symbol in df['StandardOptionSymbol'].unique():
            symbol_df = df[df['StandardOptionSymbol'] == symbol].copy()
            
            if len(symbol_df) < self.burst_thresholds['min_trades_per_burst']:
                continue
                
            symbol_bursts = self._detect_symbol_bursts(symbol_df, symbol)
            bursts.extend(symbol_bursts)
        
        # Also look for cross-symbol bursts (simultaneous activity across options)
        time_based_bursts = self._detect_time_based_bursts(df)
        bursts.extend(time_based_bursts)
        
        # Sort bursts by intensity score
        bursts.sort(key=lambda x: x.get('intensity_score', 0), reverse=True)
        
        return bursts
    
    def _detect_symbol_bursts(self, symbol_df: pd.DataFrame, symbol: str) -> list[dict]:
        """Detect bursts for a specific symbol"""
        
        bursts = []
        symbol_df = symbol_df.reset_index(drop=True)
        
        i = 0
        while i < len(symbol_df) - 1:
            burst_trades = [i]
            burst_start_time = symbol_df.iloc[i]['Time_dt']
            
            # Look ahead for rapid trades
            j = i + 1
            while j < len(symbol_df):
                current_time = symbol_df.iloc[j]['Time_dt']
                time_diff = (current_time - burst_start_time).total_seconds()
                
                if time_diff <= self.burst_thresholds['max_time_window_seconds']:
                    burst_trades.append(j)
                    j += 1
                else:
                    break
            
            # Check if this qualifies as a burst
            if len(burst_trades) >= self.burst_thresholds['min_trades_per_burst']:
                burst_data = symbol_df.iloc[burst_trades]
                
                # Calculate burst metrics
                total_volume = burst_data['TradeQuantity'].sum() if 'TradeQuantity' in burst_data.columns else 0
                total_notional = burst_data['NotionalValue'].sum() if 'NotionalValue' in burst_data.columns else 0
                duration = (burst_data['Time_dt'].max() - burst_data['Time_dt'].min()).total_seconds()
                
                # Check volume and notional thresholds
                if (total_volume >= self.burst_thresholds['min_volume_per_burst'] and
                    total_notional >= self.burst_thresholds['min_notional_per_burst']):
                    
                    frequency = len(burst_trades) / max(1, duration)
                    
                    burst_info = {
                        'symbol': symbol,
                        'start_time': burst_start_time,
                        'end_time': burst_data['Time_dt'].max(),
                        'duration_seconds': duration,
                        'trade_count': len(burst_trades),
                        'total_volume': total_volume,
                        'total_notional': total_notional,
                        'avg_trade_size': burst_data['TradeQuantity'].mean() if 'TradeQuantity' in burst_data.columns else 0,
                        'frequency_trades_per_sec': frequency,
                        'intensity_score': self._calculate_intensity_score(burst_data, frequency),
                        'primary_aggressor': self._determine_primary_aggressor(burst_data),
                        'price_range': burst_data['Trade_Price'].max() - burst_data['Trade_Price'].min() if 'Trade_Price' in burst_data.columns else 0,
                        'avg_price': burst_data['Trade_Price'].mean() if 'Trade_Price' in burst_data.columns else 0,
                        'burst_type': self._classify_burst_type(burst_data),
                        'institutional_score': self._calculate_institutional_score(burst_data),
                        'trades_data': burst_data.to_dict('records')
                    }
                    
                    bursts.append(burst_info)
            
            # Move to next potential burst start
            i = max(i + 1, j - len(burst_trades) // 2)  # Overlap allowed
        
        return bursts
    
    def _detect_time_based_bursts(self, df: pd.DataFrame) -> list[dict]:
        """Detect bursts based on time clustering across all symbols"""
        
        bursts = []
        
        # Create time windows
        df['time_window'] = df['Time_dt'].dt.floor('5S')  # 5-second windows
        
        # Analyze each time window
        for time_window, window_df in df.groupby('time_window'):
            if len(window_df) < self.burst_thresholds['min_trades_per_burst']:
                continue
            
            # Check if this window has elevated activity
            symbols_involved = window_df['StandardOptionSymbol'].nunique()
            total_volume = window_df['TradeQuantity'].sum() if 'TradeQuantity' in window_df.columns else 0
            total_notional = window_df['NotionalValue'].sum() if 'NotionalValue' in window_df.columns else 0
            
            if (symbols_involved >= 2 and 
                total_volume >= self.burst_thresholds['min_volume_per_burst'] and
                total_notional >= self.burst_thresholds['min_notional_per_burst']):
                
                duration = (window_df['Time_dt'].max() - window_df['Time_dt'].min()).total_seconds()
                frequency = len(window_df) / max(1, duration)
                
                if frequency >= self.burst_thresholds['frequency_threshold']:
                    burst_info = {
                        'symbol': 'MULTI_SYMBOL',
                        'symbols_involved': window_df['StandardOptionSymbol'].unique().tolist(),
                        'start_time': window_df['Time_dt'].min(),
                        'end_time': window_df['Time_dt'].max(),
                        'duration_seconds': duration,
                        'trade_count': len(window_df),
                        'total_volume': total_volume,
                        'total_notional': total_notional,
                        'symbols_count': symbols_involved,
                        'frequency_trades_per_sec': frequency,
                        'intensity_score': self._calculate_intensity_score(window_df, frequency),
                        'burst_type': 'cross_symbol',
                        'institutional_score': self._calculate_institutional_score(window_df),
                        'primary_aggressor': self._determine_primary_aggressor(window_df),
                        'trades_data': window_df.to_dict('records')
                    }
                    
                    bursts.append(burst_info)
        
        return bursts
    
    def _calculate_intensity_score(self, burst_data: pd.DataFrame, frequency: float) -> float:
        """Calculate intensity score for a burst"""
        
        score = 0
        
        # Frequency component (0-40 points)
        score += min(40, frequency * 20)
        
        # Volume component (0-30 points)
        if 'TradeQuantity' in burst_data.columns:
            avg_size = burst_data['TradeQuantity'].mean()
            score += min(30, avg_size / 10)
        
        # Notional component (0-20 points)
        if 'NotionalValue' in burst_data.columns:
            avg_notional = burst_data['NotionalValue'].mean()
            score += min(20, avg_notional / 10000)
        
        # Consistency component (0-10 points)
        if 'TradeQuantity' in burst_data.columns and len(burst_data) > 1:
            cv = burst_data['TradeQuantity'].std() / burst_data['TradeQuantity'].mean()
            score += max(0, 10 - cv * 5)  # Lower CV = higher consistency = higher score
        
        return min(100, score)
    
    def _determine_primary_aggressor(self, burst_data: pd.DataFrame) -> str:
        """Determine primary aggressor in burst"""
        
        if 'Aggressor' not in burst_data.columns:
            return 'Unknown'
        
        aggressor_counts = burst_data['Aggressor'].value_counts()
        
        if aggressor_counts.empty:
            return 'Unknown'
        
        primary = aggressor_counts.index[0]
        
        # Simplify aggressor designation
        if 'Buy' in primary:
            return 'Buy'
        elif 'Sell' in primary:
            return 'Sell'
        else:
            return primary
    
    def _classify_burst_type(self, burst_data: pd.DataFrame) -> str:
        """Classify the type of burst"""
        
        # Check for increasing/decreasing size pattern
        if 'TradeQuantity' in burst_data.columns and len(burst_data) >= 3:
            sizes = burst_data['TradeQuantity'].values
            
            # Check for stepping pattern
            if self._is_increasing_pattern(sizes):
                return 'accumulation_stepping'
            elif self._is_decreasing_pattern(sizes):
                return 'distribution_stepping'
        
        # Check for consistent size (algo trading)
        if 'TradeQuantity' in burst_data.columns:
            size_std = burst_data['TradeQuantity'].std()
            size_mean = burst_data['TradeQuantity'].mean()
            
            if size_mean > 0 and size_std / size_mean < 0.2:  # Low coefficient of variation
                return 'algorithmic_consistent'
        
        # Check for price movement pattern
        if 'Trade_Price' in burst_data.columns and len(burst_data) >= 3:
            price_trend = burst_data['Trade_Price'].diff().mean()
            
            if abs(price_trend) > burst_data['Trade_Price'].std() * 0.1:
                return 'momentum_driven' if price_trend > 0 else 'reversal_driven'
        
        return 'general_burst'
    
    def _calculate_institutional_score(self, burst_data: pd.DataFrame) -> float:
        """Calculate likelihood of institutional involvement"""
        
        score = 0
        
        # Large average trade size
        if 'TradeQuantity' in burst_data.columns:
            avg_size = burst_data['TradeQuantity'].mean()
            if avg_size >= 100:
                score += 30
            elif avg_size >= 50:
                score += 20
            elif avg_size >= 25:
                score += 10
        
        # High notional value
        if 'NotionalValue' in burst_data.columns:
            total_notional = burst_data['NotionalValue'].sum()
            if total_notional >= 500000:  # $500K+
                score += 25
            elif total_notional >= 100000:  # $100K+
                score += 15
            elif total_notional >= 50000:   # $50K+
                score += 10
        
        # Execution quality (trading at/near mid)
        if all(col in burst_data.columns for col in ['Trade_Price', 'Option_Bid', 'Option_Ask']):
            burst_data_clean = burst_data.dropna(subset=['Trade_Price', 'Option_Bid', 'Option_Ask'])
            if not burst_data_clean.empty:
                mid_prices = (burst_data_clean['Option_Bid'] + burst_data_clean['Option_Ask']) / 2
                price_deviations = abs(burst_data_clean['Trade_Price'] - mid_prices) / mid_prices
                avg_deviation = price_deviations.mean()
                
                if avg_deviation < 0.02:  # Within 2% of mid
                    score += 20
                elif avg_deviation < 0.05:  # Within 5% of mid
                    score += 10
        
        # Time consistency (professional execution timing)
        if len(burst_data) > 2:
            time_intervals = burst_data['Time_dt'].diff().dt.total_seconds().dropna()
            if not time_intervals.empty:
                interval_consistency = 1 / (1 + time_intervals.std())
                score += interval_consistency * 15
        
        # Exchange diversity
        if 'Exchange' in burst_data.columns:
            unique_exchanges = burst_data['Exchange'].nunique()
            if unique_exchanges >= 3:
                score += 10
            elif unique_exchanges >= 2:
                score += 5
        
        return min(100, score)
    
    def _is_increasing_pattern(self, sizes: np.ndarray) -> bool:
        """Check if sizes follow an increasing pattern"""
        if len(sizes) < 3:
            return False
        
        increasing_count = 0
        for i in range(1, len(sizes)):
            if sizes[i] > sizes[i-1]:
                increasing_count += 1
        
        return increasing_count >= len(sizes) * 0.6  # 60% increasing
    
    def _is_decreasing_pattern(self, sizes: np.ndarray) -> bool:
        """Check if sizes follow a decreasing pattern"""
        if len(sizes) < 3:
            return False
        
        decreasing_count = 0
        for i in range(1, len(sizes)):
            if sizes[i] < sizes[i-1]:
                decreasing_count += 1
        
        return decreasing_count >= len(sizes) * 0.6  # 60% decreasing
    
    def _calculate_burst_statistics(self, bursts: list[dict]) -> dict:
        """Calculate overall statistics for all detected bursts"""
        
        if not bursts:
            return {}
        
        stats = {
            'total_bursts': len(bursts),
            'avg_burst_duration': np.mean([b['duration_seconds'] for b in bursts]),
            'avg_trades_per_burst': np.mean([b['trade_count'] for b in bursts]),
            'avg_volume_per_burst': np.mean([b['total_volume'] for b in bursts]),
            'avg_notional_per_burst': np.mean([b['total_notional'] for b in bursts]),
            'avg_intensity_score': np.mean([b['intensity_score'] for b in bursts]),
            'avg_institutional_score': np.mean([b['institutional_score'] for b in bursts])
        }
        
        # Burst type distribution
        burst_types = [b.get('burst_type', 'unknown') for b in bursts]
        type_counts = pd.Series(burst_types).value_counts()
        stats['burst_type_distribution'] = type_counts.to_dict()
        
        # Aggressor distribution
        aggressors = [b.get('primary_aggressor', 'unknown') for b in bursts]
        aggressor_counts = pd.Series(aggressors).value_counts()
        stats['aggressor_distribution'] = aggressor_counts.to_dict()
        
        # Time distribution
        burst_hours = [b['start_time'].hour for b in bursts if 'start_time' in b]
        if burst_hours:
            hour_counts = pd.Series(burst_hours).value_counts()
            stats['hourly_distribution'] = hour_counts.to_dict()
        
        return stats
    
    def _identify_institutional_patterns(self, bursts: list[dict], df: pd.DataFrame) -> list[dict]:
        """Identify patterns indicating institutional activity"""
        
        institutional_patterns = []
        
        # High institutional score bursts
        high_institutional_bursts = [b for b in bursts if b.get('institutional_score', 0) >= 70]
        
        for burst in high_institutional_bursts:
            pattern = {
                'pattern_type': 'high_institutional_score',
                'burst_info': burst,
                'confidence': burst.get('institutional_score', 0),
                'indicators': self._get_institutional_indicators(burst)
            }
            institutional_patterns.append(pattern)
        
        # Look for coordinated bursts (multiple symbols, similar timing)
        coordinated_bursts = self._find_coordinated_bursts(bursts)
        for coord_group in coordinated_bursts:
            pattern = {
                'pattern_type': 'coordinated_activity',
                'bursts': coord_group,
                'confidence': 80,
                'indicators': ['multiple_symbols', 'synchronized_timing', 'similar_execution']
            }
            institutional_patterns.append(pattern)
        
        return institutional_patterns
    
    def _get_institutional_indicators(self, burst: dict) -> list[str]:
        """Get indicators suggesting institutional activity"""
        
        indicators = []
        
        if burst.get('total_volume', 0) >= 500:
            indicators.append('large_volume')
        
        if burst.get('total_notional', 0) >= 200000:
            indicators.append('high_notional')
        
        if burst.get('avg_trade_size', 0) >= 100:
            indicators.append('large_average_size')
        
        if burst.get('frequency_trades_per_sec', 0) >= 2:
            indicators.append('high_frequency')
        
        if 'algorithmic' in burst.get('burst_type', ''):
            indicators.append('algorithmic_execution')
        
        return indicators
    
    def _find_coordinated_bursts(self, bursts: list[dict]) -> list[list[dict]]:
        """Find coordinated bursts across symbols"""
        
        coordinated_groups = []
        used_bursts = set()
        
        for i, burst1 in enumerate(bursts):
            if i in used_bursts:
                continue
                
            group = [burst1]
            used_bursts.add(i)
            
            for j, burst2 in enumerate(bursts[i+1:], i+1):
                if j in used_bursts:
                    continue
                
                # Check for coordination
                time_diff = abs((burst1['start_time'] - burst2['start_time']).total_seconds())
                
                if (time_diff <= 60 and  # Within 1 minute
                    burst1['symbol'] != burst2['symbol'] and  # Different symbols
                    abs(burst1.get('institutional_score', 0) - burst2.get('institutional_score', 0)) <= 20):  # Similar institutional scores
                    
                    group.append(burst2)
                    used_bursts.add(j)
            
            if len(group) >= 2:
                coordinated_groups.append(group)
        
        return coordinated_groups
    
    def _analyze_momentum_bursts(self, bursts: list[dict], df: pd.DataFrame) -> list[dict]:
        """Analyze momentum-driven bursts"""
        
        momentum_bursts = []
        
        for burst in bursts:
            if burst.get('burst_type') in ['momentum_driven', 'reversal_driven']:
                # Analyze the momentum characteristics
                burst_trades = pd.DataFrame(burst.get('trades_data', []))
                
                if not burst_trades.empty and 'Trade_Price' in burst_trades.columns:
                    price_momentum = self._calculate_price_momentum(burst_trades)
                    volume_momentum = self._calculate_volume_momentum(burst_trades)
                    
                    momentum_analysis = {
                        'burst_info': burst,
                        'price_momentum': price_momentum,
                        'volume_momentum': volume_momentum,
                        'momentum_strength': (price_momentum + volume_momentum) / 2,
                        'direction': 'bullish' if price_momentum > 0 else 'bearish'
                    }
                    
                    momentum_bursts.append(momentum_analysis)
        
        return momentum_bursts
    
    def _calculate_price_momentum(self, trades_df: pd.DataFrame) -> float:
        """Calculate price momentum for trades"""
        
        if 'Trade_Price' not in trades_df.columns or len(trades_df) < 2:
            return 0
        
        price_changes = trades_df['Trade_Price'].pct_change().dropna()
        
        if price_changes.empty:
            return 0
        
        return price_changes.mean() * 100  # Convert to percentage
    
    def _calculate_volume_momentum(self, trades_df: pd.DataFrame) -> float:
        """Calculate volume momentum for trades"""
        
        if 'TradeQuantity' not in trades_df.columns or len(trades_df) < 2:
            return 0
        
        # Simple volume trend analysis
        volumes = trades_df['TradeQuantity'].values
        
        if len(volumes) < 3:
            return 0
        
        # Calculate if volume is generally increasing or decreasing
        increasing_count = sum(1 for i in range(1, len(volumes)) if volumes[i] > volumes[i-1])
        total_comparisons = len(volumes) - 1
        
        momentum_ratio = increasing_count / total_comparisons
        
        # Convert to momentum score (-50 to +50)
        return (momentum_ratio - 0.5) * 100
    
    def _analyze_cross_symbol_bursts(self, bursts: list[dict]) -> list[dict]:
        """Analyze bursts that involve multiple symbols"""
        
        cross_symbol_bursts = [b for b in bursts if b.get('burst_type') == 'cross_symbol']
        
        analysis = []
        
        for burst in cross_symbol_bursts:
            symbols = burst.get('symbols_involved', [])
            
            if len(symbols) >= 2:
                analysis_info = {
                    'burst_info': burst,
                    'symbol_count': len(symbols),
                    'symbols': symbols,
                    'coordination_score': self._calculate_coordination_score(burst),
                    'market_impact_potential': self._estimate_market_impact_potential(burst)
                }
                
                analysis.append(analysis_info)
        
        return analysis
    
    def _calculate_coordination_score(self, burst: dict) -> float:
        """Calculate how coordinated a cross-symbol burst appears"""
        
        score = 50  # Base score
        
        # More symbols = higher coordination
        symbol_count = len(burst.get('symbols_involved', []))
        score += min(30, symbol_count * 5)
        
        # Higher institutional score = more coordination
        institutional_score = burst.get('institutional_score', 0)
        score += institutional_score * 0.2
        
        # Shorter duration with more trades = more coordination
        duration = burst.get('duration_seconds', 1)
        trade_count = burst.get('trade_count', 1)
        
        if duration > 0:
            intensity = trade_count / duration
            score += min(20, intensity * 2)
        
        return min(100, score)
    
    def _estimate_market_impact_potential(self, burst: dict) -> str:
        """Estimate potential market impact of burst"""
        
        total_notional = burst.get('total_notional', 0)
        symbol_count = len(burst.get('symbols_involved', []))
        
        if total_notional >= 1000000 and symbol_count >= 5:
            return 'High'
        elif total_notional >= 500000 and symbol_count >= 3:
            return 'Medium'
        elif total_notional >= 100000:
            return 'Low'
        else:
            return 'Minimal'
    
    def _analyze_temporal_patterns(self, bursts: list[dict]) -> dict:
        """Analyze temporal patterns in bursts"""
        
        if not bursts:
            return {}
        
        # Extract burst times
        burst_times = [b['start_time'] for b in bursts if 'start_time' in b]
        
        if not burst_times:
            return {}
        
        patterns = {}
        
        # Hourly distribution
        hours = [t.hour for t in burst_times]
        hour_counts = pd.Series(hours).value_counts().sort_index()
        patterns['hourly_distribution'] = hour_counts.to_dict()
        
        # Peak activity hours
        if not hour_counts.empty:
            patterns['peak_hour'] = hour_counts.idxmax()
            patterns['peak_hour_count'] = hour_counts.max()
        
        # Day of week distribution
        weekdays = [t.weekday() for t in burst_times]
        weekday_counts = pd.Series(weekdays).value_counts().sort_index()
        patterns['weekday_distribution'] = weekday_counts.to_dict()
        
        # Time clustering analysis
        patterns['time_clustering'] = self._analyze_time_clustering(burst_times)
        
        return patterns
    
    def _analyze_time_clustering(self, burst_times: list[datetime]) -> dict:
        """Analyze clustering of burst times"""
        
        if len(burst_times) < 2:
            return {}
        
        # Calculate time differences between consecutive bursts
        sorted_times = sorted(burst_times)
        time_diffs = [(sorted_times[i+1] - sorted_times[i]).total_seconds() 
                     for i in range(len(sorted_times)-1)]
        
        clustering = {
            'avg_time_between_bursts': np.mean(time_diffs),
            'median_time_between_bursts': np.median(time_diffs),
            'min_time_between_bursts': min(time_diffs),
            'max_time_between_bursts': max(time_diffs)
        }
        
        # Identify clusters (bursts within 5 minutes of each other)
        cluster_threshold = 300  # 5 minutes
        clusters = []
        current_cluster = [sorted_times[0]]
        
        for i in range(1, len(sorted_times)):
            time_diff = (sorted_times[i] - sorted_times[i-1]).total_seconds()
            
            if time_diff <= cluster_threshold:
                current_cluster.append(sorted_times[i])
            else:
                if len(current_cluster) >= 2:
                    clusters.append(current_cluster)
                current_cluster = [sorted_times[i]]
        
        # Add final cluster if it has multiple bursts
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        clustering['cluster_count'] = len(clusters)
        clustering['avg_cluster_size'] = np.mean([len(c) for c in clusters]) if clusters else 0
        
        return clustering
    
    def _detect_volume_spikes(self, df: pd.DataFrame) -> list[dict]:
        """Detect significant volume spikes"""
        
        if 'TradeQuantity' not in df.columns or df.empty:
            return []
        
        volume_spikes = []
        
        # Calculate rolling average and identify spikes
        if len(df) >= 10:
            df_sorted = df.sort_values('Time_dt') if 'Time_dt' in df.columns else df
            
            # Calculate rolling average volume
            window_size = min(20, len(df_sorted) // 2)
            df_sorted['volume_ma'] = df_sorted['TradeQuantity'].rolling(window_size).mean()
            
            # Identify spikes
            spike_multiplier = self.burst_thresholds['volume_spike_multiplier']
            
            for idx, row in df_sorted.iterrows():
                if pd.notna(row['volume_ma']) and row['volume_ma'] > 0:
                    spike_ratio = row['TradeQuantity'] / row['volume_ma']
                    
                    if spike_ratio >= spike_multiplier:
                        spike_info = {
                            'timestamp': row.get('Time_dt', ''),
                            'symbol': row.get('StandardOptionSymbol', ''),
                            'volume': row.get('TradeQuantity', 0),
                            'volume_average': row['volume_ma'],
                            'spike_ratio': spike_ratio,
                            'price': row.get('Trade_Price', 0),
                            'notional': row.get('NotionalValue', 0),
                            'aggressor': row.get('Aggressor', ''),
                            'spike_severity': 'extreme' if spike_ratio >= 5 else 'high' if spike_ratio >= 4 else 'moderate'
                        }
                        volume_spikes.append(spike_info)
        
        # Sort by spike ratio
        volume_spikes.sort(key=lambda x: x['spike_ratio'], reverse=True)
        
        return volume_spikes[:20]  # Return top 20 spikes
    
    def _analyze_execution_quality(self, bursts: list[dict], df: pd.DataFrame) -> dict:
        """Analyze execution quality of bursts"""
        
        execution_analysis = {
            'avg_execution_score': 0,
            'execution_distribution': {},
            'price_improvement_analysis': {},
            'timing_analysis': {}
        }
        
        if not bursts:
            return execution_analysis
        
        execution_scores = []
        
        for burst in bursts:
            burst_trades = pd.DataFrame(burst.get('trades_data', []))
            
            if not burst_trades.empty:
                score = self._calculate_execution_score(burst_trades)
                execution_scores.append(score)
        
        if execution_scores:
            execution_analysis['avg_execution_score'] = np.mean(execution_scores)
            execution_analysis['execution_distribution'] = {
                'excellent': sum(1 for s in execution_scores if s >= 80),
                'good': sum(1 for s in execution_scores if 60 <= s < 80),
                'fair': sum(1 for s in execution_scores if 40 <= s < 60),
                'poor': sum(1 for s in execution_scores if s < 40)
            }
        
        return execution_analysis
    
    def _calculate_execution_score(self, trades_df: pd.DataFrame) -> float:
        """Calculate execution quality score for trades"""
        
        score = 50  # Base score
        
        # Price execution quality
        if all(col in trades_df.columns for col in ['Trade_Price', 'Option_Bid', 'Option_Ask']):
            clean_trades = trades_df.dropna(subset=['Trade_Price', 'Option_Bid', 'Option_Ask'])
            
            if not clean_trades.empty:
                mid_prices = (clean_trades['Option_Bid'] + clean_trades['Option_Ask']) / 2
                spreads = clean_trades['Option_Ask'] - clean_trades['Option_Bid']
                
                # Calculate price improvement
                improvements = []
                for _, trade in clean_trades.iterrows():
                    if 'Buy' in str(trade.get('Aggressor', '')):
                        # For buys, improvement is paying less than ask
                        improvement = (trade['Option_Ask'] - trade['Trade_Price']) / spreads.loc[trade.name]
                    else:
                        # For sells, improvement is receiving more than bid
                        improvement = (trade['Trade_Price'] - trade['Option_Bid']) / spreads.loc[trade.name]
                    
                    improvements.append(max(0, improvement))
                
                if improvements:
                    avg_improvement = np.mean(improvements)
                    score += min(30, avg_improvement * 100)
        
        # Size consistency (professional execution)
        if 'TradeQuantity' in trades_df.columns and len(trades_df) > 1:
            size_cv = trades_df['TradeQuantity'].std() / trades_df['TradeQuantity'].mean()
            consistency_score = max(0, 20 - size_cv * 10)
            score += consistency_score
        
        return min(100, score)
    
    def _analyze_burst_clustering(self, bursts: list[dict]) -> dict:
        """Analyze clustering patterns in bursts"""
        
        clustering = {
            'temporal_clusters': [],
            'symbol_clusters': {},
            'size_clusters': {}
        }
        
        if not bursts:
            return clustering
        
        # Temporal clustering
        burst_times = [(b['start_time'], b) for b in bursts if 'start_time' in b]
        burst_times.sort(key=lambda x: x[0])
        
        # Group bursts that occur within 10 minutes of each other
        cluster_window = timedelta(minutes=10)
        current_cluster = []
        
        for timestamp, burst in burst_times:
            if not current_cluster:
                current_cluster.append(burst)
            else:
                last_time = current_cluster[-1]['start_time']
                if timestamp - last_time <= cluster_window:
                    current_cluster.append(burst)
                else:
                    if len(current_cluster) >= 2:
                        clustering['temporal_clusters'].append({
                            'burst_count': len(current_cluster),
                            'start_time': current_cluster[0]['start_time'],
                            'end_time': current_cluster[-1]['start_time'],
                            'total_volume': sum(b.get('total_volume', 0) for b in current_cluster),
                            'total_notional': sum(b.get('total_notional', 0) for b in current_cluster)
                        })
                    current_cluster = [burst]
        
        # Add final cluster
        if len(current_cluster) >= 2:
            clustering['temporal_clusters'].append({
                'burst_count': len(current_cluster),
                'start_time': current_cluster[0]['start_time'],
                'end_time': current_cluster[-1]['start_time'],
                'total_volume': sum(b.get('total_volume', 0) for b in current_cluster),
                'total_notional': sum(b.get('total_notional', 0) for b in current_cluster)
            })
        
        # Symbol clustering
        symbol_burst_counts = defaultdict(int)
        for burst in bursts:
            symbol = burst.get('symbol', 'unknown')
            symbol_burst_counts[symbol] += 1
        
        clustering['symbol_clusters'] = dict(symbol_burst_counts)
        
        return clustering
    
    def _analyze_market_impact(self, bursts: list[dict], df: pd.DataFrame) -> dict:
        """Analyze potential market impact of bursts"""
        
        impact_analysis = {
            'high_impact_bursts': [],
            'impact_distribution': {},
            'cumulative_impact_score': 0
        }
        
        if not bursts:
            return impact_analysis
        
        impact_scores = []
        
        for burst in bursts:
            impact_score = self._calculate_market_impact_score(burst)
            impact_scores.append(impact_score)
            
            if impact_score >= 70:
                impact_analysis['high_impact_bursts'].append({
                    'burst': burst,
                    'impact_score': impact_score,
                    'impact_factors': self._identify_impact_factors(burst)
                })
        
        if impact_scores:
            impact_analysis['cumulative_impact_score'] = sum(impact_scores)
            impact_analysis['avg_impact_score'] = np.mean(impact_scores)
            
            # Impact distribution
            impact_analysis['impact_distribution'] = {
                'high': sum(1 for s in impact_scores if s >= 70),
                'medium': sum(1 for s in impact_scores if 40 <= s < 70),
                'low': sum(1 for s in impact_scores if s < 40)
            }
        
        return impact_analysis
    
    def _calculate_market_impact_score(self, burst: dict) -> float:
        """Calculate potential market impact score for a burst"""
        
        score = 0
        
        # Volume impact
        total_volume = burst.get('total_volume', 0)
        if total_volume >= 1000:
            score += 30
        elif total_volume >= 500:
            score += 20
        elif total_volume >= 100:
            score += 10
        
        # Notional impact
        total_notional = burst.get('total_notional', 0)
        if total_notional >= 1000000:  # $1M+
            score += 25
        elif total_notional >= 500000:  # $500K+
            score += 15
        elif total_notional >= 100000:  # $100K+
            score += 10
        
        # Speed impact (faster = more impact)
        frequency = burst.get('frequency_trades_per_sec', 0)
        if frequency >= 3:
            score += 20
        elif frequency >= 1:
            score += 15
        elif frequency >= 0.5:
            score += 10
        
        # Institutional score impact
        institutional_score = burst.get('institutional_score', 0)
        score += institutional_score * 0.25
        
        return min(100, score)
    
    def _identify_impact_factors(self, burst: dict) -> list[str]:
        """Identify factors contributing to market impact"""
        
        factors = []
        
        if burst.get('total_volume', 0) >= 500:
            factors.append('high_volume')
        
        if burst.get('total_notional', 0) >= 500000:
            factors.append('high_notional')
        
        if burst.get('frequency_trades_per_sec', 0) >= 2:
            factors.append('high_frequency')
        
        if burst.get('institutional_score', 0) >= 80:
            factors.append('institutional_activity')
        
        if burst.get('burst_type') == 'cross_symbol':
            factors.append('cross_symbol_coordination')
        
        if 'momentum' in burst.get('burst_type', ''):
            factors.append('momentum_driven')
        
        return factors
    
    def generate_burst_summary_report(self, burst_analysis: dict) -> str:
        """Generate a summary report of burst analysis"""
        
        report = "HIGH-FREQUENCY BURST ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Summary statistics
        bursts = burst_analysis.get('detected_bursts', [])
        stats = burst_analysis.get('burst_statistics', {})
        
        report += f"Total Bursts Detected: {len(bursts)}\n"
        
        if stats:
            report += f"Average Burst Duration: {stats.get('avg_burst_duration', 0):.1f} seconds\n"
            report += f"Average Trades per Burst: {stats.get('avg_trades_per_burst', 0):.1f}\n"
            report += f"Average Volume per Burst: {stats.get('avg_volume_per_burst', 0):.0f} contracts\n"
            report += f"Average Intensity Score: {stats.get('avg_intensity_score', 0):.1f}/100\n"
            report += f"Average Institutional Score: {stats.get('avg_institutional_score', 0):.1f}/100\n\n"
        
        # Top bursts
        if bursts:
            report += "TOP HIGH-INTENSITY BURSTS:\n"
            report += "-" * 30 + "\n"
            
            top_bursts = sorted(bursts, key=lambda x: x.get('intensity_score', 0), reverse=True)[:5]
            
            for i, burst in enumerate(top_bursts, 1):
                report += f"{i}. {burst.get('symbol', 'Unknown')}\n"
                report += f"   Time: {burst.get('start_time', 'N/A')}\n"
                report += f"   Volume: {burst.get('total_volume', 0):,} contracts\n"
                report += f"   Notional: ${burst.get('total_notional', 0):,.0f}\n"
                report += f"   Intensity: {burst.get('intensity_score', 0):.1f}/100\n"
                report += f"   Type: {burst.get('burst_type', 'unknown')}\n\n"
        
        # Institutional patterns
        institutional = burst_analysis.get('institutional_patterns', [])
        if institutional:
            report += f"INSTITUTIONAL ACTIVITY DETECTED:\n"
            report += f"High-confidence institutional patterns: {len(institutional)}\n\n"
        
        # Market impact
        impact = burst_analysis.get('market_impact_analysis', {})
        high_impact = impact.get('high_impact_bursts', [])
        if high_impact:
            report += f"HIGH MARKET IMPACT BURSTS:\n"
            report += f"Count: {len(high_impact)}\n"
            report += f"Cumulative Impact Score: {impact.get('cumulative_impact_score', 0):.0f}\n\n"
        
        report += "=" * 50 + "\n"
        
        return report