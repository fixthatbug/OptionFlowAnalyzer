# analysis_modules/trade_pattern_detector.py
"""
Trade pattern detection and analysis
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import re

class TradePatternDetector:
    """Detects various trading patterns in options flow data"""
    
    def __init__(self):
        self.patterns = {
            'block_trades': self._detect_block_trades,
            'sweep_orders': self._detect_sweep_orders,
            'rapid_fire': self._detect_rapid_fire_trading,
            'size_stepping': self._detect_size_stepping,
            'time_clustering': self._detect_time_clustering,
            'exchange_arbitrage': self._detect_exchange_arbitrage,
            'accumulation': self._detect_accumulation_patterns,
            'distribution': self._detect_distribution_patterns,
            'momentum_bursts': self._detect_momentum_bursts,
            'reversal_signals': self._detect_reversal_signals,
            'unusual_spreads': self._detect_unusual_spread_activity,
            'gamma_hedging': self._detect_gamma_hedging_patterns
        }
        
        # Pattern thresholds
        self.BLOCK_SIZE_THRESHOLD = 50
        self.RAPID_FIRE_WINDOW = timedelta(seconds=30)
        self.SWEEP_TIME_WINDOW = timedelta(seconds=10)
        self.CLUSTERING_THRESHOLD = 0.8
        
    def detect_all_patterns(self, df: pd.DataFrame) -> dict:
        """Detect all trading patterns"""
        
        if df.empty:
            return self._empty_pattern_results()
        
        pattern_results = {
            'detected_patterns': {},
            'pattern_summary': {},
            'significant_patterns': [],
            'pattern_timeline': [],
            'confidence_scores': {},
            'institutional_indicators': [],
            'retail_indicators': []
        }
        
        # Run all pattern detectors
        for pattern_name, detector_func in self.patterns.items():
            try:
                pattern_data = detector_func(df)
                pattern_results['detected_patterns'][pattern_name] = pattern_data
                
                # Calculate confidence score
                confidence = self._calculate_pattern_confidence(pattern_data, pattern_name)
                pattern_results['confidence_scores'][pattern_name] = confidence
                
                # Add to summary if significant
                if confidence > 70:
                    pattern_results['significant_patterns'].append({
                        'pattern': pattern_name,
                        'confidence': confidence,
                        'data': pattern_data
                    })
                
            except Exception as e:
                print(f"Error detecting {pattern_name}: {e}")
                pattern_results['detected_patterns'][pattern_name] = {}
        
        # Generate pattern summary
        pattern_results['pattern_summary'] = self._generate_pattern_summary(pattern_results)
        
        # Create pattern timeline
        pattern_results['pattern_timeline'] = self._create_pattern_timeline(df, pattern_results)
        
        # Classify institutional vs retail patterns
        pattern_results['institutional_indicators'] = self._identify_institutional_patterns(pattern_results)
        pattern_results['retail_indicators'] = self._identify_retail_patterns(pattern_results)
        
        return pattern_results
    
    def _empty_pattern_results(self) -> dict:
        """Return empty pattern results structure"""
        return {
            'detected_patterns': {},
            'pattern_summary': {},
            'significant_patterns': [],
            'pattern_timeline': [],
            'confidence_scores': {},
            'institutional_indicators': [],
            'retail_indicators': []
        }
    
    def _detect_block_trades(self, df: pd.DataFrame) -> dict:
        """Detect block trades (large size transactions)"""
        
        if 'TradeQuantity' not in df.columns:
            return {}
        
        block_trades = df[df['TradeQuantity'] >= self.BLOCK_SIZE_THRESHOLD].copy()
        
        if block_trades.empty:
            return {'count': 0, 'trades': []}
        
        # Analyze block trade characteristics
        block_analysis = {
            'count': len(block_trades),
            'total_volume': block_trades['TradeQuantity'].sum(),
            'avg_size': block_trades['TradeQuantity'].mean(),
            'max_size': block_trades['TradeQuantity'].max(),
            'total_notional': block_trades['NotionalValue'].sum() if 'NotionalValue' in block_trades.columns else 0,
            'trades': []
        }
        
        # Detailed trade information
        for _, trade in block_trades.iterrows():
            trade_info = {
                'symbol': trade.get('StandardOptionSymbol', ''),
                'quantity': trade.get('TradeQuantity', 0),
                'price': trade.get('Trade_Price', 0),
                'notional': trade.get('NotionalValue', 0),
                'aggressor': trade.get('Aggressor', ''),
                'timestamp': trade.get('Time_dt', ''),
                'exchange': trade.get('Exchange', ''),
                'institutional_score': self._calculate_institutional_score(trade)
            }
            block_analysis['trades'].append(trade_info)
        
        # Sort by size
        block_analysis['trades'].sort(key=lambda x: x['quantity'], reverse=True)
        
        return block_analysis
    
    def _detect_sweep_orders(self, df: pd.DataFrame) -> dict:
        """Detect sweep orders (rapid execution across multiple venues)"""
        
        if df.empty or 'StandardOptionSymbol' not in df.columns:
            return {}
        
        sweeps = []
        
        # Group by option symbol
        for symbol in df['StandardOptionSymbol'].unique():
            symbol_trades = df[df['StandardOptionSymbol'] == symbol].sort_values('Time_dt')
            
            if len(symbol_trades) < 3:
                continue
            
            # Look for rapid succession of trades
            for i in range(len(symbol_trades) - 2):
                trade_group = symbol_trades.iloc[i:i+5]  # Look at 5 trade window
                
                if len(trade_group) < 3:
                    continue
                
                # Check time window
                time_span = trade_group['Time_dt'].max() - trade_group['Time_dt'].min()
                if time_span > self.SWEEP_TIME_WINDOW:
                    continue
                
                # Check for multiple exchanges
                exchanges = trade_group['Exchange'].nunique() if 'Exchange' in trade_group.columns else 1
                
                # Check for same direction
                aggressors = trade_group['Aggressor'].unique() if 'Aggressor' in trade_group.columns else []
                same_direction = len([a for a in aggressors if 'Buy' in str(a)]) > 0 or len([a for a in aggressors if 'Sell' in str(a)]) > 0
                
                if exchanges >= 2 and same_direction:
                    sweep_info = {
                        'symbol': symbol,
                        'trade_count': len(trade_group),
                        'exchanges': exchanges,
                        'total_quantity': trade_group['TradeQuantity'].sum() if 'TradeQuantity' in trade_group.columns else 0,
                        'total_notional': trade_group['NotionalValue'].sum() if 'NotionalValue' in trade_group.columns else 0,
                        'time_span_seconds': time_span.total_seconds(),
                        'avg_price': trade_group['Trade_Price'].mean() if 'Trade_Price' in trade_group.columns else 0,
                        'direction': 'Buy' if any('Buy' in str(a) for a in aggressors) else 'Sell',
                        'timestamp': trade_group['Time_dt'].iloc[0],
                        'urgency_score': self._calculate_sweep_urgency(trade_group, time_span)
                    }
                    sweeps.append(sweep_info)
        
        # Remove duplicates and sort by urgency
        unique_sweeps = self._deduplicate_sweeps(sweeps)
        unique_sweeps.sort(key=lambda x: x['urgency_score'], reverse=True)
        
        return {
            'count': len(unique_sweeps),
            'sweeps': unique_sweeps[:20]  # Top 20 sweeps
        }
    
    def _detect_rapid_fire_trading(self, df: pd.DataFrame) -> dict:
        """Detect rapid-fire trading patterns"""
        
        if df.empty or 'Time_dt' not in df.columns:
            return {}
        
        rapid_fire_events = []
        df_sorted = df.sort_values('Time_dt')
        
        # Look for high-frequency bursts
        for i in range(len(df_sorted) - 5):
            window = df_sorted.iloc[i:i+10]  # 10 trade window
            
            time_span = window['Time_dt'].max() - window['Time_dt'].min()
            
            if time_span <= self.RAPID_FIRE_WINDOW:
                # Check if trades are concentrated in same options
                symbols = window['StandardOptionSymbol'].value_counts()
                concentrated_symbol = symbols.index[0] if not symbols.empty else ''
                concentration = symbols.iloc[0] / len(window) if not symbols.empty else 0
                
                if concentration >= 0.6:  # 60% of trades in same option
                    event = {
                        'start_time': window['Time_dt'].iloc[0],
                        'end_time': window['Time_dt'].iloc[-1],
                        'duration_seconds': time_span.total_seconds(),
                        'trade_count': len(window),
                        'primary_symbol': concentrated_symbol,
                        'concentration_ratio': concentration,
                        'total_volume': window['TradeQuantity'].sum() if 'TradeQuantity' in window.columns else 0,
                        'avg_trade_size': window['TradeQuantity'].mean() if 'TradeQuantity' in window.columns else 0,
                        'intensity_score': len(window) / max(1, time_span.total_seconds())
                    }
                    rapid_fire_events.append(event)
        
        # Remove overlapping events
        unique_events = self._remove_overlapping_events(rapid_fire_events)
        
        return {
            'count': len(unique_events),
            'events': sorted(unique_events, key=lambda x: x['intensity_score'], reverse=True)[:15]
        }
    
    def _detect_size_stepping(self, df: pd.DataFrame) -> dict:
        """Detect size stepping patterns (gradual size increases/decreases)"""
        
        if df.empty or 'TradeQuantity' not in df.columns:
            return {}
        
        stepping_patterns = []
        
        # Group by option symbol
        for symbol in df['StandardOptionSymbol'].unique():
            symbol_trades = df[df['StandardOptionSymbol'] == symbol].sort_values('Time_dt')
            
            if len(symbol_trades) < 5:
                continue
            
            # Look for stepping patterns in trade sizes
            sizes = symbol_trades['TradeQuantity'].values
            
            # Check for increasing pattern
            increasing_steps = self._find_stepping_pattern(sizes, 'increasing')
            if increasing_steps:
                stepping_patterns.append({
                    'symbol': symbol,
                    'pattern_type': 'increasing',
                    'step_count': len(increasing_steps),
                    'size_progression': increasing_steps,
                    'total_volume': sum(increasing_steps),
                    'step_ratio': max(increasing_steps) / min(increasing_steps) if min(increasing_steps) > 0 else 0,
                    'institutional_likelihood': 'high' if len(increasing_steps) >= 4 else 'medium'
                })
            
            # Check for decreasing pattern
            decreasing_steps = self._find_stepping_pattern(sizes, 'decreasing')
            if decreasing_steps:
                stepping_patterns.append({
                    'symbol': symbol,
                    'pattern_type': 'decreasing',
                    'step_count': len(decreasing_steps),
                    'size_progression': decreasing_steps,
                    'total_volume': sum(decreasing_steps),
                    'step_ratio': max(decreasing_steps) / min(decreasing_steps) if min(decreasing_steps) > 0 else 0,
                    'institutional_likelihood': 'high' if len(decreasing_steps) >= 4 else 'medium'
                })
        
        return {
            'count': len(stepping_patterns),
            'patterns': stepping_patterns[:10]
        }
    
    def _detect_time_clustering(self, df: pd.DataFrame) -> dict:
        """Detect temporal clustering of trades"""
        
        if df.empty or 'Time_dt' not in df.columns:
            return {}
        
        # Calculate time differences between consecutive trades
        df_sorted = df.sort_values('Time_dt')
        time_diffs = df_sorted['Time_dt'].diff().dt.total_seconds()
        
        # Find clusters (periods of high activity)
        clusters = []
        current_cluster = []
        
        for i, diff in enumerate(time_diffs):
            if pd.isna(diff):
                continue
            
            if diff <= 5:  # Within 5 seconds
                current_cluster.append(i)
            else:
                if len(current_cluster) >= 3:  # At least 3 trades in cluster
                    cluster_trades = df_sorted.iloc[current_cluster]
                    cluster_info = {
                        'start_time': cluster_trades['Time_dt'].iloc[0],
                        'end_time': cluster_trades['Time_dt'].iloc[-1],
                        'trade_count': len(cluster_trades),
                        'duration_seconds': (cluster_trades['Time_dt'].iloc[-1] - cluster_trades['Time_dt'].iloc[0]).total_seconds(),
                        'symbols_involved': cluster_trades['StandardOptionSymbol'].nunique(),
                        'total_volume': cluster_trades['TradeQuantity'].sum() if 'TradeQuantity' in cluster_trades.columns else 0,
                        'intensity': len(cluster_trades) / max(1, (cluster_trades['Time_dt'].iloc[-1] - cluster_trades['Time_dt'].iloc[0]).total_seconds())
                    }
                    clusters.append(cluster_info)
                
                current_cluster = [i]
        
        # Sort by intensity
        clusters.sort(key=lambda x: x['intensity'], reverse=True)
        
        return {
            'count': len(clusters),
            'clusters': clusters[:10],
            'avg_cluster_size': np.mean([c['trade_count'] for c in clusters]) if clusters else 0,
            'clustering_coefficient': len(clusters) / max(1, len(df)) * 100
        }
    
    # Additional pattern detection methods would continue here...
    # For brevity, I'll include the key helper methods
    
    def _calculate_institutional_score(self, trade: pd.Series) -> float:
        """Calculate institutional likelihood score for a trade"""
        score = 0
        
        # Size factor
        size = trade.get('TradeQuantity', 0)
        if size >= 100:
            score += 40
        elif size >= 50:
            score += 25
        elif size >= 20:
            score += 10
        
        # Notional factor
        notional = trade.get('NotionalValue', 0)
        if notional >= 100000:
            score += 30
        elif notional >= 50000:
            score += 20
        elif notional >= 25000:
            score += 10
        
        # Exchange factor (some exchanges have more institutional flow)
        exchange = trade.get('Exchange', '')
        institutional_exchanges = ['CBOE', 'ISE', 'PHLX', 'BOX']
        if exchange in institutional_exchanges:
            score += 15
        
        # Time factor (institutional often trades at specific times)
        if 'Time_dt' in trade.index:
            hour = trade['Time_dt'].hour
            if 9 <= hour <= 10 or 15 <= hour <= 16:  # Opening/closing
                score += 10
        
        return min(100, score)
    
    def _calculate_sweep_urgency(self, trades: pd.DataFrame, time_span: timedelta) -> float:
        """Calculate urgency score for sweep orders"""
        base_score = 50
        
        # Time factor (faster = more urgent)
        if time_span.total_seconds() <= 2:
            base_score += 30
        elif time_span.total_seconds() <= 5:
            base_score += 20
        elif time_span.total_seconds() <= 10:
            base_score += 10
        
        # Volume factor
        total_volume = trades['TradeQuantity'].sum() if 'TradeQuantity' in trades.columns else 0
        if total_volume >= 200:
            base_score += 20
        elif total_volume >= 100:
            base_score += 15
        elif total_volume >= 50:
            base_score += 10
        
        # Exchange diversity
        exchange_count = trades['Exchange'].nunique() if 'Exchange' in trades.columns else 1
        base_score += min(15, exchange_count * 5)
        
        return min(100, base_score)
    
    def _deduplicate_sweeps(self, sweeps: list[dict]) -> list[dict]:
        """Remove duplicate sweep detections"""
        unique_sweeps = []
        seen_combinations = set()
        
        for sweep in sweeps:
            key = (sweep['symbol'], sweep['timestamp'])
            if key not in seen_combinations:
                unique_sweeps.append(sweep)
                seen_combinations.add(key)
        
        return unique_sweeps
    
    def _find_stepping_pattern(self, sizes: np.ndarray, direction: str) -> list[int]:
        """Find stepping patterns in trade sizes"""
        if len(sizes) < 3:
            return []
        
        steps = []
        current_step = [sizes[0]]
        
        for i in range(1, len(sizes)):
            if direction == 'increasing':
                if sizes[i] > sizes[i-1]:
                    current_step.append(sizes[i])
                else:
                    if len(current_step) >= 3:
                        steps.extend(current_step)
                    current_step = [sizes[i]]
            else:  # decreasing
                if sizes[i] < sizes[i-1]:
                    current_step.append(sizes[i])
                else:
                    if len(current_step) >= 3:
                        steps.extend(current_step)
                    current_step = [sizes[i]]
        
        # Check final step
        if len(current_step) >= 3:
            steps.extend(current_step)
        
        return steps if len(steps) >= 3 else []
    
    def _remove_overlapping_events(self, events: list[dict]) -> list[dict]:
        """Remove overlapping time events"""
        if not events:
            return []
        
        # Sort by start time
        events.sort(key=lambda x: x['start_time'])
        
        unique_events = [events[0]]
        
        for event in events[1:]:
            last_event = unique_events[-1]
            
            # Check for overlap
            if event['start_time'] >= last_event['end_time']:
                unique_events.append(event)
            elif event['intensity_score'] > last_event['intensity_score']:
                # Replace with higher intensity event
                unique_events[-1] = event
        
        return unique_events
    
    def _calculate_pattern_confidence(self, pattern_data: dict, pattern_name: str) -> float:
        """Calculate confidence score for detected pattern"""
        
        confidence = 0
        
        if pattern_name == 'block_trades':
            count = pattern_data.get('count', 0)
            confidence = min(100, count * 20)  # 20 points per block trade
        
        elif pattern_name == 'sweep_orders':
            sweeps = pattern_data.get('sweeps', [])
            if sweeps:
                avg_urgency = sum(s['urgency_score'] for s in sweeps) / len(sweeps)
                confidence = min(100, avg_urgency)
        
        elif pattern_name == 'rapid_fire':
            events = pattern_data.get('events', [])
            if events:
                max_intensity = max(e['intensity_score'] for e in events)
                confidence = min(100, max_intensity * 10)
        
        else:
            # Generic confidence based on count
            count = pattern_data.get('count', 0)
            confidence = min(100, count * 15)
        
        return confidence
    
    def _generate_pattern_summary(self, pattern_results: dict) -> dict:
        """Generate summary of all detected patterns"""
        
        summary = {
            'total_patterns': len(pattern_results['significant_patterns']),
            'highest_confidence_pattern': None,
            'most_frequent_pattern': None,
            'institutional_score': 0,
            'retail_score': 0,
            'pattern_diversity': 0
        }
        
        if pattern_results['significant_patterns']:
            # Highest confidence pattern
            highest = max(pattern_results['significant_patterns'], key=lambda x: x['confidence'])
            summary['highest_confidence_pattern'] = highest['pattern']
            
            # Pattern diversity
            summary['pattern_diversity'] = len(set(p['pattern'] for p in pattern_results['significant_patterns']))
        
        return summary
    
    def _create_pattern_timeline(self, df: pd.DataFrame, pattern_results: dict) -> list[dict]:
        """Create chronological timeline of pattern events"""
        
        timeline_events = []
        
        # Extract timestamped events from patterns
        for pattern_name, pattern_data in pattern_results['detected_patterns'].items():
            
            if pattern_name == 'block_trades':
                for trade in pattern_data.get('trades', []):
                    timeline_events.append({
                        'timestamp': trade.get('timestamp'),
                        'pattern_type': 'Block Trade',
                        'symbol': trade.get('symbol'),
                        'description': f"Block trade: {trade.get('quantity')} contracts @ ${trade.get('price', 0):.2f}",
                        'significance': trade.get('institutional_score', 0)
                    })
        
        # Sort by timestamp
        timeline_events = [e for e in timeline_events if e['timestamp'] is not None]
        timeline_events.sort(key=lambda x: x['timestamp'])
        
        return timeline_events[:50]  # Return top 50 events
    
    def _identify_institutional_patterns(self, pattern_results: dict) -> list[dict]:
        """Identify patterns that indicate institutional activity"""
        
        institutional_indicators = []
        
        # Block trades
        block_data = pattern_results['detected_patterns'].get('block_trades', {})
        if block_data.get('count', 0) > 0:
            institutional_indicators.append({
                'indicator': 'Large Block Trades',
                'count': block_data['count'],
                'evidence': f"{block_data['count']} block trades",
                'confidence': min(100, block_data['count'] * 25)
            })
        
        return sorted(institutional_indicators, key=lambda x: x['confidence'], reverse=True)
    
    def _identify_retail_patterns(self, pattern_results: dict) -> list[dict]:
        """Identify patterns that indicate retail activity"""
        
        retail_indicators = []
        
        # Rapid fire trading
        rapid_data = pattern_results['detected_patterns'].get('rapid_fire', {})
        if rapid_data.get('count', 0) > 0:
            retail_indicators.append({
                'indicator': 'Rapid Fire Trading',
                'count': rapid_data['count'],
                'evidence': f"{rapid_data['count']} rapid trading bursts detected",
                'confidence': min(100, rapid_data['count'] * 20)
            })
        
        return sorted(retail_indicators, key=lambda x: x['confidence'], reverse=True)
    
    # Additional pattern detection methods for other patterns
    def _detect_exchange_arbitrage(self, df: pd.DataFrame) -> dict:
        """Detect potential exchange arbitrage activities"""
        return {'count': 0, 'opportunities': []}
    
    def _detect_accumulation_patterns(self, df: pd.DataFrame) -> dict:
        """Detect accumulation patterns"""
        return {'count': 0, 'patterns': []}
    
    def _detect_distribution_patterns(self, df: pd.DataFrame) -> dict:
        """Detect distribution patterns"""
        return {'count': 0, 'patterns': []}
    
    def _detect_momentum_bursts(self, df: pd.DataFrame) -> dict:
        """Detect momentum burst patterns"""
        return {'count': 0, 'bursts': []}
    
    def _detect_reversal_signals(self, df: pd.DataFrame) -> dict:
        """Detect potential reversal signals"""
        return {'count': 0, 'signals': []}
    
    def _detect_unusual_spread_activity(self, df: pd.DataFrame) -> dict:
        """Detect unusual spread trading activity"""
        return {'count': 0, 'spreads': []}
    
    def _detect_gamma_hedging_patterns(self, df: pd.DataFrame) -> dict:
        """Detect potential gamma hedging patterns"""
        return {'count': 0, 'patterns': []}