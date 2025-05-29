# market_microstructure_analyzer.py
"""
Advanced market microstructure analysis for alpha extraction
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class MicrostructureMetrics:
    """Container for market microstructure metrics"""
    effective_spread: float
    realized_spread: float
    price_impact: float
    order_flow_imbalance: float
    trade_intensity: float
    quote_stuffing_index: float
    adverse_selection_cost: float
    market_maker_participation: float

class MarketMicrostructureAnalyzer:
    """Analyzes market microstructure for alpha signals"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.trade_history = []
        self.quote_history = []
        self.metrics_cache = {}
        
    def analyze_microstructure(self, trades_df: pd.DataFrame) -> dict:
        """Comprehensive microstructure analysis"""
        if trades_df.empty:
            return {}
        
        analysis = {
            'liquidity_metrics': self._analyze_liquidity(trades_df),
            'toxicity_metrics': self._analyze_toxicity(trades_df),
            'market_maker_behavior': self._analyze_market_maker_behavior(trades_df),
            'order_flow_dynamics': self._analyze_order_flow_dynamics(trades_df),
            'price_discovery': self._analyze_price_discovery(trades_df)
        }
        
        return analysis
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> dict:
        """Analyze liquidity conditions"""
        metrics = {}
        
        # Effective spread
        df['spread'] = df['Option_Ask'] - df['Option_Bid']
        df['mid_price'] = (df['Option_Ask'] + df['Option_Bid']) / 2
        df['effective_spread'] = 2 * abs(df['Trade_Price'] - df['mid_price'])
        
        metrics['avg_spread'] = df['spread'].mean()
        metrics['avg_effective_spread'] = df['effective_spread'].mean()
        metrics['spread_volatility'] = df['spread'].std()
        
        # Depth analysis
        metrics['avg_bid_size'] = df['Option_Bid'].count()  # Placeholder
        metrics['avg_ask_size'] = df['Option_Ask'].count()  # Placeholder
        
        # Resiliency (how quickly spreads recover after trades)
        metrics['spread_resiliency'] = self._calculate_spread_resiliency(df)
        
        return metrics
    
    def _analyze_toxicity(self, df: pd.DataFrame) -> dict:
        """Analyze flow toxicity using VPIN and other metrics"""
        metrics = {}
        
        # Volume-synchronized Probability of Informed Trading (VPIN)
        metrics['vpin'] = self._calculate_vpin(df)
        
        # Order flow imbalance
        buy_volume = df[df['Aggressor'].str.contains('Buy', na=False)]['TradeQuantity'].sum()
        sell_volume = df[df['Aggressor'].str.contains('Sell', na=False)]['TradeQuantity'].sum()
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            metrics['order_imbalance'] = (buy_volume - sell_volume) / total_volume
            metrics['order_imbalance_ratio'] = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        else:
            metrics['order_imbalance'] = 0
            metrics['order_imbalance_ratio'] = 1
        
        # Adverse selection
        metrics['adverse_selection'] = self._calculate_adverse_selection(df)
        
        # Trade size distribution (informed traders tend to trade in specific sizes)
        trade_sizes = df['TradeQuantity']
        metrics['trade_size_entropy'] = stats.entropy(trade_sizes.value_counts())
        metrics['large_trade_ratio'] = (trade_sizes >= 100).sum() / len(trade_sizes) if len(trade_sizes) > 0 else 0
        
        return metrics
    
    def _analyze_market_maker_behavior(self, df: pd.DataFrame) -> dict:
        """Analyze market maker positioning and behavior"""
        metrics = {}
        
        # Identify potential market maker trades (often at mid or improving price)
        df['at_mid'] = abs(df['Trade_Price'] - df['mid_price']) < 0.01
        metrics['mm_participation_rate'] = df['at_mid'].sum() / len(df) if len(df) > 0 else 0
        
        # Quote stability (market makers withdrawing = instability)
        if 'spread' in df.columns:
            metrics['quote_stability'] = 1 / (1 + df['spread'].std())
        
        # Inventory risk (inferred from one-sided flow)
        buy_flow = df[df['Aggressor'].str.contains('Buy', na=False)]['NotionalValue'].sum()
        sell_flow = df[df['Aggressor'].str.contains('Sell', na=False)]['NotionalValue'].sum()
        metrics['mm_inventory_pressure'] = abs(buy_flow - sell_flow) / (buy_flow + sell_flow) if (buy_flow + sell_flow) > 0 else 0
        
        # Time between trades (market makers provide liquidity continuously)
        if len(df) > 1:
            time_diffs = df['Time_dt'].diff().dt.total_seconds()
            metrics['avg_time_between_trades'] = time_diffs.mean()
            metrics['trade_clustering'] = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 0
        
        return metrics
    
    def _analyze_order_flow_dynamics(self, df: pd.DataFrame) -> dict:
        """Analyze order flow patterns and dynamics"""
        metrics = {}
        
        # Momentum indicators
        df['signed_volume'] = df.apply(
            lambda x: x['TradeQuantity'] if 'Buy' in x['Aggressor'] else -x['TradeQuantity'],
            axis=1
        )
        
        # Cumulative order flow
        df['cumulative_flow'] = df['signed_volume'].cumsum()
        metrics['flow_momentum'] = df['cumulative_flow'].iloc[-1] if len(df) > 0 else 0
        
        # Acceleration
        if len(df) > 10:
            recent_flow = df['signed_volume'].tail(10).sum()
            earlier_flow = df['signed_volume'].iloc[-20:-10].sum() if len(df) > 20 else 0
            metrics['flow_acceleration'] = recent_flow - earlier_flow
        else:
            metrics['flow_acceleration'] = 0
        
        # Persistence (autocorrelation of order flow)
        if len(df) > 2:
            metrics['flow_persistence'] = df['signed_volume'].autocorr(lag=1)
        else:
            metrics['flow_persistence'] = 0
        
        # Hidden liquidity detection
        metrics['hidden_liquidity_indicator'] = self._detect_hidden_liquidity(df)
        
        return metrics
    
    def _analyze_price_discovery(self, df: pd.DataFrame) -> dict:
        """Analyze price discovery and information content"""
        metrics = {}
        
        if len(df) < 2:
            return metrics
        
        # Weighted Price Contribution (WPC)
        df['price_change'] = df['Trade_Price'].diff()
        df['signed_price_change'] = df['price_change'] * df['signed_volume'].apply(np.sign)
        
        metrics['price_discovery_measure'] = df['signed_price_change'].sum()
        
        # Information share (simplified)
        price_variance = df['Trade_Price'].var()
        if price_variance > 0:
            metrics['information_ratio'] = df['price_change'].var() / price_variance
        else:
            metrics['information_ratio'] = 0
        
        # Permanent vs temporary price impact
        metrics['permanent_impact'] = self._calculate_permanent_impact(df)
        metrics['temporary_impact'] = self._calculate_temporary_impact(df)
        
        return metrics
    
    def _calculate_vpin(self, df: pd.DataFrame) -> float:
        """Calculate Volume-synchronized Probability of Informed Trading"""
        if len(df) < 10:
            return 0.5  # Default neutral value
        
        # Simplified VPIN calculation
        # Group trades into volume buckets
        bucket_size = df['TradeQuantity'].sum() / 10 if df['TradeQuantity'].sum() > 0 else 1
        
        df['volume_bucket'] = (df['TradeQuantity'].cumsum() / bucket_size).astype(int)
        
        vpin_values = []
        for bucket in df['volume_bucket'].unique():
            bucket_trades = df[df['volume_bucket'] == bucket]
            buy_vol = bucket_trades[bucket_trades['Aggressor'].str.contains('Buy', na=False)]['TradeQuantity'].sum()
            sell_vol = bucket_trades[bucket_trades['Aggressor'].str.contains('Sell', na=False)]['TradeQuantity'].sum()
            total_vol = buy_vol + sell_vol
            
            if total_vol > 0:
                vpin = abs(buy_vol - sell_vol) / total_vol
                vpin_values.append(vpin)
        
        return np.mean(vpin_values) if vpin_values else 0.5
    
    def _calculate_spread_resiliency(self, df: pd.DataFrame) -> float:
        """Calculate how quickly spreads recover after trades"""
        if len(df) < 2:
            return 1.0
        
        # Look at spread changes after large trades
        large_trades = df[df['TradeQuantity'] >= df['TradeQuantity'].quantile(0.75)]
        
        if len(large_trades) == 0:
            return 1.0
        
        resiliency_scores = []
        for idx in large_trades.index:
            if idx + 1 < len(df):
                pre_spread = df.loc[idx, 'spread']
                post_spread = df.loc[idx + 1, 'spread']
                if pre_spread > 0:
                    resiliency = 1 - abs(post_spread - pre_spread) / pre_spread
                    resiliency_scores.append(max(0, resiliency))
        
        return np.mean(resiliency_scores) if resiliency_scores else 1.0
    
    def _calculate_adverse_selection(self, df: pd.DataFrame) -> float:
        """Calculate adverse selection component of spread"""
        if len(df) < 5:
            return 0.0
        
        # Look at price movement after trades
        adverse_selection_costs = []
        
        for i in range(len(df) - 5):
            trade = df.iloc[i]
            future_mid = df.iloc[i+1:i+6]['mid_price'].mean()
            
            if 'Buy' in trade['Aggressor']:
                cost = future_mid - trade['Trade_Price']
            else:
                cost = trade['Trade_Price'] - future_mid
            
            adverse_selection_costs.append(max(0, cost))
        
        return np.mean(adverse_selection_costs) if adverse_selection_costs else 0.0
    
    def _detect_hidden_liquidity(self, df: pd.DataFrame) -> float:
        """Detect presence of hidden/iceberg orders"""
        if len(df) < 5:
            return 0.0
        
        # Look for repeated trades at same price/size
        df['trade_signature'] = df['Trade_Price'].astype(str) + '_' + df['TradeQuantity'].astype(str)
        signature_counts = df['trade_signature'].value_counts()
        
        # High repetition suggests algorithmic/hidden orders
        repetition_score = (signature_counts > 2).sum() / len(signature_counts) if len(signature_counts) > 0 else 0
        
        return min(1.0, repetition_score * 2)  # Scale to 0-1
    
    def _calculate_permanent_impact(self, df: pd.DataFrame) -> float:
        """Calculate permanent price impact of trades"""
        if len(df) < 10:
            return 0.0
        
        # Look at price level 5 trades later
        impacts = []
        for i in range(len(df) - 5):
            current_price = df.iloc[i]['Trade_Price']
            future_price = df.iloc[i+5]['Trade_Price']
            
            if 'Buy' in df.iloc[i]['Aggressor']:
                impact = (future_price - current_price) / current_price
            else:
                impact = (current_price - future_price) / current_price
            
            impacts.append(impact)
        
        return np.mean(impacts) if impacts else 0.0
    
    def _calculate_temporary_impact(self, df: pd.DataFrame) -> float:
        """Calculate temporary price impact (immediate reversion)"""
        if len(df) < 3:
            return 0.0
        
        # Look at immediate price reversion
        reversions = []
        for i in range(1, len(df) - 1):
            pre_price = df.iloc[i-1]['Trade_Price']
            trade_price = df.iloc[i]['Trade_Price']
            post_price = df.iloc[i+1]['Trade_Price']
            
            if 'Buy' in df.iloc[i]['Aggressor']:
                reversion = (trade_price - pre_price) - (post_price - trade_price)
            else:
                reversion = (pre_price - trade_price) - (trade_price - post_price)
            
            reversions.append(max(0, reversion))
        
        return np.mean(reversions) if reversions else 0.0

def integrate_microstructure_with_alpha(
    alpha_signals: list,
    microstructure_analysis: dict
) -> list:
    """Enhance alpha signals with microstructure analysis"""
    
    # Extract key metrics
    toxicity = microstructure_analysis.get('toxicity_metrics', {})
    liquidity = microstructure_analysis.get('liquidity_metrics', {})
    mm_behavior = microstructure_analysis.get('market_maker_behavior', {})
    
    # Enhance each signal
    for signal in alpha_signals:
        # Adjust confidence based on microstructure
        if toxicity.get('vpin', 0.5) > 0.7:
            signal.confidence *= 1.1  # Higher toxicity = more informed flow
        
        if liquidity.get('avg_effective_spread', 0) > liquidity.get('avg_spread', 1) * 1.5:
            signal.urgency_score *= 1.2  # Wide effective spread = urgent
        
        if mm_behavior.get('mm_inventory_pressure', 0) > 0.7:
            signal.metadata['mm_hedging_likely'] = True
            signal.trade_recommendation += " (MM hedging likely)"
        
        # Add microstructure metadata
        signal.metadata['microstructure'] = {
            'vpin': toxicity.get('vpin', 0.5),
            'order_imbalance': toxicity.get('order_imbalance', 0),
            'spread_conditions': liquidity.get('avg_effective_spread', 0),
            'mm_participation': mm_behavior.get('mm_participation_rate', 0)
        }
    
    return alpha_signals