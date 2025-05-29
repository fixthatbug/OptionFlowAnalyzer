# analysis_modules/flow_calculator.py
"""
Options flow calculation and analysis module
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict

class FlowCalculator:
    """Calculates comprehensive flow metrics for options trading data"""
    
    def __init__(self):
        self.flow_cache = {}
        self.aggregation_intervals = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '1hour': timedelta(hours=1)
        }
    
    def calculate_comprehensive_flow_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate comprehensive flow metrics"""
        
        if df.empty:
            return self._empty_flow_metrics()
        
        flow_metrics = {
            'basic_metrics': self._calculate_basic_metrics(df),
            'directional_flow': self._calculate_directional_flow(df),
            'size_analysis': self._calculate_size_analysis(df),
            'time_analysis': self._calculate_time_analysis(df),
            'exchange_analysis': self._calculate_exchange_analysis(df),
            'premium_flow': self._calculate_premium_flow(df),
            'volatility_flow': self._calculate_volatility_flow(df),
            'greek_weighted_flow': self._calculate_greek_weighted_flow(df),
            'momentum_indicators': self._calculate_momentum_indicators(df),
            'concentration_metrics': self._calculate_concentration_metrics(df),
            'efficiency_metrics': self._calculate_efficiency_metrics(df),
            'significant_flows': self._identify_significant_flows(df)
        }
        
        # Calculate composite scores
        flow_metrics['composite_scores'] = self._calculate_composite_scores(flow_metrics)
        
        return flow_metrics
    
    def _empty_flow_metrics(self) -> dict:
        """Return empty flow metrics structure"""
        return {
            'basic_metrics': {},
            'directional_flow': {},
            'size_analysis': {},
            'time_analysis': {},
            'exchange_analysis': {},
            'premium_flow': {},
            'volatility_flow': {},
            'greek_weighted_flow': {},
            'momentum_indicators': {},
            'concentration_metrics': {},
            'efficiency_metrics': {},
            'significant_flows': [],
            'composite_scores': {}
        }
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate basic flow metrics"""
        
        metrics = {
            'total_trades': len(df),
            'total_contracts': df['TradeQuantity'].sum() if 'TradeQuantity' in df.columns else 0,
            'total_premium': df['NotionalValue'].sum() if 'NotionalValue' in df.columns else 0,
            'avg_trade_size': df['TradeQuantity'].mean() if 'TradeQuantity' in df.columns else 0,
            'median_trade_size': df['TradeQuantity'].median() if 'TradeQuantity' in df.columns else 0,
            'avg_premium_per_trade': df['NotionalValue'].mean() if 'NotionalValue' in df.columns else 0,
            'unique_options': df['StandardOptionSymbol'].nunique() if 'StandardOptionSymbol' in df.columns else 0,
            'unique_strikes': df['Strike_Price_calc'].nunique() if 'Strike_Price_calc' in df.columns else 0,
            'unique_expirations': df['Expiration_Date_calc'].nunique() if 'Expiration_Date_calc' in df.columns else 0
        }
        
        # Price statistics
        if 'Trade_Price' in df.columns:
            metrics.update({
                'avg_price': df['Trade_Price'].mean(),
                'price_range': df['Trade_Price'].max() - df['Trade_Price'].min(),
                'price_std': df['Trade_Price'].std()
            })
        
        # Time span
        if 'Time_dt' in df.columns and len(df) > 1:
            time_span = df['Time_dt'].max() - df['Time_dt'].min()
            metrics['time_span_minutes'] = time_span.total_seconds() / 60
            metrics['trades_per_minute'] = len(df) / max(1, metrics['time_span_minutes'])
        
        return metrics
    
    def _calculate_directional_flow(self, df: pd.DataFrame) -> dict:
        """Calculate directional flow metrics"""
        
        directional = {
            'call_volume': 0,
            'put_volume': 0,
            'call_premium': 0,
            'put_premium': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'buy_premium': 0,
            'sell_premium': 0
        }
        
        # Call vs Put analysis
        if 'Option_Type_calc' in df.columns:
            calls = df[df['Option_Type_calc'] == 'Call']
            puts = df[df['Option_Type_calc'] == 'Put']
            
            directional['call_volume'] = calls['TradeQuantity'].sum() if 'TradeQuantity' in calls.columns else 0
            directional['put_volume'] = puts['TradeQuantity'].sum() if 'TradeQuantity' in puts.columns else 0
            directional['call_premium'] = calls['NotionalValue'].sum() if 'NotionalValue' in calls.columns else 0
            directional['put_premium'] = puts['NotionalValue'].sum() if 'NotionalValue' in puts.columns else 0
            
            # Calculate ratios
            if directional['put_volume'] > 0:
                directional['call_put_volume_ratio'] = directional['call_volume'] / directional['put_volume']
            else:
                directional['call_put_volume_ratio'] = float('inf') if directional['call_volume'] > 0 else 0
            
            if directional['put_premium'] > 0:
                directional['call_put_premium_ratio'] = directional['call_premium'] / directional['put_premium']
            else:
                directional['call_put_premium_ratio'] = float('inf') if directional['call_premium'] > 0 else 0
        
        # Buy vs Sell analysis
        if 'Aggressor' in df.columns:
            buy_trades = df[df['Aggressor'].str.contains('Buy', na=False)]
            sell_trades = df[df['Aggressor'].str.contains('Sell', na=False)]
            
            directional['buy_volume'] = buy_trades['TradeQuantity'].sum() if 'TradeQuantity' in buy_trades.columns else 0
            directional['sell_volume'] = sell_trades['TradeQuantity'].sum() if 'TradeQuantity' in sell_trades.columns else 0
            directional['buy_premium'] = buy_trades['NotionalValue'].sum() if 'NotionalValue' in buy_trades.columns else 0
            directional['sell_premium'] = sell_trades['NotionalValue'].sum() if 'NotionalValue' in sell_trades.columns else 0
            
            # Net flow
            directional['net_volume'] = directional['buy_volume'] - directional['sell_volume']
            directional['net_premium'] = directional['buy_premium'] - directional['sell_premium']
            
            # Flow bias
            total_volume = directional['buy_volume'] + directional['sell_volume']
            if total_volume > 0:
                directional['buy_flow_percentage'] = (directional['buy_volume'] / total_volume) * 100
                directional['sell_flow_percentage'] = (directional['sell_volume'] / total_volume) * 100
        
        return directional
    
    def _calculate_size_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze trade size distribution"""
        
        if 'TradeQuantity' not in df.columns or df.empty:
            return {}
        
        sizes = df['TradeQuantity']
        
        size_analysis = {
            'min_size': sizes.min(),
            'max_size': sizes.max(),
            'median_size': sizes.median(),
            'q25_size': sizes.quantile(0.25),
            'q75_size': sizes.quantile(0.75),
            'q95_size': sizes.quantile(0.95),
            'size_std': sizes.std(),
            'size_skew': sizes.skew(),
            'size_kurtosis': sizes.kurtosis()
        }
        
        # Size categories
        size_analysis['small_trades'] = (sizes <= 10).sum()  # <= 10 contracts
        size_analysis['medium_trades'] = ((sizes > 10) & (sizes <= 50)).sum()  # 11-50 contracts
        size_analysis['large_trades'] = ((sizes > 50) & (sizes <= 100)).sum()  # 51-100 contracts
        size_analysis['block_trades'] = (sizes > 100).sum()  # > 100 contracts
        
        # Percentages
        total_trades = len(sizes)
        if total_trades > 0:
            size_analysis['small_trades_pct'] = (size_analysis['small_trades'] / total_trades) * 100
            size_analysis['medium_trades_pct'] = (size_analysis['medium_trades'] / total_trades) * 100
            size_analysis['large_trades_pct'] = (size_analysis['large_trades'] / total_trades) * 100
            size_analysis['block_trades_pct'] = (size_analysis['block_trades'] / total_trades) * 100
        
        # Volume by size category
        small_volume = sizes[sizes <= 10].sum()
        medium_volume = sizes[(sizes > 10) & (sizes <= 50)].sum()
        large_volume = sizes[(sizes > 50) & (sizes <= 100)].sum()
        block_volume = sizes[sizes > 100].sum()
        
        total_volume = sizes.sum()
        if total_volume > 0:
            size_analysis['small_volume_pct'] = (small_volume / total_volume) * 100
            size_analysis['medium_volume_pct'] = (medium_volume / total_volume) * 100
            size_analysis['large_volume_pct'] = (large_volume / total_volume) * 100
            size_analysis['block_volume_pct'] = (block_volume / total_volume) * 100
        
        return size_analysis
    
    def _calculate_time_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze temporal patterns in flow"""
        
        if 'Time_dt' not in df.columns or df.empty:
            return {}
        
        df_time = df.copy()
        df_time['hour'] = df_time['Time_dt'].dt.hour
        df_time['minute'] = df_time['Time_dt'].dt.minute
        
        time_analysis = {
            'trading_hours': {},
            'minute_distribution': {},
            'intensity_metrics': {}
        }
        
        # Hourly analysis
        hourly_volume = df_time.groupby('hour')['TradeQuantity'].sum() if 'TradeQuantity' in df.columns else pd.Series()
        hourly_trades = df_time.groupby('hour').size()
        
        time_analysis['trading_hours']['volume_by_hour'] = hourly_volume.to_dict()
        time_analysis['trading_hours']['trades_by_hour'] = hourly_trades.to_dict()
        
        if not hourly_volume.empty:
            peak_hour = hourly_volume.idxmax()
            time_analysis['trading_hours']['peak_volume_hour'] = int(peak_hour)
            time_analysis['trading_hours']['peak_volume'] = int(hourly_volume.max())
        
        # Minute-level clustering
        if len(df_time) > 1:
            time_diffs = df_time['Time_dt'].diff().dt.total_seconds()
            time_analysis['intensity_metrics']['avg_time_between_trades'] = time_diffs.mean()
            time_analysis['intensity_metrics']['median_time_between_trades'] = time_diffs.median()
            time_analysis['intensity_metrics']['min_time_between_trades'] = time_diffs.min()
            
            # Burst detection (trades within 5 seconds)
            burst_threshold = 5  # seconds
            bursts = (time_diffs <= burst_threshold).sum()
            time_analysis['intensity_metrics']['burst_trades'] = bursts
            time_analysis['intensity_metrics']['burst_percentage'] = (bursts / len(time_diffs)) * 100 if len(time_diffs) > 0 else 0
        
        return time_analysis
    
    def _calculate_exchange_analysis(self, df: pd.DataFrame) -> dict:
        """Analyze flow by exchange"""
        
        if 'Exchange' not in df.columns or df.empty:
            return {}
        
        exchange_analysis = {}
        
        # Volume by exchange
        exchange_volume = df.groupby('Exchange')['TradeQuantity'].sum() if 'TradeQuantity' in df.columns else pd.Series()
        exchange_trades = df.groupby('Exchange').size()
        exchange_premium = df.groupby('Exchange')['NotionalValue'].sum() if 'NotionalValue' in df.columns else pd.Series()
        
        exchange_analysis['volume_by_exchange'] = exchange_volume.to_dict()
        exchange_analysis['trades_by_exchange'] = exchange_trades.to_dict()
        exchange_analysis['premium_by_exchange'] = exchange_premium.to_dict()
        
        # Market share
        total_volume = exchange_volume.sum()
        if total_volume > 0:
            market_share = (exchange_volume / total_volume * 100).to_dict()
            exchange_analysis['market_share'] = market_share
            
            # Concentration
            exchange_analysis['exchange_concentration'] = self._calculate_herfindahl_index(exchange_volume)
        
        # Average trade size by exchange
        avg_size_by_exchange = df.groupby('Exchange')['TradeQuantity'].mean() if 'TradeQuantity' in df.columns else pd.Series()
        exchange_analysis['avg_trade_size_by_exchange'] = avg_size_by_exchange.to_dict()
        
        return exchange_analysis
    
    def _calculate_premium_flow(self, df: pd.DataFrame) -> dict:
        """Analyze premium flow patterns"""
        
        if 'NotionalValue' not in df.columns or df.empty:
            return {}
        
        premium_flow = {
            'total_premium_traded': df['NotionalValue'].sum(),
            'avg_premium_per_trade': df['NotionalValue'].mean(),
            'median_premium_per_trade': df['NotionalValue'].median(),
            'premium_std': df['NotionalValue'].std(),
            'max_single_trade_notional': df['NotionalValue'].max(),
            'min_single_trade_notional': df['NotionalValue'].min()
        }
        
        # Premium percentiles
        premium_flow['premium_q25'] = df['NotionalValue'].quantile(0.25)
        premium_flow['premium_q75'] = df['NotionalValue'].quantile(0.75)
        premium_flow['premium_q95'] = df['NotionalValue'].quantile(0.95)
        premium_flow['premium_q99'] = df['NotionalValue'].quantile(0.99)
        
        # Premium categories
        small_premium = df['NotionalValue'] <= 10000  # <= $10K
        medium_premium = (df['NotionalValue'] > 10000) & (df['NotionalValue'] <= 50000)  # $10K-$50K
        large_premium = (df['NotionalValue'] > 50000) & (df['NotionalValue'] <= 100000)  # $50K-$100K
        block_premium = df['NotionalValue'] > 100000  # > $100K
        
        premium_flow['small_premium_trades'] = small_premium.sum()
        premium_flow['medium_premium_trades'] = medium_premium.sum()
        premium_flow['large_premium_trades'] = large_premium.sum()
        premium_flow['block_premium_trades'] = block_premium.sum()
        
        # Premium concentration
        total_premium = df['NotionalValue'].sum()
        if total_premium > 0:
            premium_flow['small_premium_pct'] = (df[small_premium]['NotionalValue'].sum() / total_premium) * 100
            premium_flow['medium_premium_pct'] = (df[medium_premium]['NotionalValue'].sum() / total_premium) * 100
            premium_flow['large_premium_pct'] = (df[large_premium]['NotionalValue'].sum() / total_premium) * 100
            premium_flow['block_premium_pct'] = (df[block_premium]['NotionalValue'].sum() / total_premium) * 100
        
        # Top trades by premium
        top_trades = df.nlargest(10, 'NotionalValue')[['StandardOptionSymbol', 'TradeQuantity', 'Trade_Price', 'NotionalValue', 'Aggressor']]
        premium_flow['top_premium_trades'] = top_trades.to_dict('records')
        
        return premium_flow
    
    def _calculate_volatility_flow(self, df: pd.DataFrame) -> dict:
        """Analyze flow patterns related to volatility"""
        
        if 'IV' not in df.columns or df.empty:
            return {}
        
        vol_flow = {
            'avg_iv': df['IV'].mean(),
            'median_iv': df['IV'].median(),
            'iv_std': df['IV'].std(),
            'min_iv': df['IV'].min(),
            'max_iv': df['IV'].max(),
            'iv_range': df['IV'].max() - df['IV'].min()
        }
        
        # IV percentiles
        vol_flow['iv_q25'] = df['IV'].quantile(0.25)
        vol_flow['iv_q75'] = df['IV'].quantile(0.75)
        vol_flow['iv_q95'] = df['IV'].quantile(0.95)
        
        # Volume-weighted IV
        if 'TradeQuantity' in df.columns:
            total_volume = df['TradeQuantity'].sum()
            if total_volume > 0:
                vol_flow['volume_weighted_iv'] = (df['IV'] * df['TradeQuantity']).sum() / total_volume
        
        # Premium-weighted IV
        if 'NotionalValue' in df.columns:
            total_premium = df['NotionalValue'].sum()
            if total_premium > 0:
                vol_flow['premium_weighted_iv'] = (df['IV'] * df['NotionalValue']).sum() / total_premium
        
        # IV categories
        low_iv = df['IV'] <= df['IV'].quantile(0.33)
        med_iv = (df['IV'] > df['IV'].quantile(0.33)) & (df['IV'] <= df['IV'].quantile(0.67))
        high_iv = df['IV'] > df['IV'].quantile(0.67)
        
        vol_flow['low_iv_trades'] = low_iv.sum()
        vol_flow['med_iv_trades'] = med_iv.sum()
        vol_flow['high_iv_trades'] = high_iv.sum()
        
        # Volume in each IV category
        if 'TradeQuantity' in df.columns:
            vol_flow['low_iv_volume'] = df[low_iv]['TradeQuantity'].sum()
            vol_flow['med_iv_volume'] = df[med_iv]['TradeQuantity'].sum()
            vol_flow['high_iv_volume'] = df[high_iv]['TradeQuantity'].sum()
        
        return vol_flow
    
    def _calculate_greek_weighted_flow(self, df: pd.DataFrame) -> dict:
        """Calculate Greek-weighted flow metrics"""
        
        greek_flow = {}
        
        # Delta-weighted metrics
        if 'Delta' in df.columns and 'TradeQuantity' in df.columns:
            df_clean = df.dropna(subset=['Delta', 'TradeQuantity'])
            if not df_clean.empty:
                greek_flow['total_delta_exposure'] = (df_clean['Delta'] * df_clean['TradeQuantity'] * 100).sum()
                greek_flow['avg_delta'] = df_clean['Delta'].mean()
                greek_flow['volume_weighted_delta'] = (df_clean['Delta'] * df_clean['TradeQuantity']).sum() / df_clean['TradeQuantity'].sum()
                
                # Net delta by direction
                if 'Aggressor' in df.columns:
                    buy_trades = df_clean[df_clean['Aggressor'].str.contains('Buy', na=False)]
                    sell_trades = df_clean[df_clean['Aggressor'].str.contains('Sell', na=False)]
                    
                    buy_delta_exposure = (buy_trades['Delta'] * buy_trades['TradeQuantity'] * 100).sum()
                    sell_delta_exposure = (sell_trades['Delta'] * sell_trades['TradeQuantity'] * 100).sum()
                    
                    greek_flow['net_delta_exposure'] = buy_delta_exposure - sell_delta_exposure
                    greek_flow['buy_delta_exposure'] = buy_delta_exposure
                    greek_flow['sell_delta_exposure'] = sell_delta_exposure
        
        # Gamma-weighted metrics
        if 'Gamma' in df.columns and 'TradeQuantity' in df.columns:
            df_clean = df.dropna(subset=['Gamma', 'TradeQuantity'])
            if not df_clean.empty:
                greek_flow['total_gamma_exposure'] = (df_clean['Gamma'] * df_clean['TradeQuantity'] * 100).sum()
                greek_flow['avg_gamma'] = df_clean['Gamma'].mean()
                greek_flow['volume_weighted_gamma'] = (df_clean['Gamma'] * df_clean['TradeQuantity']).sum() / df_clean['TradeQuantity'].sum()
        
        # Theta-weighted metrics
        if 'Theta' in df.columns and 'TradeQuantity' in df.columns:
            df_clean = df.dropna(subset=['Theta', 'TradeQuantity'])
            if not df_clean.empty:
                greek_flow['total_theta_exposure'] = (df_clean['Theta'] * df_clean['TradeQuantity'] * 100).sum()
                greek_flow['avg_theta'] = df_clean['Theta'].mean()
        
        # Vega-weighted metrics
        if 'Vega' in df.columns and 'TradeQuantity' in df.columns:
            df_clean = df.dropna(subset=['Vega', 'TradeQuantity'])
            if not df_clean.empty:
                greek_flow['total_vega_exposure'] = (df_clean['Vega'] * df_clean['TradeQuantity'] * 100).sum()
                greek_flow['avg_vega'] = df_clean['Vega'].mean()
        
        return greek_flow
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate flow momentum indicators"""
        
        if len(df) < 10:
            return {}
        
        momentum = {}
        
        # Sort by time
        df_sorted = df.sort_values('Time_dt') if 'Time_dt' in df.columns else df
        
        # Volume momentum
        if 'TradeQuantity' in df.columns:
            volumes = df_sorted['TradeQuantity']
            momentum['volume_sma_10'] = volumes.rolling(10).mean().iloc[-1] if len(volumes) >= 10 else volumes.mean()
            momentum['volume_sma_30'] = volumes.rolling(30).mean().iloc[-1] if len(volumes) >= 30 else volumes.mean()
            
            if momentum['volume_sma_30'] > 0:
                momentum['volume_momentum'] = momentum['volume_sma_10'] / momentum['volume_sma_30']
            
            # Volume acceleration
            recent_avg = volumes.tail(10).mean()
            earlier_avg = volumes.iloc[-30:-10].mean() if len(volumes) >= 30 else volumes.mean()
            if earlier_avg > 0:
                momentum['volume_acceleration'] = (recent_avg - earlier_avg) / earlier_avg
        
        # Premium momentum
        if 'NotionalValue' in df.columns:
            premiums = df_sorted['NotionalValue']
            momentum['premium_sma_10'] = premiums.rolling(10).mean().iloc[-1] if len(premiums) >= 10 else premiums.mean()
            momentum['premium_sma_30'] = premiums.rolling(30).mean().iloc[-1] if len(premiums) >= 30 else premiums.mean()
            
            if momentum['premium_sma_30'] > 0:
                momentum['premium_momentum'] = momentum['premium_sma_10'] / momentum['premium_sma_30']
        
        # Directional momentum
        if 'Aggressor' in df.columns and 'TradeQuantity' in df.columns:
            df_sorted['signed_volume'] = df_sorted.apply(
                lambda x: x['TradeQuantity'] if 'Buy' in str(x['Aggressor']) else -x['TradeQuantity'], axis=1
            )
            
            momentum['cumulative_flow'] = df_sorted['signed_volume'].cumsum().iloc[-1]
            momentum['flow_trend'] = df_sorted['signed_volume'].rolling(10).mean().iloc[-1] if len(df_sorted) >= 10 else 0
            
            # Flow persistence (autocorrelation)
            if len(df_sorted) > 1:
                try:
                    momentum['flow_persistence'] = df_sorted['signed_volume'].autocorr(lag=1)
                except:
                    momentum['flow_persistence'] = 0
        
        return momentum
    
    def _calculate_concentration_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate flow concentration metrics"""
        
        concentration = {}
        
        # Symbol concentration
        if 'StandardOptionSymbol' in df.columns and 'TradeQuantity' in df.columns:
            symbol_volume = df.groupby('StandardOptionSymbol')['TradeQuantity'].sum()
            concentration['symbol_herfindahl'] = self._calculate_herfindahl_index(symbol_volume)
            
            # Top 5 symbols percentage
            top_5_volume = symbol_volume.nlargest(5).sum()
            total_volume = symbol_volume.sum()
            if total_volume > 0:
                concentration['top_5_symbols_pct'] = (top_5_volume / total_volume) * 100
            
            concentration['symbol_count'] = len(symbol_volume)
        
        # Strike concentration
        if 'Strike_Price_calc' in df.columns and 'TradeQuantity' in df.columns:
            strike_volume = df.groupby('Strike_Price_calc')['TradeQuantity'].sum()
            concentration['strike_herfindahl'] = self._calculate_herfindahl_index(strike_volume)
            
            # Most active strike
            if not strike_volume.empty:
                most_active_strike = strike_volume.idxmax()
                concentration['most_active_strike'] = most_active_strike
                concentration['most_active_strike_volume'] = strike_volume.max()
                concentration['most_active_strike_pct'] = (strike_volume.max() / strike_volume.sum()) * 100
        
        # Expiration concentration
        if 'Expiration_Date_calc' in df.columns and 'TradeQuantity' in df.columns:
            exp_volume = df.groupby('Expiration_Date_calc')['TradeQuantity'].sum()
            concentration['expiration_herfindahl'] = self._calculate_herfindahl_index(exp_volume)
        
        return concentration
    
    def _calculate_efficiency_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate flow efficiency metrics"""
        
        efficiency = {}
        
        # Bid-ask spread analysis
        if all(col in df.columns for col in ['Option_Bid', 'Option_Ask', 'Trade_Price']):
            df_clean = df.dropna(subset=['Option_Bid', 'Option_Ask', 'Trade_Price'])
            if not df_clean.empty:
                df_clean['spread'] = df_clean['Option_Ask'] - df_clean['Option_Bid']
                df_clean['mid_price'] = (df_clean['Option_Ask'] + df_clean['Option_Bid']) / 2
                df_clean['price_improvement'] = abs(df_clean['Trade_Price'] - df_clean['mid_price'])
                
                efficiency['avg_spread'] = df_clean['spread'].mean()
                efficiency['avg_price_improvement'] = df_clean['price_improvement'].mean()
                efficiency['spread_efficiency'] = (df_clean['price_improvement'] / df_clean['spread']).mean()
        
        # Volume efficiency (large trades vs small trades impact)
        if 'TradeQuantity' in df.columns and 'Trade_Price' in df.columns:
            large_trades = df[df['TradeQuantity'] >= 50]
            small_trades = df[df['TradeQuantity'] < 50]
            
            if not large_trades.empty and not small_trades.empty:
                large_price_impact = large_trades['Trade_Price'].std()
                small_price_impact = small_trades['Trade_Price'].std()
                
                if small_price_impact > 0:
                    efficiency['size_impact_ratio'] = large_price_impact / small_price_impact
        
        return efficiency
    
    def _identify_significant_flows(self, df: pd.DataFrame) -> list[dict]:
        """Identify significant flow events"""
        
        significant_flows = []
        
        if df.empty:
            return significant_flows
        
        # Large volume trades
        if 'TradeQuantity' in df.columns:
            volume_threshold = df['TradeQuantity'].quantile(0.95)
            large_volume_trades = df[df['TradeQuantity'] >= volume_threshold]
            
            for _, trade in large_volume_trades.iterrows():
                significant_flows.append({
                    'type': 'large_volume',
                    'symbol': trade.get('StandardOptionSymbol', ''),
                    'volume': trade.get('TradeQuantity', 0),
                    'price': trade.get('Trade_Price', 0),
                    'notional': trade.get('NotionalValue', 0),
                    'timestamp': trade.get('Time_dt', ''),
                    'aggressor': trade.get('Aggressor', ''),
                    'significance_score': trade.get('TradeQuantity', 0) / df['TradeQuantity'].mean() if df['TradeQuantity'].mean() > 0 else 0
                })
        
        # Large premium trades
        if 'NotionalValue' in df.columns:
            premium_threshold = df['NotionalValue'].quantile(0.95)
            large_premium_trades = df[df['NotionalValue'] >= premium_threshold]
            
            for _, trade in large_premium_trades.iterrows():
                if not any(flow['symbol'] == trade.get('StandardOptionSymbol', '') and 
                          flow['timestamp'] == trade.get('Time_dt', '') for flow in significant_flows):
                    significant_flows.append({
                        'type': 'large_premium',
                        'symbol': trade.get('StandardOptionSymbol', ''),
                        'volume': trade.get('TradeQuantity', 0),
                        'price': trade.get('Trade_Price', 0),
                        'notional': trade.get('NotionalValue', 0),
                        'timestamp': trade.get('Time_dt', ''),
                        'aggressor': trade.get('Aggressor', ''),
                        'significance_score': trade.get('NotionalValue', 0) / df['NotionalValue'].mean() if df['NotionalValue'].mean() > 0 else 0
                    })
        
        # Sort by significance score
        significant_flows.sort(key=lambda x: x['significance_score'], reverse=True)
        
        return significant_flows[:20]  # Top 20 significant flows
    
    def _calculate_composite_scores(self, flow_metrics: dict) -> dict:
        """Calculate composite flow scores"""
        
        scores = {
            'flow_intensity': 0,
            'directional_bias': 0,
            'institutional_presence': 0,
            'market_impact': 0,
            'overall_flow_score': 0
        }
        
        # Flow intensity (based on volume and frequency)
        basic = flow_metrics.get('basic_metrics', {})
        if basic.get('trades_per_minute', 0) > 0:
            scores['flow_intensity'] = min(100, basic['trades_per_minute'] * 10)
        
        # Directional bias
        directional = flow_metrics.get('directional_flow', {})
        call_put_ratio = directional.get('call_put_volume_ratio', 1)
        if call_put_ratio != 1:
            scores['directional_bias'] = min(100, abs(call_put_ratio - 1) * 50)
        
        # Institutional presence (based on large trades)
        size_analysis = flow_metrics.get('size_analysis', {})
        block_pct = size_analysis.get('block_trades_pct', 0)
        scores['institutional_presence'] = min(100, block_pct * 2)
        
        # Market impact (based on premium and concentration)
        premium = flow_metrics.get('premium_flow', {})
        total_premium = premium.get('total_premium_traded', 0)
        if total_premium > 0:
            scores['market_impact'] = min(100, (total_premium / 1000000) * 10)  # $1M = 10 points
        
        # Overall score (weighted average)
        weights = {'flow_intensity': 0.25, 'directional_bias': 0.25, 'institutional_presence': 0.3, 'market_impact': 0.2}
        scores['overall_flow_score'] = sum(scores[key] * weights[key] for key in weights)
        
        return scores
    
    def _calculate_herfindahl_index(self, series: pd.Series) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        
        if series.empty or series.sum() == 0:
            return 0
        
        shares = series / series.sum()
        return (shares ** 2).sum()
    
    def calculate_flow_comparison(self, current_df: pd.DataFrame, historical_df: pd.DataFrame) -> dict:
        """Compare current flow to historical patterns"""
        
        current_metrics = self.calculate_comprehensive_flow_metrics(current_df)
        historical_metrics = self.calculate_comprehensive_flow_metrics(historical_df)
        
        comparison = {
            'volume_change': self._calculate_percentage_change(
                current_metrics['basic_metrics'].get('total_contracts', 0),
                historical_metrics['basic_metrics'].get('total_contracts', 0)
            ),
            'premium_change': self._calculate_percentage_change(
                current_metrics['basic_metrics'].get('total_premium', 0),
                historical_metrics['basic_metrics'].get('total_premium', 0)
            ),
            'intensity_change': self._calculate_percentage_change(
                current_metrics['composite_scores'].get('flow_intensity', 0),
                historical_metrics['composite_scores'].get('flow_intensity', 0)
            ),
            'directional_shift': self._calculate_directional_shift(
                current_metrics['directional_flow'],
                historical_metrics['directional_flow']
            )
        }
        
        return comparison
    
    def _calculate_percentage_change(self, current: float, historical: float) -> float:
        """Calculate percentage change"""
        if historical == 0:
            return 0 if current == 0 else 100
        return ((current - historical) / historical) * 100
    
    def _calculate_directional_shift(self, current_dir: dict, historical_dir: dict) -> dict:
        """Calculate shift in directional bias"""
        
        current_ratio = current_dir.get('call_put_volume_ratio', 1)
        historical_ratio = historical_dir.get('call_put_volume_ratio', 1)
        
        return {
            'call_put_ratio_change': current_ratio - historical_ratio,
            'bias_shift': 'more_bullish' if current_ratio > historical_ratio else 'more_bearish' if current_ratio < historical_ratio else 'no_change'
        }