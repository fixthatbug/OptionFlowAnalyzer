# analysis_modules/volatility_analyzer.py
"""
Implied volatility and volatility surface analysis
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class VolatilityAnalyzer:
    """Analyzes implied volatility patterns and dynamics"""
    
    def __init__(self):
        self.iv_percentiles = {}
        self.vol_cache = {}
        
    def analyze_volatility_dynamics(self, df: pd.DataFrame) -> dict:
        """Comprehensive volatility analysis"""
        
        if df.empty or 'IV' not in df.columns:
            return self._empty_volatility_results()
        
        vol_analysis = {
            'basic_iv_stats': self._calculate_basic_iv_stats(df),
            'iv_distribution': self._analyze_iv_distribution(df),
            'vol_surface': self._analyze_volatility_surface(df),
            'vol_skew': self._analyze_volatility_skew(df),
            'term_structure': self._analyze_term_structure(df),
            'vol_smile': self._analyze_volatility_smile(df),
            'iv_rank_percentile': self._calculate_iv_rank_percentile(df),
            'vol_clustering': self._detect_volatility_clustering(df),
            'vol_mean_reversion': self._analyze_vol_mean_reversion(df),
            'vol_spillovers': self._analyze_vol_spillovers(df),
            'unusual_iv_activity': self._detect_unusual_iv_activity(df),
            'vol_trading_opportunities': self._identify_vol_opportunities(df)
        }
        
        # Calculate composite volatility scores
        vol_analysis['composite_scores'] = self._calculate_vol_composite_scores(vol_analysis)
        
        return vol_analysis
    
    def _empty_volatility_results(self) -> dict:
        """Return empty volatility analysis structure"""
        return {
            'basic_iv_stats': {},
            'iv_distribution': {},
            'vol_surface': {},
            'vol_skew': {},
            'term_structure': {},
            'vol_smile': {},
            'iv_rank_percentile': {},
            'vol_clustering': {},
            'vol_mean_reversion': {},
            'vol_spillovers': {},
            'unusual_iv_activity': {},
            'vol_trading_opportunities': {},
            'composite_scores': {}
        }
    
    def _calculate_basic_iv_stats(self, df: pd.DataFrame) -> dict:
        """Calculate basic implied volatility statistics"""
        
        iv_data = df['IV'].dropna()
        
        if iv_data.empty:
            return {}
        
        stats = {
            'count': len(iv_data),
            'mean_iv': iv_data.mean(),
            'median_iv': iv_data.median(),
            'std_iv': iv_data.std(),
            'min_iv': iv_data.min(),
            'max_iv': iv_data.max(),
            'range_iv': iv_data.max() - iv_data.min(),
            'skewness': iv_data.skew(),
            'kurtosis': iv_data.kurtosis()
        }
        
        # Percentiles
        percentiles = [10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'iv_p{p}'] = iv_data.quantile(p/100)
        
        # Volume-weighted IV
        if 'TradeQuantity' in df.columns:
            df_clean = df.dropna(subset=['IV', 'TradeQuantity'])
            if not df_clean.empty:
                total_volume = df_clean['TradeQuantity'].sum()
                if total_volume > 0:
                    stats['volume_weighted_iv'] = (df_clean['IV'] * df_clean['TradeQuantity']).sum() / total_volume
        
        # Premium-weighted IV
        if 'NotionalValue' in df.columns:
            df_clean = df.dropna(subset=['IV', 'NotionalValue'])
            if not df_clean.empty:
                total_premium = df_clean['NotionalValue'].sum()
                if total_premium > 0:
                    stats['premium_weighted_iv'] = (df_clean['IV'] * df_clean['NotionalValue']).sum() / total_premium
        
        return stats
    
    def _analyze_iv_distribution(self, df: pd.DataFrame) -> dict:
        """Analyze the distribution of implied volatility"""
        
        iv_data = df['IV'].dropna()
        
        if len(iv_data) < 10:
            return {}
        
        distribution = {
            'normality_test': None,
            'distribution_type': 'unknown',
            'outliers': [],
            'clusters': []
        }
        
        # Test for normality
        try:
            statistic, p_value = stats.normaltest(iv_data)
            distribution['normality_test'] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except:
            pass
        
        # Identify outliers using IQR method
        q1 = iv_data.quantile(0.25)
        q3 = iv_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df['IV'] < lower_bound) | (df['IV'] > upper_bound)]
        if not outliers.empty:
            distribution['outliers'] = outliers[['StandardOptionSymbol', 'IV', 'TradeQuantity', 'Trade_Price']].to_dict('records')
        
        # Identify IV clusters
        distribution['clusters'] = self._identify_iv_clusters(iv_data)
        
        # Distribution shape analysis
        if iv_data.std() > 0:
            cv = iv_data.std() / iv_data.mean()
            distribution['coefficient_variation'] = cv
            
            if iv_data.skew() > 1:
                distribution['distribution_type'] = 'right_skewed'
            elif iv_data.skew() < -1:
                distribution['distribution_type'] = 'left_skewed'
            else:
                distribution['distribution_type'] = 'approximately_normal'
        
        return distribution
    
    def _analyze_volatility_surface(self, df: pd.DataFrame) -> dict:
        """Analyze volatility surface across strikes and expirations"""
        
        if not all(col in df.columns for col in ['IV', 'Strike_Price_calc', 'Expiration_Date_calc']):
            return {}
        
        # Create volatility surface grid
        df_clean = df.dropna(subset=['IV', 'Strike_Price_calc', 'Expiration_Date_calc'])
        
        if df_clean.empty:
            return {}
        
        surface = {
            'surface_points': [],
            'strike_iv_relationship': {},
            'time_iv_relationship': {},
            'surface_interpolation': {},
            'arbitrage_violations': []
        }
        
        # Group by strike and expiration
        surface_grid = df_clean.groupby(['Strike_Price_calc', 'Expiration_Date_calc']).agg({
            'IV': ['mean', 'std', 'count'],
            'TradeQuantity': 'sum',
            'NotionalValue': 'sum'
        }).reset_index()
        
        surface_grid.columns = ['strike', 'expiration', 'iv_mean', 'iv_std', 'trade_count', 'volume', 'notional']
        
        # Convert to surface points
        for _, row in surface_grid.iterrows():
            surface['surface_points'].append({
                'strike': row['strike'],
                'expiration': row['expiration'],
                'iv': row['iv_mean'],
                'iv_std': row['iv_std'],
                'volume': row['volume'],
                'liquidity_score': min(100, row['trade_count'] * 10)
            })
        
        # Analyze strike-IV relationship
        surface['strike_iv_relationship'] = self._analyze_strike_iv_relationship(df_clean)
        
        # Analyze time-IV relationship
        surface['time_iv_relationship'] = self._analyze_time_iv_relationship(df_clean)
        
        return surface
    
    def _analyze_volatility_skew(self, df: pd.DataFrame) -> dict:
        """Analyze volatility skew patterns"""
        
        if not all(col in df.columns for col in ['IV', 'Strike_Price_calc', 'Underlying_Price']):
            return {}
        
        df_clean = df.dropna(subset=['IV', 'Strike_Price_calc', 'Underlying_Price'])
        
        if df_clean.empty:
            return {}
        
        skew_analysis = {
            'put_skew': {},
            'call_skew': {},
            'skew_slope': 0,
            'term_skew': {},
            'skew_trading_signals': []
        }
        
        # Calculate moneyness
        df_clean['moneyness'] = df_clean['Strike_Price_calc'] / df_clean['Underlying_Price']
        
        # Separate calls and puts
        if 'Option_Type_calc' in df.columns:
            calls = df_clean[df_clean['Option_Type_calc'] == 'Call']
            puts = df_clean[df_clean['Option_Type_calc'] == 'Put']
            
            # Analyze put skew
            if not puts.empty:
                skew_analysis['put_skew'] = self._calculate_skew_metrics(puts, 'put')
            
            # Analyze call skew
            if not calls.empty:
                skew_analysis['call_skew'] = self._calculate_skew_metrics(calls, 'call')
        
        # Overall skew slope
        if len(df_clean) >= 3:
            try:
                # Fit linear regression: IV = a + b * moneyness
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_clean['moneyness'], df_clean['IV']
                )
                skew_analysis['skew_slope'] = slope
                skew_analysis['skew_r_squared'] = r_value ** 2
                skew_analysis['skew_significance'] = p_value < 0.05
            except:
                pass
        
        # Identify skew trading opportunities
        skew_analysis['skew_trading_signals'] = self._identify_skew_opportunities(df_clean)
        
        return skew_analysis
    
    def _analyze_term_structure(self, df: pd.DataFrame) -> dict:
        """Analyze volatility term structure"""
        
        if not all(col in df.columns for col in ['IV', 'DTE_calc']):
            return {}
        
        df_clean = df.dropna(subset=['IV', 'DTE_calc'])
        df_clean = df_clean[df_clean['DTE_calc'] > 0]  # Only future expirations
        
        if df_clean.empty:
            return {}
        
        term_structure = {
            'term_structure_points': [],
            'term_structure_slope': 0,
            'contango_backwardation': 'neutral',
            'term_structure_opportunities': []
        }
        
        # Group by DTE and calculate average IV
        dte_groups = df_clean.groupby('DTE_calc').agg({
            'IV': ['mean', 'std', 'count'],
            'TradeQuantity': 'sum'
        }).reset_index()
        
        dte_groups.columns = ['dte', 'iv_mean', 'iv_std', 'trade_count', 'volume']
        dte_groups = dte_groups[dte_groups['trade_count'] >= 3]  # Minimum trades for reliability
        
        for _, row in dte_groups.iterrows():
            term_structure['term_structure_points'].append({
                'dte': row['dte'],
                'iv': row['iv_mean'],
                'iv_std': row['iv_std'],
                'volume': row['volume'],
                'reliability': min(100, row['trade_count'] * 5)
            })
        
        # Calculate term structure slope
        if len(dte_groups) >= 3:
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    dte_groups['dte'], dte_groups['iv_mean']
                )
                term_structure['term_structure_slope'] = slope
                term_structure['slope_r_squared'] = r_value ** 2
                
                # Determine contango/backwardation
                if slope > 0.001:  # Positive slope
                    term_structure['contango_backwardation'] = 'contango'
                elif slope < -0.001:  # Negative slope
                    term_structure['contango_backwardation'] = 'backwardation'
                else:
                    term_structure['contango_backwardation'] = 'flat'
                    
            except:
                pass
        
        # Identify term structure trading opportunities
        term_structure['term_structure_opportunities'] = self._identify_term_structure_opportunities(dte_groups)
        
        return term_structure
    
    def _analyze_volatility_smile(self, df: pd.DataFrame) -> dict:
        """Analyze volatility smile patterns"""
        
        if not all(col in df.columns for col in ['IV', 'Strike_Price_calc', 'Underlying_Price']):
            return {}
        
        df_clean = df.dropna(subset=['IV', 'Strike_Price_calc', 'Underlying_Price'])
        
        if df_clean.empty:
            return {}
        
        # Calculate moneyness
        df_clean['moneyness'] = df_clean['Strike_Price_calc'] / df_clean['Underlying_Price']
        
        smile_analysis = {
            'smile_shape': 'unknown',
            'atm_iv': 0,
            'smile_asymmetry': 0,
            'smile_curvature': 0,
            'smile_points': []
        }
        
        # Group by moneyness buckets
        moneyness_buckets = pd.cut(df_clean['moneyness'], bins=10, labels=False)
        df_clean['moneyness_bucket'] = moneyness_buckets
        
        bucket_stats = df_clean.groupby('moneyness_bucket').agg({
            'moneyness': 'mean',
            'IV': ['mean', 'std', 'count'],
            'TradeQuantity': 'sum'
        }).reset_index()
        
        bucket_stats.columns = ['bucket', 'avg_moneyness', 'iv_mean', 'iv_std', 'trade_count', 'volume']
        bucket_stats = bucket_stats[bucket_stats['trade_count'] >= 2]
        
        # Store smile points
        for _, row in bucket_stats.iterrows():
            smile_analysis['smile_points'].append({
                'moneyness': row['avg_moneyness'],
                'iv': row['iv_mean'],
                'volume': row['volume']
            })
        
        # Find ATM IV
        atm_bucket = bucket_stats.iloc[(bucket_stats['avg_moneyness'] - 1.0).abs().argsort()[:1]]
        if not atm_bucket.empty:
            smile_analysis['atm_iv'] = atm_bucket['iv_mean'].iloc[0]
        
        # Analyze smile shape
        if len(bucket_stats) >= 5:
            # Calculate asymmetry (difference between put wing and call wing)
            otm_puts = bucket_stats[bucket_stats['avg_moneyness'] < 0.95]
            otm_calls = bucket_stats[bucket_stats['avg_moneyness'] > 1.05]
            
            if not otm_puts.empty and not otm_calls.empty:
                put_wing_iv = otm_puts['iv_mean'].mean()
                call_wing_iv = otm_calls['iv_mean'].mean()
                smile_analysis['smile_asymmetry'] = put_wing_iv - call_wing_iv
                
                # Determine smile shape
                if smile_analysis['smile_asymmetry'] > 0.05:
                    smile_analysis['smile_shape'] = 'put_skewed'
                elif smile_analysis['smile_asymmetry'] < -0.05:
                    smile_analysis['smile_shape'] = 'call_skewed'
                else:
                    smile_analysis['smile_shape'] = 'symmetric'
            
            # Calculate curvature using quadratic fit
            try:
                coeffs = np.polyfit(bucket_stats['avg_moneyness'], bucket_stats['iv_mean'], 2)
                smile_analysis['smile_curvature'] = coeffs[0]  # Second-order coefficient
            except:
                pass
        
        return smile_analysis
    
    def _calculate_iv_rank_percentile(self, df: pd.DataFrame) -> dict:
        """Calculate IV rank and percentile metrics"""
        
        if 'IV' not in df.columns:
            return {}
        
        iv_data = df['IV'].dropna()
        
        if iv_data.empty:
            return {}
        
        # Calculate current IV statistics
        current_iv = iv_data.mean()
        iv_range = iv_data.max() - iv_data.min()
        
        rank_percentile = {
            'current_avg_iv': current_iv,
            'iv_range': iv_range,
            'iv_rank': 0,  # Would need historical data
            'iv_percentile': 0,  # Would need historical data
            'iv_regime': 'unknown'
        }
        
        # Estimate IV regime based on current levels
        if current_iv > 0.5:  # 50% IV
            rank_percentile['iv_regime'] = 'high_vol'
        elif current_iv > 0.3:  # 30% IV
            rank_percentile['iv_regime'] = 'elevated_vol'
        elif current_iv > 0.15:  # 15% IV
            rank_percentile['iv_regime'] = 'normal_vol'
        else:
            rank_percentile['iv_regime'] = 'low_vol'
        
        # Calculate relative IV metrics within current session
        session_percentiles = {}
        for percentile in [10, 25, 50, 75, 90]:
            session_percentiles[f'session_p{percentile}'] = iv_data.quantile(percentile/100)
        
        rank_percentile['session_percentiles'] = session_percentiles
        
        return rank_percentile
    
    def _detect_volatility_clustering(self, df: pd.DataFrame) -> dict:
        """Detect volatility clustering patterns"""
        
        if not all(col in df.columns for col in ['IV', 'Time_dt']):
            return {}
        
        df_sorted = df.sort_values('Time_dt')
        iv_changes = df_sorted['IV'].diff().abs()
        
        clustering = {
            'volatility_clusters': [],
            'clustering_score': 0,
            'garch_effects': False
        }
        
        # Identify periods of high volatility changes
        high_vol_threshold = iv_changes.quantile(0.8)
        
        cluster_periods = []
        in_cluster = False
        cluster_start = None
        
        for i, (idx, change) in enumerate(iv_changes.items()):
            if pd.isna(change):
                continue
                
            if change > high_vol_threshold and not in_cluster:
                # Start of cluster
                in_cluster = True
                cluster_start = i
            elif change <= high_vol_threshold and in_cluster:
                # End of cluster
                in_cluster = False
                if i - cluster_start >= 3:  # At least 3 points
                    cluster_periods.append((cluster_start, i))
        
        # Analyze identified clusters
        for start_idx, end_idx in cluster_periods:
            cluster_data = df_sorted.iloc[start_idx:end_idx+1]
            
            clustering['volatility_clusters'].append({
                'start_time': cluster_data['Time_dt'].iloc[0],
                'end_time': cluster_data['Time_dt'].iloc[-1],
                'duration_minutes': (cluster_data['Time_dt'].iloc[-1] - cluster_data['Time_dt'].iloc[0]).total_seconds() / 60,
                'avg_iv_change': iv_changes.iloc[start_idx:end_idx+1].mean(),
                'max_iv_change': iv_changes.iloc[start_idx:end_idx+1].max(),
                'cluster_intensity': iv_changes.iloc[start_idx:end_idx+1].std()
            })
        
        # Calculate clustering score
        if len(iv_changes) > 0:
            clustering['clustering_score'] = len(cluster_periods) / len(iv_changes) * 100
        
        return clustering
    
    def _analyze_vol_mean_reversion(self, df: pd.DataFrame) -> dict:
        """Analyze volatility mean reversion patterns"""
        
        if 'IV' not in df.columns or len(df) < 10:
            return {}
        
        df_sorted = df.sort_values('Time_dt') if 'Time_dt' in df.columns else df
        iv_series = df_sorted['IV'].dropna()
        
        if len(iv_series) < 10:
            return {}
        
        mean_reversion = {
            'mean_reversion_speed': 0,
            'long_term_mean': iv_series.mean(),
            'current_deviation': 0,
            'reversion_signals': []
        }
        
        # Calculate deviations from mean
        long_term_mean = iv_series.mean()
        deviations = iv_series - long_term_mean
        
        # Estimate mean reversion using AR(1) model
        try:
            from scipy.stats import linregress
            
            # Lag the series
            iv_lag = iv_series.shift(1).dropna()
            iv_current = iv_series[1:]
            
            if len(iv_lag) > 5:
                slope, intercept, r_value, p_value, std_err = linregress(iv_lag, iv_current)
                
                # Mean reversion coefficient (1 - slope)
                mean_reversion['mean_reversion_speed'] = 1 - slope
                mean_reversion['r_squared'] = r_value ** 2
                mean_reversion['significance'] = p_value < 0.05
        except:
            pass
        
        # Current deviation from long-term mean
        if not iv_series.empty:
            current_iv = iv_series.iloc[-1]
            mean_reversion['current_deviation'] = (current_iv - long_term_mean) / long_term_mean
        
        # Identify potential reversion signals
        threshold = iv_series.std() * 1.5
        extreme_deviations = abs(deviations) > threshold
        
        for i, is_extreme in enumerate(extreme_deviations):
            if is_extreme:
                deviation_value = deviations.iloc[i]
                signal_type = 'oversold' if deviation_value < 0 else 'overbought'
                
                mean_reversion['reversion_signals'].append({
                    'timestamp': df_sorted.iloc[i].get('Time_dt', i),
                    'iv_level': iv_series.iloc[i],
                    'deviation': deviation_value,
                    'signal_type': signal_type,
                    'reversion_probability': min(95, abs(deviation_value) / threshold * 50 + 50)
                })
        
        return mean_reversion
    
    def _analyze_vol_spillovers(self, df: pd.DataFrame) -> dict:
        """Analyze volatility spillovers between options"""
        
        if 'StandardOptionSymbol' not in df.columns or df['StandardOptionSymbol'].nunique() < 2:
            return {}
        
        spillovers = {
            'cross_correlations': {},
            'lead_lag_relationships': {},
            'volatility_transmission': []
        }
        
        # Calculate IV correlations between different options
        iv_pivot = df.pivot_table(
            values='IV', 
            index='Time_dt' if 'Time_dt' in df.columns else df.index, 
            columns='StandardOptionSymbol',
            aggfunc='mean'
        )
        
        if iv_pivot.shape[1] >= 2:
            # Calculate correlation matrix
            corr_matrix = iv_pivot.corr()
            
            # Store significant correlations
            for i, symbol1 in enumerate(corr_matrix.columns):
                for j, symbol2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        correlation = corr_matrix.loc[symbol1, symbol2]
                        if abs(correlation) > 0.5:  # Significant correlation
                            spillovers['cross_correlations'][f"{symbol1}_{symbol2}"] = {
                                'correlation': correlation,
                                'relationship': 'positive' if correlation > 0 else 'negative'
                            }
        
        return spillovers
    
    def _detect_unusual_iv_activity(self, df: pd.DataFrame) -> dict:
        """Detect unusual implied volatility activity"""
        
        if 'IV' not in df.columns:
            return {}
        
        unusual_activity = {
            'iv_spikes': [],
            'iv_suppressions': [],
            'unusual_patterns': [],
            'statistical_anomalies': []
        }
        
        iv_data = df['IV'].dropna()
        
        if len(iv_data) < 10:
            return unusual_activity
        
        # Calculate Z-scores for IV
        iv_mean = iv_data.mean()
        iv_std = iv_data.std()
        
        if iv_std > 0:
            df['iv_zscore'] = (df['IV'] - iv_mean) / iv_std
            
            # Identify IV spikes (Z-score > 2)
            spikes = df[df['iv_zscore'] > 2]
            for _, spike in spikes.iterrows():
                unusual_activity['iv_spikes'].append({
                    'symbol': spike.get('StandardOptionSymbol', ''),
                    'iv_level': spike.get('IV', 0),
                    'z_score': spike.get('iv_zscore', 0),
                    'timestamp': spike.get('Time_dt', ''),
                    'volume': spike.get('TradeQuantity', 0)
                })
            
            # Identify IV suppressions (Z-score < -2)
            suppressions = df[df['iv_zscore'] < -2]
            for _, suppression in suppressions.iterrows():
                unusual_activity['iv_suppressions'].append({
                    'symbol': suppression.get('StandardOptionSymbol', ''),
                    'iv_level': suppression.get('IV', 0),
                    'z_score': suppression.get('iv_zscore', 0),
                    'timestamp': suppression.get('Time_dt', ''),
                    'volume': suppression.get('TradeQuantity', 0)
                })
        
        # Detect rapid IV changes
        if 'Time_dt' in df.columns:
            df_sorted = df.sort_values('Time_dt')
            iv_changes = df_sorted['IV'].diff().abs()
            rapid_changes = iv_changes > iv_changes.quantile(0.95)
            
            for idx in iv_changes[rapid_changes].index:
                if idx in df_sorted.index:
                    trade = df_sorted.loc[idx]
                    unusual_activity['unusual_patterns'].append({
                        'type': 'rapid_iv_change',
                        'symbol': trade.get('StandardOptionSymbol', ''),
                        'iv_change': iv_changes.loc[idx],
                        'timestamp': trade.get('Time_dt', ''),
                        'new_iv': trade.get('IV', 0)
                    })
        
        return unusual_activity
    
    def _identify_vol_opportunities(self, df: pd.DataFrame) -> dict:
        """Identify volatility trading opportunities"""
        
        opportunities = {
            'cheap_vol': [],
            'expensive_vol': [],
            'vol_arbitrage': [],
            'calendar_spreads': [],
            'dispersion_trades': []
        }
        
        if 'IV' not in df.columns:
            return opportunities
        
        # Calculate IV percentiles for each option
        for symbol in df['StandardOptionSymbol'].unique():
            symbol_data = df[df['StandardOptionSymbol'] == symbol]
            
            if len(symbol_data) < 5:
                continue
            
            avg_iv = symbol_data['IV'].mean()
            iv_percentile = (symbol_data['IV'] <= avg_iv).mean() * 100
            
            # Volume-weighted price
            if 'TradeQuantity' in symbol_data.columns and 'Trade_Price' in symbol_data.columns:
                total_volume = symbol_data['TradeQuantity'].sum()
                if total_volume > 0:
                    vwap = (symbol_data['Trade_Price'] * symbol_data['TradeQuantity']).sum() / total_volume
                else:
                    vwap = symbol_data['Trade_Price'].mean()
            else:
                vwap = symbol_data.get('Trade_Price', pd.Series([0])).mean()
            
            # Identify opportunities
            if iv_percentile < 20:  # Low IV percentile
                opportunities['cheap_vol'].append({
                    'symbol': symbol,
                    'current_iv': avg_iv,
                    'iv_percentile': iv_percentile,
                    'vwap': vwap,
                    'opportunity_type': 'buy_vol',
                    'confidence': 100 - iv_percentile
                })
            elif iv_percentile > 80:  # High IV percentile
                opportunities['expensive_vol'].append({
                    'symbol': symbol,
                    'current_iv': avg_iv,
                    'iv_percentile': iv_percentile,
                    'vwap': vwap,
                    'opportunity_type': 'sell_vol',
                    'confidence': iv_percentile
                })
        
        # Identify calendar spread opportunities
        opportunities['calendar_spreads'] = self._identify_calendar_opportunities(df)
        
        return opportunities
    
    def _identify_calendar_opportunities(self, df: pd.DataFrame) -> list[dict]:
        """Identify calendar spread opportunities based on term structure"""
        
        calendar_opps = []
        
        if not all(col in df.columns for col in ['IV', 'DTE_calc', 'Strike_Price_calc']):
            return calendar_opps
        
        # Group by strike
        for strike in df['Strike_Price_calc'].unique():
            strike_data = df[df['Strike_Price_calc'] == strike]
            
            if strike_data['DTE_calc'].nunique() < 2:
                continue
            
            # Calculate IV by DTE
            dte_iv = strike_data.groupby('DTE_calc')['IV'].mean().sort_index()
            
            if len(dte_iv) >= 2:
                # Look for inverted term structure (near > far)
                for i in range(len(dte_iv) - 1):
                    near_dte = dte_iv.index[i]
                    far_dte = dte_iv.index[i + 1]
                    near_iv = dte_iv.iloc[i]
                    far_iv = dte_iv.iloc[i + 1]
                    
                    iv_spread = near_iv - far_iv
                    
                    if iv_spread > 0.05:  # 5% IV difference
                        calendar_opps.append({
                            'strike': strike,
                            'near_dte': near_dte,
                            'far_dte': far_dte,
                            'near_iv': near_iv,
                            'far_iv': far_iv,
                            'iv_spread': iv_spread,
                            'trade_type': 'sell_near_buy_far',
                            'opportunity_score': iv_spread * 100
                        })
        
        return sorted(calendar_opps, key=lambda x: x['opportunity_score'], reverse=True)[:10]
    
    # Helper methods for complex calculations
    
    def _identify_iv_clusters(self, iv_data: pd.Series) -> list[dict]:
        """Identify clusters in IV data using simple binning"""
        
        clusters = []
        
        if len(iv_data) < 10:
            return clusters
        
        # Create bins
        n_bins = min(10, len(iv_data) // 3)
        bins = pd.cut(iv_data, bins=n_bins)
        bin_counts = bins.value_counts()
        
        # Identify significant clusters (bins with > 20% of data)
        threshold = len(iv_data) * 0.2
        
        for bin_range, count in bin_counts.items():
            if count >= threshold:
                clusters.append({
                    'iv_range': f"{bin_range.left:.3f} - {bin_range.right:.3f}",
                    'count': count,
                    'percentage': (count / len(iv_data)) * 100,
                    'center': (bin_range.left + bin_range.right) / 2
                })
        
        return clusters
    
    def _analyze_strike_iv_relationship(self, df: pd.DataFrame) -> dict:
        """Analyze relationship between strike and IV"""
        
        relationship = {}
        
        # Calculate correlation between strike and IV
        if len(df) >= 3:
            try:
                correlation = df['Strike_Price_calc'].corr(df['IV'])
                relationship['strike_iv_correlation'] = correlation
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df['Strike_Price_calc'], df['IV']
                )
                
                relationship['slope'] = slope
                relationship['r_squared'] = r_value ** 2
                relationship['significance'] = p_value < 0.05
                
            except:
                pass
        
        return relationship
    
    def _analyze_time_iv_relationship(self, df: pd.DataFrame) -> dict:
        """Analyze relationship between time to expiration and IV"""
        
        relationship = {}
        
        if 'DTE_calc' in df.columns and len(df) >= 3:
            try:
                correlation = df['DTE_calc'].corr(df['IV'])
                relationship['time_iv_correlation'] = correlation
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df['DTE_calc'], df['IV']
                )
                
                relationship['slope'] = slope
                relationship['r_squared'] = r_value ** 2
                relationship['significance'] = p_value < 0.05
                
            except:
                pass
        
        return relationship
    
    def _calculate_skew_metrics(self, options_data: pd.DataFrame, option_type: str) -> dict:
        """Calculate skew metrics for calls or puts"""
        
        skew_metrics = {
            'skew_slope': 0,
            'skew_curvature': 0,
            'wing_levels': {}
        }
        
        if len(options_data) < 3:
            return skew_metrics
        
        # Sort by moneyness
        sorted_data = options_data.sort_values('moneyness')
        
        # Calculate skew slope using linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                sorted_data['moneyness'], sorted_data['IV']
            )
            skew_metrics['skew_slope'] = slope
            skew_metrics['skew_r_squared'] = r_value ** 2
        except:
            pass
        
        # Identify wing levels
        if option_type == 'put':
            # For puts, look at downside (OTM puts)
            otm_puts = sorted_data[sorted_data['moneyness'] < 0.95]
            if not otm_puts.empty:
                skew_metrics['wing_levels']['10_delta'] = otm_puts['IV'].iloc[0] if len(otm_puts) > 0 else None
                skew_metrics['wing_levels']['25_delta'] = otm_puts['IV'].quantile(0.5) if len(otm_puts) > 2 else None
        else:
            # For calls, look at upside (OTM calls)
            otm_calls = sorted_data[sorted_data['moneyness'] > 1.05]
            if not otm_calls.empty:
                skew_metrics['wing_levels']['10_delta'] = otm_calls['IV'].iloc[-1] if len(otm_calls) > 0 else None
                skew_metrics['wing_levels']['25_delta'] = otm_calls['IV'].quantile(0.5) if len(otm_calls) > 2 else None
        
        return skew_metrics
    
    def _identify_skew_opportunities(self, df: pd.DataFrame) -> list[dict]:
        """Identify skew trading opportunities"""
        
        opportunities = []
        
        # Look for extreme skew situations
        if len(df) < 5:
            return opportunities
        
        # Group by expiration
        for expiry in df['Expiration_Date_calc'].unique():
            expiry_data = df[df['Expiration_Date_calc'] == expiry]
            
            if len(expiry_data) < 5:
                continue
            
            # Calculate skew slope
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    expiry_data['moneyness'], expiry_data['IV']
                )
                
                # Extreme negative skew (unusual)
                if slope < -0.5 and r_value ** 2 > 0.3:
                    opportunities.append({
                        'expiration': expiry,
                        'skew_type': 'extreme_negative',
                        'skew_slope': slope,
                        'trade_suggestion': 'Buy low strike puts, sell high strike puts',
                        'confidence': min(100, abs(slope) * 100)
                    })
                
                # Extreme positive skew (very unusual)
                elif slope > 0.3 and r_value ** 2 > 0.3:
                    opportunities.append({
                        'expiration': expiry,
                        'skew_type': 'extreme_positive',
                        'skew_slope': slope,
                        'trade_suggestion': 'Sell low strike puts, buy high strike puts',
                        'confidence': min(100, slope * 100)
                    })
                    
            except:
                pass
        
        return opportunities
    
    def _identify_term_structure_opportunities(self, dte_groups: pd.DataFrame) -> list[dict]:
        """Identify term structure trading opportunities"""
        
        opportunities = []
        
        if len(dte_groups) < 3:
            return opportunities
        
        # Look for inversions and steep curves
        dte_groups = dte_groups.sort_values('dte')
        
        for i in range(len(dte_groups) - 1):
            near_term = dte_groups.iloc[i]
            far_term = dte_groups.iloc[i + 1]
            
            iv_diff = near_term['iv_mean'] - far_term['iv_mean']
            time_diff = far_term['dte'] - near_term['dte']
            
            # Inversion: near > far
            if iv_diff > 0.05:  # 5% inversion
                opportunities.append({
                    'near_dte': near_term['dte'],
                    'far_dte': far_term['dte'],
                    'near_iv': near_term['iv_mean'],
                    'far_iv': far_term['iv_mean'],
                    'iv_difference': iv_diff,
                    'opportunity_type': 'calendar_spread',
                    'trade_suggestion': 'Sell near-term, buy far-term',
                    'expected_profit': iv_diff * 100
                })
        
        return opportunities
    
    def _calculate_vol_composite_scores(self, vol_analysis: dict) -> dict:
        """Calculate composite volatility scores"""
        
        scores = {
            'volatility_regime': 'unknown',
            'vol_attractiveness': 50,
            'trading_opportunity_score': 0,
            'risk_score': 50
        }
        
        # Determine volatility regime
        basic_stats = vol_analysis.get('basic_iv_stats', {})
        mean_iv = basic_stats.get('mean_iv', 0.2)
        
        if mean_iv > 0.5:
            scores['volatility_regime'] = 'high'
            scores['vol_attractiveness'] = 25  # High vol less attractive for buying
        elif mean_iv > 0.3:
            scores['volatility_regime'] = 'elevated'
            scores['vol_attractiveness'] = 40
        elif mean_iv > 0.15:
            scores['volatility_regime'] = 'normal'
            scores['vol_attractiveness'] = 60
        else:
            scores['volatility_regime'] = 'low'
            scores['vol_attractiveness'] = 85  # Low vol attractive for buying
        
        # Calculate trading opportunity score
        opportunities = vol_analysis.get('vol_trading_opportunities', {})
        cheap_vol_count = len(opportunities.get('cheap_vol', []))
        expensive_vol_count = len(opportunities.get('expensive_vol', []))
        calendar_count = len(opportunities.get('calendar_spreads', []))
        
        scores['trading_opportunity_score'] = min(100, (cheap_vol_count + expensive_vol_count + calendar_count) * 10)
        
        # Calculate risk score based on volatility clustering and regime
        clustering = vol_analysis.get('vol_clustering', {})
        clustering_score = clustering.get('clustering_score', 0)
        
        scores['risk_score'] = min(100, 50 + clustering_score + (mean_iv * 100))
        
        return scores