# analysis_modules/greek_flow_analyzer.py
"""
Analyzer for Greek flows and exposures
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime

class GreekFlowAnalyzer:
    """Analyzes option Greek flows and market maker positioning"""
    
    def __init__(self, spot_price: float):
        self.spot_price = spot_price
        self.greek_cache = {}
        
    def analyze_greek_flows(self, df: pd.DataFrame) -> dict:
        """Comprehensive Greek flow analysis"""
        analysis = {
            'delta_flow': self._analyze_delta_flow(df),
            'gamma_flow': self._analyze_gamma_flow(df),
            'vega_flow': self._analyze_vega_flow(df),
            'theta_flow': self._analyze_theta_flow(df),
            'aggregate_exposure': self._calculate_aggregate_exposure(df),
            'market_maker_position': self._estimate_mm_position(df),
            'greek_concentrations': self._find_greek_concentrations(df),
            'hedging_flows': self._identify_hedging_flows(df)
        }
        
        return analysis
    
    def _analyze_delta_flow(self, df: pd.DataFrame) -> dict:
        """Analyze delta flows and directional exposure"""
        delta_analysis = {
            'net_delta': 0,
            'buy_delta': 0,
            'sell_delta': 0,
            'delta_distribution': {},
            'directional_flow': []
        }
        
        if 'Delta' not in df.columns:
            return delta_analysis
        
        # Calculate net delta exposure
        df['signed_delta'] = df.apply(
            lambda row: row['Delta'] * row['TradeQuantity'] * 100 * 
                       (1 if 'Buy' in row['Aggressor'] else -1),
            axis=1
        )
        
        delta_analysis['net_delta'] = df['signed_delta'].sum()
        delta_analysis['buy_delta'] = df[df['Aggressor'].str.contains('Buy', na=False)]['signed_delta'].sum()
        delta_analysis['sell_delta'] = df[df['Aggressor'].str.contains('Sell', na=False)]['signed_delta'].sum()
        
        # Delta distribution by strike
        if 'Strike_Price_calc' in df.columns:
            delta_by_strike = df.groupby('Strike_Price_calc')['signed_delta'].sum()
            delta_analysis['delta_distribution'] = delta_by_strike.to_dict()
        
        # Identify significant directional flows
        large_delta_trades = df[df['signed_delta'].abs() > 1000]  # 1000 delta shares
        
        for _, trade in large_delta_trades.iterrows():
            delta_analysis['directional_flow'].append({
                'symbol': trade['StandardOptionSymbol'],
                'time': trade['Time'],
                'delta_shares': trade['signed_delta'],
                'notional_equivalent': abs(trade['signed_delta'] * self.spot_price),
                'direction': 'Long' if trade['signed_delta'] > 0 else 'Short'
            })
        
        return delta_analysis
    
    def _analyze_gamma_flow(self, df: pd.DataFrame) -> dict:
        """Analyze gamma flows and potential squeeze levels"""
        gamma_analysis = {
            'net_gamma': 0,
            'gamma_by_strike': {},
            'max_gamma_strike': None,
            'gamma_flip_points': [],
            'squeeze_potential': {}
        }
        
        if 'Gamma' not in df.columns:
            return gamma_analysis
        
        # Calculate gamma exposure
        df['gamma_dollars'] = df['Gamma'] * df['TradeQuantity'] * 100 * self.spot_price * self.spot_price / 100
        df['signed_gamma'] = df.apply(
            lambda row: row['gamma_dollars'] * (1 if 'Buy' in row['Aggressor'] else -1),
            axis=1
        )
        
        gamma_analysis['net_gamma'] = df['signed_gamma'].sum()
        
        # Gamma by strike
        if 'Strike_Price_calc' in df.columns:
            gamma_by_strike = df.groupby('Strike_Price_calc')['signed_gamma'].sum()
            gamma_analysis['gamma_by_strike'] = gamma_by_strike.to_dict()
            
            # Find max gamma strike
            if not gamma_by_strike.empty:
                max_gamma_strike = gamma_by_strike.abs().idxmax()
                gamma_analysis['max_gamma_strike'] = max_gamma_strike
                
                # Identify potential squeeze levels
                for strike, gamma in gamma_by_strike.items():
                    if abs(gamma) > gamma_by_strike.abs().quantile(0.9):
                        squeeze_direction = 'Up' if gamma > 0 else 'Down'
                        gamma_analysis['squeeze_potential'][strike] = {
                            'gamma_$': gamma,
                            'squeeze_direction': squeeze_direction,
                            'distance_from_spot': abs(strike - self.spot_price)
                        }
        
        # Identify gamma flip points (where net gamma changes sign)
        sorted_strikes = sorted(gamma_analysis['gamma_by_strike'].keys())
        for i in range(1, len(sorted_strikes)):
            prev_gamma = gamma_analysis['gamma_by_strike'][sorted_strikes[i-1]]
            curr_gamma = gamma_analysis['gamma_by_strike'][sorted_strikes[i]]
            
            if prev_gamma * curr_gamma < 0:  # Sign change
                flip_point = (sorted_strikes[i-1] + sorted_strikes[i]) / 2
                gamma_analysis['gamma_flip_points'].append(flip_point)
        
        return gamma_analysis
    
    def _analyze_vega_flow(self, df: pd.DataFrame) -> dict:
        """Analyze vega flows and volatility positioning"""
        vega_analysis = {
            'net_vega': 0,
            'vega_by_expiry': {},
            'volatility_trades': [],
            'iv_vs_vega': {}
        }
        
        if 'Vega' not in df.columns:
            return vega_analysis
        
        # Calculate vega exposure
        df['vega_dollars'] = df['Vega'] * df['TradeQuantity'] * 100
        df['signed_vega'] = df.apply(
            lambda row: row['vega_dollars'] * (1 if 'Buy' in row['Aggressor'] else -1),
            axis=1
        )
        
        vega_analysis['net_vega'] = df['signed_vega'].sum()
        
        # Vega by expiry
        if 'Expiration_Date_calc' in df.columns:
            vega_by_expiry = df.groupby('Expiration_Date_calc')['signed_vega'].sum()
            vega_analysis['vega_by_expiry'] = {
                str(k): v for k, v in vega_by_expiry.to_dict().items()
            }
        
        # Identify significant volatility trades
        large_vega_trades = df[df['vega_dollars'].abs() > 10000]  # $10k vega
        
        for _, trade in large_vega_trades.iterrows():
            vega_analysis['volatility_trades'].append({
                'symbol': trade['StandardOptionSymbol'],
                'time': trade['Time'],
                'vega_dollars': trade['signed_vega'],
                'iv': trade.get('IV', 0),
                'position': 'Long Vol' if trade['signed_vega'] > 0 else 'Short Vol'
            })
        
        # Analyze IV vs Vega relationship
        if 'IV' in df.columns:
            # Group by IV buckets
            df['iv_bucket'] = pd.cut(df['IV'], bins=[0, 0.2, 0.3, 0.4, 0.5, 1.0])
            iv_vega_relationship = df.groupby('iv_bucket')['signed_vega'].sum()
            vega_analysis['iv_vs_vega'] = {
                str(k): v for k, v in iv_vega_relationship.to_dict().items()
            }
        
        return vega_analysis
    
    def _analyze_theta_flow(self, df: pd.DataFrame) -> dict:
        """Analyze theta flows and time decay positioning"""
        theta_analysis = {
            'net_theta': 0,
            'theta_by_dte': {},
            'theta_capture_trades': [],
            'weekend_theta': 0
        }
        
        if 'Theta' not in df.columns:
            return theta_analysis
        
        # Calculate theta exposure
        df['theta_dollars'] = df['Theta'] * df['TradeQuantity'] * 100
        df['signed_theta'] = df.apply(
            lambda row: row['theta_dollars'] * (1 if 'Buy' in row['Aggressor'] else -1),
            axis=1
        )
        
        theta_analysis['net_theta'] = df['signed_theta'].sum()
        
        # Theta by DTE
        if 'DTE_calc' in df.columns:
            df['dte_bucket'] = pd.cut(df['DTE_calc'], bins=[0, 7, 14, 30, 60, 365])
            theta_by_dte = df.groupby('dte_bucket')['signed_theta'].sum()
            theta_analysis['theta_by_dte'] = {
                str(k): v for k, v in theta_by_dte.to_dict().items()
            }
        
        # Identify theta capture trades (selling short-dated options)
        short_dated_sells = df[
            (df['DTE_calc'] <= 7) & 
            (df['Aggressor'].str.contains('Sell', na=False)) &
            (df['theta_dollars'].abs() > 100)
        ]
        
        for _, trade in short_dated_sells.iterrows():
            theta_analysis['theta_capture_trades'].append({
                'symbol': trade['StandardOptionSymbol'],
                'time': trade['Time'],
                'theta_daily': trade['theta_dollars'],
                'dte': trade.get('DTE_calc', 0),
                'premium_collected': trade['NotionalValue']
            })
        
        # Calculate weekend theta (Friday positions)
        current_day = datetime.now().weekday()
        if current_day == 4:  # Friday
            # Continuing greek_flow_analyzer.py
            theta_analysis['weekend_theta'] = df['signed_theta'].sum() * 2.5  # Friday + weekend
        
        return theta_analysis
    
    def _calculate_aggregate_exposure(self, df: pd.DataFrame) -> dict:
        """Calculate aggregate Greek exposures"""
        exposure = {
            'total_delta_$': 0,
            'total_gamma_$': 0,
            'total_vega_$': 0,
            'total_theta_$': 0,
            'direction': 'Neutral',
            'volatility_stance': 'Neutral',
            'risk_metrics': {}
        }
        
        # Sum all Greek exposures
        if 'signed_delta' in df.columns:
            exposure['total_delta_$'] = df['signed_delta'].sum() * self.spot_price
        
        if 'signed_gamma' in df.columns:
            exposure['total_gamma_$'] = df['signed_gamma'].sum()
        
        if 'signed_vega' in df.columns:
            exposure['total_vega_$'] = df['signed_vega'].sum()
        
        if 'signed_theta' in df.columns:
            exposure['total_theta_$'] = df['signed_theta'].sum()
        
        # Determine directional stance
        if abs(exposure['total_delta_$']) > 100000:  # $100k delta
            exposure['direction'] = 'Bullish' if exposure['total_delta_$'] > 0 else 'Bearish'
        
        # Determine volatility stance
        if abs(exposure['total_vega_$']) > 50000:  # $50k vega
            exposure['volatility_stance'] = 'Long Vol' if exposure['total_vega_$'] > 0 else 'Short Vol'
        
        # Calculate risk metrics
        if exposure['total_delta_$'] != 0:
            exposure['risk_metrics']['1pct_move_pnl'] = (
                exposure['total_delta_$'] * 0.01 +
                exposure['total_gamma_$'] * 0.01 * 0.01 * 0.5
            )
        
        if exposure['total_vega_$'] != 0:
            exposure['risk_metrics']['1vol_point_pnl'] = exposure['total_vega_$']
        
        if exposure['total_theta_$'] != 0:
            exposure['risk_metrics']['daily_decay'] = exposure['total_theta_$']
        
        return exposure
    
    def _estimate_mm_position(self, df: pd.DataFrame) -> dict:
        """Estimate market maker positioning"""
        mm_position = {
            'estimated_position': 'Neutral',
            'hedging_pressure': 0,
            'gamma_imbalance': 0,
            'pinning_strikes': [],
            'hedging_flows_expected': []
        }
        
        # Calculate customer order flow
        customer_delta = df[df['Aggressor'].str.contains('Buy|Sell', na=False)]['signed_delta'].sum() if 'signed_delta' in df.columns else 0
        customer_gamma = df[df['Aggressor'].str.contains('Buy|Sell', na=False)]['signed_gamma'].sum() if 'signed_gamma' in df.columns else 0
        
        # MM position is opposite of customer flow
        mm_delta = -customer_delta
        mm_gamma = -customer_gamma
        
        # Determine MM position
        if abs(mm_delta) > 5000:  # 5000 delta shares
            mm_position['estimated_position'] = 'Short' if mm_delta < 0 else 'Long'
        
        # Calculate hedging pressure
        if mm_gamma != 0:
            # Negative gamma means MM needs to buy on up moves, sell on down moves
            mm_position['hedging_pressure'] = -mm_gamma
            mm_position['gamma_imbalance'] = mm_gamma
        
        # Identify potential pinning strikes
        if 'Strike_Price_calc' in df.columns and 'signed_gamma' in df.columns:
            gamma_by_strike = df.groupby('Strike_Price_calc')['signed_gamma'].sum()
            
            # Large gamma concentrations suggest pinning
            gamma_threshold = gamma_by_strike.abs().quantile(0.8)
            for strike, gamma in gamma_by_strike.items():
                if abs(gamma) > gamma_threshold:
                    mm_position['pinning_strikes'].append({
                        'strike': strike,
                        'gamma_exposure': gamma,
                        'pin_strength': 'Strong' if abs(gamma) > gamma_threshold * 1.5 else 'Moderate'
                    })
        
        # Expected hedging flows
        if mm_gamma < -1000:  # Short gamma
            mm_position['hedging_flows_expected'].append({
                'condition': 'On rally',
                'action': 'MM buying',
                'magnitude': abs(mm_gamma)
            })
            mm_position['hedging_flows_expected'].append({
                'condition': 'On decline',
                'action': 'MM selling',
                'magnitude': abs(mm_gamma)
            })
        elif mm_gamma > 1000:  # Long gamma
            mm_position['hedging_flows_expected'].append({
                'condition': 'On rally',
                'action': 'MM selling',
                'magnitude': abs(mm_gamma)
            })
            mm_position['hedging_flows_expected'].append({
                'condition': 'On decline',
                'action': 'MM buying',
                'magnitude': abs(mm_gamma)
            })
        
        return mm_position
    
    def _find_greek_concentrations(self, df: pd.DataFrame) -> dict:
        """Find concentrations of Greek exposures"""
        concentrations = {
            'strike_concentrations': {},
            'expiry_concentrations': {},
            'risk_concentrations': []
        }
        
        # Strike concentrations
        if 'Strike_Price_calc' in df.columns:
            for greek in ['signed_delta', 'signed_gamma', 'signed_vega']:
                if greek in df.columns:
                    greek_by_strike = df.groupby('Strike_Price_calc')[greek].sum()
                    total_greek = greek_by_strike.abs().sum()
                    
                    for strike, value in greek_by_strike.items():
                        concentration = abs(value) / total_greek if total_greek > 0 else 0
                        if concentration > 0.2:  # 20% concentration
                            if strike not in concentrations['strike_concentrations']:
                                concentrations['strike_concentrations'][strike] = {}
                            concentrations['strike_concentrations'][strike][greek] = {
                                'value': value,
                                'concentration': concentration
                            }
        
        # Expiry concentrations
        if 'Expiration_Date_calc' in df.columns:
            for greek in ['signed_vega', 'signed_theta']:
                if greek in df.columns:
                    greek_by_expiry = df.groupby('Expiration_Date_calc')[greek].sum()
                    total_greek = greek_by_expiry.abs().sum()
                    
                    for expiry, value in greek_by_expiry.items():
                        concentration = abs(value) / total_greek if total_greek > 0 else 0
                        if concentration > 0.3:  # 30% concentration
                            if str(expiry) not in concentrations['expiry_concentrations']:
                                concentrations['expiry_concentrations'][str(expiry)] = {}
                            concentrations['expiry_concentrations'][str(expiry)][greek] = {
                                'value': value,
                                'concentration': concentration
                            }
        
        # Identify risk concentrations
        for strike, greeks in concentrations['strike_concentrations'].items():
            if len(greeks) >= 2:  # Multiple Greeks concentrated at same strike
                concentrations['risk_concentrations'].append({
                    'type': 'Strike Concentration',
                    'strike': strike,
                    'greeks': list(greeks.keys()),
                    'risk_level': 'High' if len(greeks) >= 3 else 'Moderate'
                })
        
        return concentrations
    
    def _identify_hedging_flows(self, df: pd.DataFrame) -> list[dict]:
        """Identify potential hedging flows"""
        hedging_flows = []
        
        # Pattern 1: Delta hedging (opposite direction trades in quick succession)
        if 'signed_delta' in df.columns:
            df_sorted = df.sort_values('Time_dt')
            
            for i in range(1, len(df_sorted)):
                prev_trade = df_sorted.iloc[i-1]
                curr_trade = df_sorted.iloc[i]
                
                time_diff = (curr_trade['Time_dt'] - prev_trade['Time_dt']).total_seconds()
                
                if time_diff < 60:  # Within 1 minute
                    # Check for opposite delta
                    if prev_trade['signed_delta'] * curr_trade['signed_delta'] < 0:
                        delta_offset = abs(prev_trade['signed_delta'] + curr_trade['signed_delta'])
                        if delta_offset < min(abs(prev_trade['signed_delta']), abs(curr_trade['signed_delta'])) * 0.2:
                            hedging_flows.append({
                                'type': 'Delta Hedge',
                                'time': curr_trade['Time'],
                                'trades': [
                                    {'symbol': prev_trade['StandardOptionSymbol'], 'delta': prev_trade['signed_delta']},
                                    {'symbol': curr_trade['StandardOptionSymbol'], 'delta': curr_trade['signed_delta']}
                                ],
                                'net_delta_remaining': delta_offset
                            })
        
        # Pattern 2: Gamma hedging (ATM straddle/strangle trades)
        atm_strikes = df[df['Moneyness_calc'] == 'ATM']['Strike_Price_calc'].unique() if 'Moneyness_calc' in df.columns else []
        
        for strike in atm_strikes:
            strike_trades = df[df['Strike_Price_calc'] == strike]
            
            calls = strike_trades[strike_trades['Option_Type_calc'] == 'Call']
            puts = strike_trades[strike_trades['Option_Type_calc'] == 'Put']
            
            if not calls.empty and not puts.empty:
                # Check if traded close in time
                call_times = pd.to_datetime(calls['Time_dt'])
                put_times = pd.to_datetime(puts['Time_dt'])
                
                for call_time in call_times:
                    for put_time in put_times:
                        if abs((call_time - put_time).total_seconds()) < 300:  # 5 minutes
                            hedging_flows.append({
                                'type': 'Gamma Hedge (Straddle)',
                                'strike': strike,
                                'time': min(call_time, put_time),
                                'instruments': ['Call', 'Put']
                            })
                            break
        
        return hedging_flows

def calculate_portfolio_greeks(positions: list[dict], spot_price: float) -> dict:
    """Calculate portfolio-level Greeks"""
    portfolio_greeks = {
        'total_delta': 0,
        'total_gamma': 0,
        'total_vega': 0,
        'total_theta': 0,
        'delta_dollars': 0,
        'gamma_dollars': 0,
        'gamma_pct': 0
    }
    
    for position in positions:
        quantity = position.get('quantity', 0)
        delta = position.get('delta', 0)
        gamma = position.get('gamma', 0)
        vega = position.get('vega', 0)
        theta = position.get('theta', 0)
        
        portfolio_greeks['total_delta'] += delta * quantity * 100
        portfolio_greeks['total_gamma'] += gamma * quantity * 100
        portfolio_greeks['total_vega'] += vega * quantity * 100
        portfolio_greeks['total_theta'] += theta * quantity * 100
    
    # Calculate dollar exposures
    portfolio_greeks['delta_dollars'] = portfolio_greeks['total_delta'] * spot_price
    portfolio_greeks['gamma_dollars'] = portfolio_greeks['total_gamma'] * spot_price * spot_price / 100
    
    if portfolio_greeks['total_delta'] != 0:
        portfolio_greeks['gamma_pct'] = portfolio_greeks['total_gamma'] / portfolio_greeks['total_delta'] * spot_price
    
    return portfolio_greeks