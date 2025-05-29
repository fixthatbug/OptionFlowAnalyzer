# greek_flow_analyzer.py
"""
Greek flow analysis for options trading data
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime

class GreekFlowAnalyzer:
    """Analyzes option Greek flows and market maker positioning"""
    
    def __init__(self):
        self.greek_cache = {}
        
    def analyze_greek_exposures(self, df: pd.DataFrame) -> dict:
        """Comprehensive Greek exposure analysis"""
        
        if df.empty:
            return self._empty_greek_results()
        
        analysis = {
            'delta_analysis': self._analyze_delta_exposure(df),
            'gamma_analysis': self._analyze_gamma_exposure(df),
            'theta_analysis': self._analyze_theta_exposure(df),
            'vega_analysis': self._analyze_vega_exposure(df),
            'portfolio_greeks': self._calculate_portfolio_greeks(df),
            'hedging_analysis': self._analyze_hedging_patterns(df),
            'market_maker_positioning': self._estimate_mm_positioning(df),
            'risk_metrics': self._calculate_risk_metrics(df)
        }
        
        return analysis
    
    def _empty_greek_results(self) -> dict:
        """Return empty Greek analysis structure"""
        return {
            'delta_analysis': {},
            'gamma_analysis': {},
            'theta_analysis': {},
            'vega_analysis': {},
            'portfolio_greeks': {},
            'hedging_analysis': {},
            'market_maker_positioning': {},
            'risk_metrics': {}
        }
    
    def _analyze_delta_exposure(self, df: pd.DataFrame) -> dict:
        """Analyze delta exposure and directional risk"""
        
        delta_analysis = {
            'total_delta_exposure': 0,
            'net_delta_shares': 0,
            'directional_bias': 'Neutral',
            'delta_concentration': {},
            'hedging_needs': []
        }
        
        if 'Delta' not in df.columns or 'TradeQuantity' not in df.columns:
            return delta_analysis
        
        # Calculate delta exposure
        df_clean = df.dropna(subset=['Delta', 'TradeQuantity'])
        if df_clean.empty:
            return delta_analysis
        
        # Calculate signed delta (positive for long, negative for short)
        if 'Aggressor' in df_clean.columns:
            df_clean['signed_delta'] = df_clean.apply(
                lambda row: row['Delta'] * row['TradeQuantity'] * 100 * 
                           (1 if 'Buy' in str(row['Aggressor']) else -1),
                axis=1
            )
        else:
            # If no aggressor info, assume all are buys
            df_clean['signed_delta'] = df_clean['Delta'] * df_clean['TradeQuantity'] * 100
        
        # Total delta exposure
        total_delta = df_clean['signed_delta'].sum()
        delta_analysis['total_delta_exposure'] = total_delta
        delta_analysis['net_delta_shares'] = total_delta
        
        # Determine directional bias
        if abs(total_delta) > 1000:  # Significant exposure
            if total_delta > 0:
                delta_analysis['directional_bias'] = 'Bullish'
            else:
                delta_analysis['directional_bias'] = 'Bearish'
        
        # Delta concentration by strike
        if 'Strike_Price_calc' in df_clean.columns:
            delta_by_strike = df_clean.groupby('Strike_Price_calc')['signed_delta'].sum()
            delta_analysis['delta_concentration'] = delta_by_strike.to_dict()
        
        # Identify hedging needs
        if abs(total_delta) > 5000:  # Large exposure
            hedge_direction = 'Sell' if total_delta > 0 else 'Buy'
            delta_analysis['hedging_needs'].append({
                'action': f'{hedge_direction} {abs(total_delta):.0f} shares of underlying',
                'reason': f'Hedge {delta_analysis["directional_bias"].lower()} delta exposure'
            })
        
        return delta_analysis
    
    def _analyze_gamma_exposure(self, df: pd.DataFrame) -> dict:
        """Analyze gamma exposure and convexity risk"""
        
        gamma_analysis = {
            'total_gamma_exposure': 0,
            'gamma_dollars': 0,
            'convexity_risk': 'Low',
            'gamma_hotspots': [],
            'hedging_frequency': 'Normal'
        }
        
        if 'Gamma' not in df.columns or 'TradeQuantity' not in df.columns:
            return gamma_analysis
        
        df_clean = df.dropna(subset=['Gamma', 'TradeQuantity'])
        if df_clean.empty:
            return gamma_analysis
        
        # Get underlying price for gamma dollar calculation
        underlying_price = df_clean.get('Underlying_Price', pd.Series([100])).iloc[0] if not df_clean.empty else 100
        
        # Calculate gamma exposure
        if 'Aggressor' in df_clean.columns:
            df_clean['signed_gamma'] = df_clean.apply(
                lambda row: row['Gamma'] * row['TradeQuantity'] * 100 * 
                           (1 if 'Buy' in str(row['Aggressor']) else -1),
                axis=1
            )
        else:
            df_clean['signed_gamma'] = df_clean['Gamma'] * df_clean['TradeQuantity'] * 100
        
        # Gamma in dollar terms
        df_clean['gamma_dollars'] = df_clean['signed_gamma'] * underlying_price * underlying_price / 100
        
        total_gamma = df_clean['signed_gamma'].sum()
        total_gamma_dollars = df_clean['gamma_dollars'].sum()
        
        gamma_analysis['total_gamma_exposure'] = total_gamma
        gamma_analysis['gamma_dollars'] = total_gamma_dollars
        
        # Assess convexity risk
        if abs(total_gamma_dollars) > 100000:  # $100k gamma exposure
            gamma_analysis['convexity_risk'] = 'High'
            gamma_analysis['hedging_frequency'] = 'Frequent'
        elif abs(total_gamma_dollars) > 50000:
            gamma_analysis['convexity_risk'] = 'Moderate'
            gamma_analysis['hedging_frequency'] = 'Regular'
        
        # Identify gamma hotspots
        if 'Strike_Price_calc' in df_clean.columns:
            gamma_by_strike = df_clean.groupby('Strike_Price_calc')['gamma_dollars'].sum()
            
            # Find strikes with significant gamma
            for strike, gamma_dollars in gamma_by_strike.items():
                if abs(gamma_dollars) > 25000:  # $25k threshold
                    gamma_analysis['gamma_hotspots'].append({
                        'strike': strike,
                        'gamma_dollars': gamma_dollars,
                        'distance_from_spot': abs(strike - underlying_price),
                        'potential_hedging': 'High' if abs(gamma_dollars) > 50000 else 'Moderate'
                    })
        
        return gamma_analysis
    
    def _analyze_theta_exposure(self, df: pd.DataFrame) -> dict:
        """Analyze theta exposure and time decay"""
        
        theta_analysis = {
            'total_theta_exposure': 0,
            'daily_decay': 0,
            'theta_strategy': 'Neutral',
            'weekend_risk': 0,
            'expiration_concentrations': {}
        }
        
        if 'Theta' not in df.columns or 'TradeQuantity' not in df.columns:
            return theta_analysis
        
        df_clean = df.dropna(subset=['Theta', 'TradeQuantity'])
        if df_clean.empty:
            return theta_analysis
        
        # Calculate theta exposure
        if 'Aggressor' in df_clean.columns:
            df_clean['signed_theta'] = df_clean.apply(
                lambda row: row['Theta'] * row['TradeQuantity'] * 100 * 
                           (1 if 'Buy' in str(row['Aggressor']) else -1),
                axis=1
            )
        else:
            df_clean['signed_theta'] = df_clean['Theta'] * df_clean['TradeQuantity'] * 100
        
        total_theta = df_clean['signed_theta'].sum()
        theta_analysis['total_theta_exposure'] = total_theta
        theta_analysis['daily_decay'] = total_theta
        
        # Determine theta strategy
        if total_theta < -1000:  # Losing money to time decay
            theta_analysis['theta_strategy'] = 'Long Options (Theta Negative)'
        elif total_theta > 1000:  # Benefiting from time decay
            theta_analysis['theta_strategy'] = 'Short Options (Theta Positive)'
        
        # Weekend risk (theta accelerates over weekends)
        current_day = datetime.now().weekday()
        if current_day == 4:  # Friday
            theta_analysis['weekend_risk'] = total_theta * 2.5  # Approximate weekend theta
        
        # Theta by expiration
        if 'Expiration_Date_calc' in df_clean.columns:
            theta_by_exp = df_clean.groupby('Expiration_Date_calc')['signed_theta'].sum()
            theta_analysis['expiration_concentrations'] = {
                str(exp): theta for exp, theta in theta_by_exp.items()
            }
        
        return theta_analysis
    
    def _analyze_vega_exposure(self, df: pd.DataFrame) -> dict:
        """Analyze vega exposure and volatility risk"""
        
        vega_analysis = {
            'total_vega_exposure': 0,
            'vol_sensitivity': 0,
            'volatility_bias': 'Neutral',
            'vol_risk_assessment': 'Low',
            'iv_positioning': {}
        }
        
        if 'Vega' not in df.columns or 'TradeQuantity' not in df.columns:
            return vega_analysis
        
        df_clean = df.dropna(subset=['Vega', 'TradeQuantity'])
        if df_clean.empty:
            return vega_analysis
        
        # Calculate vega exposure
        if 'Aggressor' in df_clean.columns:
            df_clean['signed_vega'] = df_clean.apply(
                lambda row: row['Vega'] * row['TradeQuantity'] * 100 * 
                           (1 if 'Buy' in str(row['Aggressor']) else -1),
                axis=1
            )
        else:
            df_clean['signed_vega'] = df_clean['Vega'] * df_clean['TradeQuantity'] * 100
        
        total_vega = df_clean['signed_vega'].sum()
        vega_analysis['total_vega_exposure'] = total_vega
        vega_analysis['vol_sensitivity'] = total_vega
        
        # Determine volatility bias
        if total_vega > 10000:  # Significant long vol exposure
            vega_analysis['volatility_bias'] = 'Long Volatility'
            vega_analysis['vol_risk_assessment'] = 'High' if total_vega > 50000 else 'Moderate'
        elif total_vega < -10000:  # Significant short vol exposure
            vega_analysis['volatility_bias'] = 'Short Volatility'
            vega_analysis['vol_risk_assessment'] = 'High' if total_vega < -50000 else 'Moderate'
        
        # Analyze IV positioning
        if 'IV' in df_clean.columns:
            # Group by IV levels
            df_clean['iv_bucket'] = pd.cut(df_clean['IV'], bins=[0, 0.2, 0.4, 0.6, 1.0], 
                                          labels=['Low', 'Medium', 'High', 'Very High'])
            
            iv_positioning = df_clean.groupby('iv_bucket')['signed_vega'].sum()
            vega_analysis['iv_positioning'] = iv_positioning.to_dict()
        
        return vega_analysis
    
    def _calculate_portfolio_greeks(self, df: pd.DataFrame) -> dict:
        """Calculate overall portfolio Greeks"""
        
        portfolio = {
            'net_delta': 0,
            'net_gamma': 0,
            'net_theta': 0,
            'net_vega': 0,
            'total_positions': 0,
            'risk_summary': 'Low Risk'
        }
        
        if df.empty:
            return portfolio
        
        # Calculate net Greeks
        greek_columns = ['Delta', 'Gamma', 'Theta', 'Vega']
        available_greeks = [col for col in greek_columns if col in df.columns]
        
        if not available_greeks or 'TradeQuantity' not in df.columns:
            return portfolio
        
        df_clean = df.dropna(subset=available_greeks + ['TradeQuantity'])
        
        if df_clean.empty:
            return portfolio
        
        # Calculate position-weighted Greeks
        for greek in available_greeks:
            if 'Aggressor' in df_clean.columns:
                signed_greek = df_clean.apply(
                    lambda row: row[greek] * row['TradeQuantity'] * 100 * 
                               (1 if 'Buy' in str(row['Aggressor']) else -1),
                    axis=1
                ).sum()
            else:
                signed_greek = (df_clean[greek] * df_clean['TradeQuantity'] * 100).sum()
            
            portfolio[f'net_{greek.lower()}'] = signed_greek
        
        portfolio['total_positions'] = df_clean['TradeQuantity'].sum()
        
        # Assess overall risk
        risk_factors = []
        
        if abs(portfolio['net_delta']) > 5000:
            risk_factors.append('High Delta')
        if abs(portfolio['net_gamma']) > 1000:
            risk_factors.append('High Gamma')
        if abs(portfolio['net_vega']) > 20000:
            risk_factors.append('High Vega')
        
        if len(risk_factors) >= 2:
            portfolio['risk_summary'] = 'High Risk'
        elif len(risk_factors) == 1:
            portfolio['risk_summary'] = 'Moderate Risk'
        
        return portfolio
    
    def _analyze_hedging_patterns(self, df: pd.DataFrame) -> dict:
        """Analyze potential hedging patterns"""
        
        hedging = {
            'potential_hedges': [],
            'hedge_ratio': 0,
            'hedge_effectiveness': 'Unknown',
            'hedging_frequency': 'Low'
        }
        
        if df.empty or len(df) < 2:
            return hedging
        
        # Look for opposite direction trades in quick succession
        df_sorted = df.sort_values('Time_dt') if 'Time_dt' in df.columns else df
        
        hedge_count = 0
        for i in range(1, len(df_sorted)):
            prev_trade = df_sorted.iloc[i-1]
            curr_trade = df_sorted.iloc[i]
            
            # Check if trades are close in time
            if 'Time_dt' in df_sorted.columns:
                time_diff = (curr_trade['Time_dt'] - prev_trade['Time_dt']).total_seconds()
                if time_diff > 300:  # More than 5 minutes apart
                    continue
            
            # Check for opposite directions
            if ('Aggressor' in df_sorted.columns and 
                'Buy' in str(prev_trade.get('Aggressor', '')) and 
                'Sell' in str(curr_trade.get('Aggressor', ''))):
                
                hedging['potential_hedges'].append({
                    'trade_1': {
                        'symbol': prev_trade.get('StandardOptionSymbol', ''),
                        'quantity': prev_trade.get('TradeQuantity', 0),
                        'side': 'Buy'
                    },
                    'trade_2': {
                        'symbol': curr_trade.get('StandardOptionSymbol', ''),
                        'quantity': curr_trade.get('TradeQuantity', 0),
                        'side': 'Sell'
                    },
                    'time_gap_seconds': time_diff if 'Time_dt' in df_sorted.columns else 0
                })
                hedge_count += 1
        
        # Calculate hedging frequency
        if len(df) > 0:
            hedge_ratio = hedge_count / len(df)
            hedging['hedge_ratio'] = hedge_ratio
            
            if hedge_ratio > 0.3:
                hedging['hedging_frequency'] = 'High'
            elif hedge_ratio > 0.1:
                hedging['hedging_frequency'] = 'Moderate'
        
        return hedging
    
    def _estimate_mm_positioning(self, df: pd.DataFrame) -> dict:
        """Estimate market maker positioning"""
        
        mm_analysis = {
            'estimated_mm_position': 'Neutral',
            'inventory_pressure': 0,
            'hedging_needs': [],
            'gamma_risk': 'Low',
            'pinning_levels': []
        }
        
        if df.empty:
            return mm_analysis
        
        # Estimate customer flow (MM takes opposite side)
        if 'Aggressor' in df.columns and 'TradeQuantity' in df.columns:
            buy_volume = df[df['Aggressor'].str.contains('Buy', na=False)]['TradeQuantity'].sum()
            sell_volume = df[df['Aggressor'].str.contains('Sell', na=False)]['TradeQuantity'].sum()
            
            net_customer_flow = buy_volume - sell_volume
            
            # MM position is opposite to customer flow
            if net_customer_flow > 100:
                mm_analysis['estimated_mm_position'] = 'Short Options'
                mm_analysis['inventory_pressure'] = -net_customer_flow
            elif net_customer_flow < -100:
                mm_analysis['estimated_mm_position'] = 'Long Options'
                mm_analysis['inventory_pressure'] = -net_customer_flow
        
        # Identify potential pinning levels
        if 'Strike_Price_calc' in df.columns and 'TradeQuantity' in df.columns:
            volume_by_strike = df.groupby('Strike_Price_calc')['TradeQuantity'].sum()
            high_volume_strikes = volume_by_strike[volume_by_strike > volume_by_strike.quantile(0.8)]
            
            for strike, volume in high_volume_strikes.items():
                mm_analysis['pinning_levels'].append({
                    'strike': strike,
                    'volume': volume,
                    'pinning_strength': 'Strong' if volume > volume_by_strike.quantile(0.9) else 'Moderate'
                })
        
        return mm_analysis
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate risk metrics for the Greek exposures"""
        
        risk_metrics = {
            'var_1_day': 0,
            'var_1_week': 0,
            'max_loss_1pct_move': 0,
            'theta_decay_daily': 0,
            'vol_risk_1pt': 0,
            'overall_risk_score': 0
        }
        
        if df.empty:
            return risk_metrics
        
        # Get portfolio Greeks from previous calculation
        portfolio = self._calculate_portfolio_greeks(df)
        
        # Calculate risk scenarios
        net_delta = portfolio.get('net_delta', 0)
        net_gamma = portfolio.get('net_gamma', 0)
        net_theta = portfolio.get('net_theta', 0)
        net_vega = portfolio.get('net_vega', 0)
        
        # 1% underlying move scenario
        if net_delta != 0 or net_gamma != 0:
            risk_metrics['max_loss_1pct_move'] = abs(net_delta * 0.01 + 0.5 * net_gamma * 0.01 * 0.01)
        
        # Daily theta decay
        risk_metrics['theta_decay_daily'] = abs(net_theta)
        
        # 1 vol point risk
        risk_metrics['vol_risk_1pt'] = abs(net_vega * 0.01)
        
        # Overall risk score (0-100)
        risk_components = [
            min(20, abs(net_delta) / 1000),
            min(20, abs(net_gamma) / 100),
            min(20, abs(net_theta) / 500),
            min(20, abs(net_vega) / 5000),
            min(20, risk_metrics['max_loss_1pct_move'] / 1000)
        ]
        
        risk_metrics['overall_risk_score'] = sum(risk_components)
        
        return risk_metrics