# analysis_modules/strategy_identifier.py
"""
Complex options strategy identifier
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict

class StrategyIdentifier:
    """Identifies complex multi-leg options strategies"""
    
    def __init__(self):
        self.strategy_patterns = {
            'vertical_spread': self._check_vertical_spread,
            'calendar_spread': self._check_calendar_spread,
            'straddle': self._check_straddle,
            'strangle': self._check_strangle,
            'butterfly': self._check_butterfly,
            'condor': self._check_condor,
            'collar': self._check_collar,
            'risk_reversal': self._check_risk_reversal,
            'ratio_spread': self._check_ratio_spread,
            'diagonal_spread': self._check_diagonal_spread
        }
        
        self.time_window = timedelta(minutes=5)  # Time window for related trades
    
    def identify_strategies(self, df: pd.DataFrame) -> dict:
        """Main function to identify all strategies"""
        strategies = {
            'identified_strategies': [],
            'strategy_summary': {},
            'complex_positions': [],
            'institutional_strategies': [],
            'retail_strategies': []
        }
        
        # Sort by time for proper grouping
        df_sorted = df.sort_values('Time_dt')
        
        # Group potential related trades
        trade_groups = self._group_related_trades(df_sorted)
        
        # Check each group for strategies
        for group_id, trades in trade_groups.items():
            if len(trades) >= 2:  # Need at least 2 legs
                identified = self._check_all_patterns(trades)
                strategies['identified_strategies'].extend(identified)
        
        # Categorize strategies
        for strategy in strategies['identified_strategies']:
            # By size
            if strategy['total_notional'] > 100000:
                strategies['institutional_strategies'].append(strategy)
            else:
                strategies['retail_strategies'].append(strategy)
            
            # Count by type
            strat_type = strategy['strategy_type']
            strategies['strategy_summary'][strat_type] = strategies['strategy_summary'].get(strat_type, 0) + 1
        
        # Identify complex positions (3+ legs)
        strategies['complex_positions'] = [
            s for s in strategies['identified_strategies'] 
            if s['num_legs'] >= 3
        ]
        
        return strategies
    
    def _group_related_trades(self, df: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Group trades that might be part of same strategy"""
        groups = {}
        group_id = 0
        used_indices = set()
        
        for i, trade in df.iterrows():
            if i in used_indices:
                continue
            
            # Start new group
            group_trades = [trade]
            used_indices.add(i)
            
            # Look for related trades
            for j, other_trade in df.iterrows():
                if j <= i or j in used_indices:
                    continue
                
                # Check if potentially related
                if self._are_trades_related(trade, other_trade):
                    group_trades.append(other_trade)
                    used_indices.add(j)
            
            # Store group if meaningful
            if len(group_trades) >= 2:
                groups[group_id] = pd.DataFrame(group_trades)
                group_id += 1
        
        return groups
    
    def _are_trades_related(self, trade1: pd.Series, trade2: pd.Series) -> bool:
        """Check if two trades might be part of same strategy"""
        # Time proximity
        time_diff = abs((trade2['Time_dt'] - trade1['Time_dt']).total_seconds())
        if time_diff > self.time_window.total_seconds():
            return False
        
        # Same underlying
        ticker1 = self._extract_ticker(trade1['StandardOptionSymbol'])
        ticker2 = self._extract_ticker(trade2['StandardOptionSymbol'])
        if ticker1 != ticker2:
            return False
        
        # Same trader (approximation: same exchange and similar aggressor)
        if trade1.get('Exchange') == trade2.get('Exchange'):
            return True
        
        # Look for spread indicators
        if 'Spread' in str(trade1.get('Condition', '')) or 'Spread' in str(trade2.get('Condition', '')):
            return True
        
        return False
    
    def _extract_ticker(self, option_symbol: str) -> str:
        """Extract underlying ticker from option symbol"""
        # Simple extraction - might need refinement
        symbol = option_symbol.lstrip('.')
        # Find where numbers start (date)
        for i, char in enumerate(symbol):
            if char.isdigit():
                return symbol[:i]
        return symbol
    
    def _check_all_patterns(self, trades: pd.DataFrame) -> list[dict]:
        """Check all strategy patterns against trade group"""
        identified = []
        
        for strategy_name, check_func in self.strategy_patterns.items():
            result = check_func(trades)
            if result:
                result['strategy_type'] = strategy_name
                result['identification_time'] = datetime.now()
                identified.append(result)
        
        return identified
    
    def _check_vertical_spread(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for vertical spread (bull/bear call/put spread)"""
        if len(trades) != 2:
            return None
        
        trade1 = trades.iloc[0]
        trade2 = trades.iloc[1]
        
        # Same expiration, same type, different strikes
        if (trade1.get('Expiration_Date_calc') == trade2.get('Expiration_Date_calc') and
            trade1.get('Option_Type_calc') == trade2.get('Option_Type_calc') and
            trade1.get('Strike_Price_calc') != trade2.get('Strike_Price_calc')):
            
            # Opposite sides (buy one, sell other)
            if (('Buy' in trade1['Aggressor'] and 'Sell' in trade2['Aggressor']) or
                ('Sell' in trade1['Aggressor'] and 'Buy' in trade2['Aggressor'])):
                
                # Determine spread type
                option_type = trade1.get('Option_Type_calc', 'Unknown')
                
                if 'Buy' in trade1['Aggressor']:
                    long_strike = trade1['Strike_Price_calc']
                    short_strike = trade2['Strike_Price_calc']
                else:
                    long_strike = trade2['Strike_Price_calc']
                    short_strike = trade1['Strike_Price_calc']
                
                if option_type == 'Call':
                    spread_type = 'Bull Call Spread' if long_strike < short_strike else 'Bear Call Spread'
                else:
                    spread_type = 'Bull Put Spread' if long_strike > short_strike else 'Bear Put Spread'
                
                return {
                    'strategy_name': spread_type,
                    'legs': trades.to_dict('records'),
                    'num_legs': 2,
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'max_profit': abs(long_strike - short_strike) * 100 - abs(self._calculate_net_premium(trades)),
                    'max_loss': abs(self._calculate_net_premium(trades)),
                    'strikes': [long_strike, short_strike],
                    'expiration': trade1.get('Expiration_Date_calc')
                }
        
        return None
    
    def _check_calendar_spread(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for calendar spread"""
        if len(trades) != 2:
            return None
        
        trade1 = trades.iloc[0]
        trade2 = trades.iloc[1]
        
        # Same strike, same type, different expirations
        if (trade1.get('Strike_Price_calc') == trade2.get('Strike_Price_calc') and
            trade1.get('Option_Type_calc') == trade2.get('Option_Type_calc') and
            trade1.get('Expiration_Date_calc') != trade2.get('Expiration_Date_calc')):
            
            # Opposite sides
            if (('Buy' in trade1['Aggressor'] and 'Sell' in trade2['Aggressor']) or
                ('Sell' in trade1['Aggressor'] and 'Buy' in trade2['Aggressor'])):
                
                return {
                    'strategy_name': 'Calendar Spread',
                    'legs': trades.to_dict('records'),
                    'num_legs': 2,
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'strike': trade1['Strike_Price_calc'],
                    'near_expiry': min(trade1.get('Expiration_Date_calc'), trade2.get('Expiration_Date_calc')),
                    'far_expiry': max(trade1.get('Expiration_Date_calc'), trade2.get('Expiration_Date_calc'))
                }
        
        return None
    
    def _check_straddle(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for straddle"""
        if len(trades) != 2:
            return None
        
        trade1 = trades.iloc[0]
        trade2 = trades.iloc[1]
        
        # Same strike, same expiration, different types (call and put)
        if (trade1.get('Strike_Price_calc') == trade2.get('Strike_Price_calc') and
            trade1.get('Expiration_Date_calc') == trade2.get('Expiration_Date_calc') and
            trade1.get('Option_Type_calc') != trade2.get('Option_Type_calc')):
            
            # Same side (both buys or both sells)
            if (('Buy' in trade1['Aggressor'] and 'Buy' in trade2['Aggressor']) or
                ('Sell' in trade1['Aggressor'] and 'Sell' in trade2['Aggressor'])):
                
                position = 'Long' if 'Buy' in trade1['Aggressor'] else 'Short'
                
                return {
                    'strategy_name': f'{position} Straddle',
                    'legs': trades.to_dict('records'),
                    'num_legs': 2,
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'strike': trade1['Strike_Price_calc'],
                    'expiration': trade1.get('Expiration_Date_calc'),
                    'breakeven_up': trade1['Strike_Price_calc'] + abs(self._calculate_net_premium(trades))/100,
                    'breakeven_down': trade1['Strike_Price_calc'] - abs(self._calculate_net_premium(trades))/100
                }
        
        return None
    
    def _check_strangle(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for strangle"""
        if len(trades) != 2:
            return None
        
        trade1 = trades.iloc[0]
        trade2 = trades.iloc[1]
        
        # Different strikes, same expiration, different types
        if (trade1.get('Strike_Price_calc') != trade2.get('Strike_Price_calc') and
            trade1.get('Expiration_Date_calc') == trade2.get('Expiration_Date_calc') and
            trade1.get('Option_Type_calc') != trade2.get('Option_Type_calc')):
            
            # Same side
            if (('Buy' in trade1['Aggressor'] and 'Buy' in trade2['Aggressor']) or
                ('Sell' in trade1['Aggressor'] and 'Sell' in trade2['Aggressor'])):
                
                position = 'Long' if 'Buy' in trade1['Aggressor'] else 'Short'
                
                call_trade = trade1 if trade1.get('Option_Type_calc') == 'Call' else trade2
                put_trade = trade2 if trade2.get('Option_Type_calc') == 'Put' else trade1
                
                return {
                    'strategy_name': f'{position} Strangle',
                    'legs': trades.to_dict('records'),
                    'num_legs': 2,
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'call_strike': call_trade['Strike_Price_calc'],
                    'put_strike': put_trade['Strike_Price_calc'],
                    'expiration': trade1.get('Expiration_Date_calc')
                }
        
        return None
    
    def _check_butterfly(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for butterfly spread"""
        if len(trades) != 3 and len(trades) != 4:  # 3 strikes, but middle might be 2 contracts
            return None
        
        # Group by strike
        strike_groups = trades.groupby('Strike_Price_calc').agg({
            'TradeQuantity': 'sum',
            'Aggressor': 'first',
            'Option_Type_calc': 'first',
            'Expiration_Date_calc': 'first'
        })
        
        if len(strike_groups) != 3:
            return None
        
        # Check if all same type and expiration
        if (strike_groups['Option_Type_calc'].nunique() == 1 and
            strike_groups['Expiration_Date_calc'].nunique() == 1):
            
            strikes = sorted(strike_groups.index)
            
            # Check for butterfly pattern: buy 1, sell 2, buy 1 (or inverse)
            # Middle strike should have double quantity
            middle_qty = strike_groups.loc[strikes[1], 'TradeQuantity']
            outer_qty = strike_groups.loc[strikes[0], 'TradeQuantity'] + strike_groups.loc[strikes[2], 'TradeQuantity']
            
            if abs(middle_qty) == abs(outer_qty):
                return {
                    'strategy_name': 'Butterfly Spread',
                    'legs': trades.to_dict('records'),
                    'num_legs': len(trades),
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'strikes': strikes,
                    'expiration': strike_groups['Expiration_Date_calc'].iloc[0],
                    'max_profit_strike': strikes[1]
                }
        
        return None
    
    def _check_condor(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for condor spread"""
        if len(trades) != 4:
            return None
        
        # Group by strike
        strike_groups = trades.groupby('Strike_Price_calc').agg({
            'TradeQuantity': 'sum',
            'Aggressor': 'first',
            'Option_Type_calc': 'first',
            'Expiration_Date_calc': 'first'
        })
        
        if len(strike_groups) != 4:
            return None
        
        # Check if all same type and expiration
        if (strike_groups['Option_Type_calc'].nunique() == 1 and
            strike_groups['Expiration_Date_calc'].nunique() == 1):
            
            strikes = sorted(strike_groups.index)
            
            # Check for condor pattern: buy 1, sell 1, sell 1, buy 1
            quantities = [strike_groups.loc[s, 'TradeQuantity'] for s in strikes]
            
            if (abs(quantities[0]) == abs(quantities[3]) and
                abs(quantities[1]) == abs(quantities[2]) and
                np.sign(quantities[0]) != np.sign(quantities[1])):
                
                return {
                    'strategy_name': 'Iron Condor' if trades['Option_Type_calc'].nunique() > 1 else 'Condor Spread',
                    'legs': trades.to_dict('records'),
                    'num_legs': 4,
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'strikes': strikes,
                    'expiration': strike_groups['Expiration_Date_calc'].iloc[0],
                    'profit_range': [strikes[1], strikes[2]]
                }
        
        return None
    
    def _check_collar(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for collar (protective put + covered call)"""
        # This would need stock position info
        # For now, check for put buy + call sell
        calls = trades[trades['Option_Type_calc'] == 'Call']
        puts = trades[trades['Option_Type_calc'] == 'Put']
        
        if len(calls) == 1 and len(puts) == 1:
            call = calls.iloc[0]
            put = puts.iloc[0]
            
            # Put buy + Call sell
            if 'Buy' in put['Aggressor'] and 'Sell' in call['Aggressor']:
                if put['Expiration_Date_calc'] == call['Expiration_Date_calc']:
                    return {
                        'strategy_name': 'Collar',
                        'legs': trades.to_dict('records'),
                        'num_legs': 2,
                        'total_notional': trades['NotionalValue'].sum(),
                        'net_debit_credit': self._calculate_net_premium(trades),
                        'put_strike': put['Strike_Price_calc'],
                        'call_strike': call['Strike_Price_calc'],
                        'expiration': put['Expiration_Date_calc'],
                        'protected_range': [put['Strike_Price_calc'], call['Strike_Price_calc']]
                    }
        
        return None
    
    def _check_risk_reversal(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for risk reversal"""
        calls = trades[trades['Option_Type_calc'] == 'Call']
        puts = trades[trades['Option_Type_calc'] == 'Put']
        
        if len(calls) == 1 and len(puts) == 1:
            call = calls.iloc[0]
            put = puts.iloc[0]
            
            # Opposite sides on call and put
            if (('Buy' in call['Aggressor'] and 'Sell' in put['Aggressor']) or
                ('Sell' in call['Aggressor'] and 'Buy' in put['Aggressor'])):
                
                if call['Expiration_Date_calc'] == put['Expiration_Date_calc']:
                    
                    # Determine type
                    if 'Buy' in call['Aggressor']:
                        reversal_type = 'Long Risk Reversal'  # Bullish
                    else:
                        reversal_type = 'Short Risk Reversal'  # Bearish
                    
                    return {
                        'strategy_name': reversal_type,
                        'legs': trades.to_dict('records'),
                        'num_legs': 2,
                        'total_notional': trades['NotionalValue'].sum(),
                        'net_debit_credit': self._calculate_net_premium(trades),
                        'call_strike': call['Strike_Price_calc'],
                        'put_strike': put['Strike_Price_calc'],
                        'expiration': call['Expiration_Date_calc'],
                        'direction': 'Bullish' if 'Buy' in call['Aggressor'] else 'Bearish'
                    }
        
        return None
    
    def _check_ratio_spread(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for ratio spread"""
        if len(trades) < 2:
            return None
        
        # Group by option type and expiration
        groups = trades.groupby(['Option_Type_calc', 'Expiration_Date_calc'])
        
        for (opt_type, expiry), group in groups:
            if len(group) < 2:
                continue
                
            # Check for different quantities at different strikes
            strike_quantities = group.groupby('Strike_Price_calc')['TradeQuantity'].sum()
            
            if len(strike_quantities) == 2:
                quantities = list(strike_quantities.values)
                strikes = list(strike_quantities.index)
                
                # Check for ratio (not 1:1)
                if quantities[0] != quantities[1]:
                    ratio = max(quantities) / min(quantities)
                    
                    if ratio >= 1.5:  # At least 1.5:1 ratio
                        return {
                            'strategy_name': f'Ratio {opt_type} Spread',
                            'legs': group.to_dict('records'),
                            'num_legs': len(group),
                            'total_notional': group['NotionalValue'].sum(),
                            'net_debit_credit': self._calculate_net_premium(group),
                            'strikes': strikes,
                            'ratio': f"{int(max(quantities))}:{int(min(quantities))}",
                            'expiration': expiry
                        }
        
        return None
    
    def _check_diagonal_spread(self, trades: pd.DataFrame) -> Optional[dict]:
        """Check for diagonal spread"""
        if len(trades) != 2:
            return None
        
        trade1 = trades.iloc[0]
        trade2 = trades.iloc[1]
        
        # Different strikes AND different expirations, same type
        if (trade1.get('Strike_Price_calc') != trade2.get('Strike_Price_calc') and
            trade1.get('Expiration_Date_calc') != trade2.get('Expiration_Date_calc') and
            trade1.get('Option_Type_calc') == trade2.get('Option_Type_calc')):
            
            # Opposite sides
            if (('Buy' in trade1['Aggressor'] and 'Sell' in trade2['Aggressor']) or
                ('Sell' in trade1['Aggressor'] and 'Buy' in trade2['Aggressor'])):
                
                return {
                    'strategy_name': 'Diagonal Spread',
                    'legs': trades.to_dict('records'),
                    'num_legs': 2,
                    'total_notional': trades['NotionalValue'].sum(),
                    'net_debit_credit': self._calculate_net_premium(trades),
                    'strikes': [trade1['Strike_Price_calc'], trade2['Strike_Price_calc']],
                    'expirations': [trade1.get('Expiration_Date_calc'), trade2.get('Expiration_Date_calc')],
                    'option_type': trade1.get('Option_Type_calc')
                }
        
        return None
    
    def _calculate_net_premium(self, trades: pd.DataFrame) -> float:
        """Calculate net premium paid/received for strategy"""
        net_premium = 0
        
        for _, trade in trades.iterrows():
            premium = trade['Trade_Price'] * trade['TradeQuantity'] * 100
            
            if 'Buy' in trade['Aggressor']:
                net_premium -= premium  # Debit
            else:
                net_premium += premium  # Credit
        
        return net_premium
    
    def analyze_strategy_quality(self, strategy: dict) -> dict:
        """Analyze quality and characteristics of identified strategy"""
        quality_metrics = {
            'risk_reward_ratio': 0,
            'complexity_score': 0,
            'execution_quality': 0,
            'market_outlook': 'Neutral',
            'time_sensitivity': 'Medium',
            'volatility_bias': 'Neutral'
        }
        
        # Calculate risk/reward if available
        if 'max_profit' in strategy and 'max_loss' in strategy:
            if strategy['max_loss'] > 0:
                quality_metrics['risk_reward_ratio'] = strategy['max_profit'] / strategy['max_loss']
        
        # Complexity score based on number of legs
        num_legs = strategy.get('num_legs', 0)
        quality_metrics['complexity_score'] = min(100, num_legs * 25)
        
        # Execution quality based on net premium vs market prices
        # This would need real market data for accurate assessment
        quality_metrics['execution_quality'] = 75  # Default good execution
        
        # Market outlook based on strategy type
        strategy_name = strategy.get('strategy_name', '').lower()
        if 'bull' in strategy_name or 'long call' in strategy_name:
            quality_metrics['market_outlook'] = 'Bullish'
        elif 'bear' in strategy_name or 'long put' in strategy_name:
            quality_metrics['market_outlook'] = 'Bearish'
        elif 'straddle' in strategy_name or 'strangle' in strategy_name:
            quality_metrics['market_outlook'] = 'High Volatility Expected'
        elif 'butterfly' in strategy_name or 'condor' in strategy_name:
            quality_metrics['market_outlook'] = 'Low Volatility Expected'
        
        # Time sensitivity
        if any(exp for exp in strategy.get('expirations', []) if exp):
            # Would calculate DTE here with real dates
            quality_metrics['time_sensitivity'] = 'High'  # Placeholder
        
        return quality_metrics
    
    def generate_strategy_report(self, strategies: dict) -> str:
        """Generate detailed strategy analysis report"""
        report = "=" * 60 + "\n"
        report += "COMPLEX OPTIONS STRATEGY ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Summary
        total_strategies = len(strategies['identified_strategies'])
        report += f"Total Strategies Identified: {total_strategies}\n"
        report += f"Institutional Strategies: {len(strategies['institutional_strategies'])}\n"
        report += f"Retail Strategies: {len(strategies['retail_strategies'])}\n"
        report += f"Complex Positions (3+ legs): {len(strategies['complex_positions'])}\n\n"
        
        # Strategy breakdown
        if strategies['strategy_summary']:
            report += "STRATEGY TYPE BREAKDOWN:\n"
            report += "-" * 30 + "\n"
            for strat_type, count in strategies['strategy_summary'].items():
                report += f"{strat_type.replace('_', ' ').title()}: {count}\n"
            report += "\n"
        
        # Detailed analysis
        if strategies['identified_strategies']:
            report += "DETAILED STRATEGY ANALYSIS:\n"
            report += "-" * 40 + "\n"
            
            for i, strategy in enumerate(strategies['identified_strategies'][:10], 1):
                report += f"\n{i}. {strategy['strategy_name']}\n"
                report += f"   Legs: {strategy['num_legs']}\n"
                report += f"   Total Notional: ${strategy['total_notional']:,.0f}\n"
                report += f"   Net Premium: ${strategy['net_debit_credit']:,.0f}\n"
                
                if 'strikes' in strategy:
                    strikes_str = " / ".join([f"${s:.0f}" for s in strategy['strikes']])
                    report += f"   Strikes: {strikes_str}\n"
                
                if 'expiration' in strategy:
                    report += f"   Expiration: {strategy['expiration']}\n"
                
                # Add quality analysis
                quality = self.analyze_strategy_quality(strategy)
                report += f"   Market Outlook: {quality['market_outlook']}\n"
                report += f"   Complexity: {quality['complexity_score']}/100\n"
                
                if quality['risk_reward_ratio'] > 0:
                    report += f"   Risk/Reward: {quality['risk_reward_ratio']:.2f}\n"
                
                report += "\n"
        
        # Key insights
        report += "KEY INSIGHTS:\n"
        report += "-" * 20 + "\n"
        
        if strategies['institutional_strategies']:
            report += f"• {len(strategies['institutional_strategies'])} large institutional strategies detected\n"
        
        if strategies['complex_positions']:
            report += f"• {len(strategies['complex_positions'])} complex multi-leg positions identified\n"
        
        # Most common strategy
        if strategies['strategy_summary']:
            most_common = max(strategies['strategy_summary'].items(), key=lambda x: x[1])
            report += f"• Most common strategy: {most_common[0].replace('_', ' ').title()} ({most_common[1]} instances)\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report

def integrate_with_alpha_extractor(strategies: dict, alpha_signals: list) -> dict:
    """Integrate strategy identification with alpha extraction"""
    enhanced_signals = []
    
    for signal in alpha_signals:
        # Check if this signal is part of a strategy
        signal_time = signal.timestamp
        related_strategies = []
        
        for strategy in strategies['identified_strategies']:
            strategy_time = strategy.get('identification_time')
            if strategy_time and abs((signal_time - strategy_time).total_seconds()) < 300:  # 5 minutes
                # Check if option symbol matches any leg
                for leg in strategy.get('legs', []):
                    if leg.get('StandardOptionSymbol') == signal.option_symbol:
                        related_strategies.append(strategy)
                        break
        
        # Enhance signal with strategy context
        if related_strategies:
            signal.metadata['related_strategies'] = related_strategies
            signal.metadata['strategy_context'] = True
            
            # Boost confidence for strategy trades
            signal.confidence *= 1.15
            signal.smart_money_score *= 1.1
            
            # Update recommendation
            strategy_names = [s['strategy_name'] for s in related_strategies]
            signal.trade_recommendation += f" (Part of {', '.join(strategy_names)})"
        
        enhanced_signals.append(signal)
    
    return enhanced_signals