# alpha_extractor.py (with modifications for compatibility)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta # Keep timedelta if used by thresholds
from typing import List, Tuple, Optional, Dict, Callable, Any # Keep typing
from dataclasses import dataclass
from enum import Enum
# Assuming config and alpha_config are available in your environment
# import config 
# import alpha_config 

# (SignalType, SignalStrength, AlphaSignal dataclass remain unchanged)
class SignalType(Enum):
    BLOCK_TRADE = "Block Trade"; SWEEP_ORDER = "Sweep Order"; UNUSUAL_VOLUME = "Unusual Volume"
    IV_SPIKE = "IV Spike"; SMART_MONEY_ACCUMULATION = "Smart Money Accumulation"
    VOLATILITY_ARBITRAGE = "Volatility Arbitrage"; GAMMA_SQUEEZE = "Gamma Squeeze"
    FLOW_DIVERGENCE = "Flow Divergence"
class SignalStrength(Enum): WEAK = 1; MODERATE = 2; STRONG = 3; VERY_STRONG = 4
@dataclass
class AlphaSignal:
    timestamp: datetime; signal_type: SignalType; strength: SignalStrength; ticker: str
    option_symbol: str; direction: str; confidence: float; entry_price: float
    target_price: Optional[float]; stop_price: Optional[float]; notional_value: float
    urgency_score: float; smart_money_score: float; metadata: dict; trade_recommendation: str


class AlphaExtractor:
    def __init__(self, ticker: str, underlying_price: float):
        self.ticker = ticker
        self.underlying_price = underlying_price
        self.signals = []
        self.flow_history = [] # Not explicitly used in provided snippet, but part of class state
        
        # Thresholds (keep as is, or move to alpha_config.py if preferred)
        self.BLOCK_SIZE_THRESHOLD = 50
        self.SWEEP_TIME_WINDOW = timedelta(seconds=5)
        self.SWEEP_MIN_LEGS = 3
        self.UNUSUAL_VOLUME_MULTIPLIER = 3.0 
        self.SMART_MONEY_MIN_PREMIUM = 10000 
        self.IV_SPIKE_THRESHOLD = 0.15

    def process_trade_flow(self, df: pd.DataFrame) -> list[AlphaSignal]:
        self._validate_data(df) # Validation will now check against columns provided by new data_utils
        signals = []
        block_signals = self._extract_block_trades(df); signals.extend(block_signals)
        sweep_signals = self._extract_sweep_orders(df); signals.extend(sweep_signals)
        smart_money_signals = self._extract_smart_money(df); signals.extend(smart_money_signals)
        vol_arb_signals = self._extract_volatility_arbitrage(df); signals.extend(vol_arb_signals)
        toxicity_enhanced_signals = self._analyze_flow_toxicity(signals, df)
        ranked_signals = self._rank_and_filter_signals(toxicity_enhanced_signals)
        return ranked_signals
    
    # --- _extract_block_trades (No changes needed in logic, uses Time_dt, StandardOptionSymbol) ---
    def _extract_block_trades(self, df: pd.DataFrame) -> list[AlphaSignal]:
        signals = []
        block_trades = df[df['TradeQuantity'] >= self.BLOCK_SIZE_THRESHOLD].copy()
        for idx, trade in block_trades.iterrows():
            notional = trade['NotionalValue'] # Use pre-calculated NotionalValue
            urgency = 50; direction = "NEUTRAL"
            if pd.notna(trade['Aggressor']):
                if 'Ask' in trade['Aggressor']: # Compatible with "Buyer (At Ask)"
                    urgency = 90 if pd.notna(trade['Option_Ask']) and trade['Trade_Price'] >= trade['Option_Ask'] else 70
                    direction = "BULLISH"
                elif 'Bid' in trade['Aggressor']: # Compatible with "Seller (At Bid)"
                    urgency = 90 if pd.notna(trade['Option_Bid']) and trade['Trade_Price'] <= trade['Option_Bid'] else 70
                    direction = "BEARISH"

            smart_score = self._calculate_smart_money_score(trade)
            signal = AlphaSignal(
                timestamp=trade['Time_dt'], signal_type=SignalType.BLOCK_TRADE,
                strength=self._determine_signal_strength(trade), ticker=self.ticker,
                option_symbol=trade['StandardOptionSymbol'], direction=direction,
                confidence=self._calculate_confidence(trade, df), entry_price=trade['Trade_Price'],
                target_price=self._calculate_target(trade), stop_price=self._calculate_stop(trade),
                notional_value=notional, urgency_score=urgency, smart_money_score=smart_score,
                metadata={'quantity': trade['TradeQuantity'], 'iv': trade['IV'], 'delta': trade['Delta'],
                          'underlying_price': trade['Underlying_Price'], 'exchange': trade['Exchange'],
                          'condition': trade.get('Condition', '')},
                trade_recommendation=self._generate_trade_recommendation(trade, direction)
            ); signals.append(signal)
        return signals

    # --- _extract_sweep_orders (No changes needed in logic, uses Time_dt, StandardOptionSymbol) ---
    def _extract_sweep_orders(self, df: pd.DataFrame) -> list[AlphaSignal]:
        signals = []; df_sorted = df.sort_values(['StandardOptionSymbol', 'Time_dt'])
        for symbol in df_sorted['StandardOptionSymbol'].unique():
            symbol_trades = df_sorted[df_sorted['StandardOptionSymbol'] == symbol]
            i = 0
            while i < len(symbol_trades):
                sweep_group = [i]; current_time = symbol_trades.iloc[i]['Time_dt']
                current_side = symbol_trades.iloc[i]['Aggressor']; j = i + 1
                exchanges_seen = {symbol_trades.iloc[i]['Exchange']}
                while j < len(symbol_trades):
                    time_diff = symbol_trades.iloc[j]['Time_dt'] - current_time
                    if (time_diff <= self.SWEEP_TIME_WINDOW and 
                        symbol_trades.iloc[j]['Aggressor'] == current_side):
                        sweep_group.append(j); exchanges_seen.add(symbol_trades.iloc[j]['Exchange'])
                    else: break
                    j += 1
                if len(sweep_group) >= self.SWEEP_MIN_LEGS and len(exchanges_seen) >= 2:
                    sweep_trades = symbol_trades.iloc[sweep_group]
                    total_quantity = sweep_trades['TradeQuantity'].sum()
                    avg_price = (sweep_trades['Trade_Price'] * sweep_trades['TradeQuantity']).sum() / total_quantity
                    total_notional = sweep_trades['NotionalValue'].sum() # Use sum of NotionalValue
                    direction = "BULLISH" if pd.notna(current_side) and 'Ask' in current_side else "BEARISH"
                    urgency = 95
                    signal = AlphaSignal(
                        timestamp=sweep_trades.iloc[0]['Time_dt'], signal_type=SignalType.SWEEP_ORDER,
                        strength=SignalStrength.VERY_STRONG, ticker=self.ticker, option_symbol=str(symbol),
                        direction=direction, confidence=85.0, entry_price=avg_price,
                        target_price=self._calculate_target(sweep_trades.iloc[0]),
                        stop_price=self._calculate_stop(sweep_trades.iloc[0]),
                        notional_value=total_notional, urgency_score=urgency, smart_money_score=90.0,
                        metadata={'total_quantity': total_quantity, 'num_legs': len(sweep_group),
                                  'exchanges': list(exchanges_seen),
                                  'duration_ms': (sweep_trades.iloc[-1]['Time_dt'] - sweep_trades.iloc[0]['Time_dt']).total_seconds() * 1000,
                                  'avg_iv': sweep_trades['IV'].mean(), 'avg_delta': sweep_trades['Delta'].mean()},
                        trade_recommendation=f"Follow sweep: Consider {direction} {symbol} @ {avg_price:.2f}"
                    ); signals.append(signal)
                i = j # Move to the end of the processed group
        return signals

    # --- _extract_smart_money (Uses str.contains which is fine with new Aggressor values) ---
    def _extract_smart_money(self, df: pd.DataFrame) -> list[AlphaSignal]:
        signals = []
        # Ensure NotionalValue is present for aggregation
        if 'NotionalValue' not in df.columns: df['NotionalValue'] = df['TradeQuantity'] * df['Trade_Price'] * 100

        grouped = df.groupby(['StandardOptionSymbol']).agg(
            TradeQuantity_sum=('TradeQuantity', 'sum'), TradeQuantity_count=('TradeQuantity', 'count'),
            TradeQuantity_mean=('TradeQuantity', 'mean'), NotionalValue_sum=('NotionalValue', 'sum'),
            Trade_Price_mean=('Trade_Price', 'mean'), IV_mean=('IV', 'mean'), Delta_mean=('Delta', 'mean'),
            Aggressor_mode=('Aggressor', lambda x: x.mode()[0] if not x.empty and not x.mode().empty else 'Unknown'),
            Last_Time_dt=('Time_dt', 'last') # Get the last timestamp for the group
        )
        for symbol, row in grouped.iterrows():
            total_notional = row['NotionalValue_sum']
            if total_notional >= self.SMART_MONEY_MIN_PREMIUM:
                symbol_trades = df[df['StandardOptionSymbol'] == symbol]
                # Ensure Aggressor column is string type for .str.contains
                buy_trades = symbol_trades[symbol_trades['Aggressor'].astype(str).str.contains('Buy|Ask', na=False, case=False)]
                sell_trades = symbol_trades[symbol_trades['Aggressor'].astype(str).str.contains('Sell|Bid', na=False, case=False)]
                buy_volume = buy_trades['TradeQuantity'].sum(); sell_volume = sell_trades['TradeQuantity'].sum()
                direction = ""; confidence = 0.0
                if buy_volume > sell_volume * 2:
                    direction = "BULLISH"; confidence = min(95, (buy_volume / (buy_volume + sell_volume + 1e-6)) * 100)
                elif sell_volume > buy_volume * 2:
                    direction = "BEARISH"; confidence = min(95, (sell_volume / (buy_volume + sell_volume + 1e-6)) * 100)
                else: continue
                signal = AlphaSignal(
                    timestamp=row['Last_Time_dt'], signal_type=SignalType.SMART_MONEY_ACCUMULATION,
                    strength=SignalStrength.STRONG, ticker=self.ticker, option_symbol=str(symbol), direction=direction,
                    confidence=confidence, entry_price=row['Trade_Price_mean'], target_price=None, stop_price=None,
                    notional_value=total_notional, urgency_score=75, smart_money_score=85,
                    metadata={'total_trades': row['TradeQuantity_count'], 'avg_trade_size': row['TradeQuantity_mean'],
                              'total_volume': row['TradeQuantity_sum'], 'buy_volume': buy_volume, 'sell_volume': sell_volume,
                              'avg_iv': row['IV_mean'], 'avg_delta': row['Delta_mean']},
                    trade_recommendation=self._generate_accumulation_recommendation(str(symbol), direction, row)
                ); signals.append(signal)
        return signals
        
    def _extract_volatility_arbitrage(self, df: pd.DataFrame) -> list[AlphaSignal]:
        signals = []
        # MODIFIED: Use 'Expiration_Date' and 'Strike_Price'
        for expiry in df['Expiration_Date'].unique(): # Use 'Expiration_Date'
            if pd.isna(expiry): continue
            expiry_trades = df[df['Expiration_Date'] == expiry]
            if 'NotionalValue' not in expiry_trades.columns: # Ensure NotionalValue for sum
                 expiry_trades['NotionalValue'] = expiry_trades['TradeQuantity'] * expiry_trades['Trade_Price'] * 100

            iv_by_strike = expiry_trades.groupby('Strike_Price').agg( # Use 'Strike_Price'
                IV_mean=('IV', 'mean'), IV_std=('IV', 'std'), IV_count=('IV', 'count'),
                TradeQuantity_sum=('TradeQuantity', 'sum'),
                NotionalValue_sum=('NotionalValue','sum'), # Added for signal
                Last_Time_dt=('Time_dt', 'last'),          # Added for signal
                Last_Symbol=('StandardOptionSymbol', 'last'), # Added for signal
                Last_Trade_Price=('Trade_Price','mean')    # Added for signal
            )
            if len(iv_by_strike) < 3: continue
            
            mean_overall_iv = iv_by_strike['IV_mean'].mean() # Mean of mean IVs for the expiry
            std_overall_iv = iv_by_strike['IV_mean'].std()   # Std of mean IVs for the expiry
            if pd.isna(std_overall_iv) or std_overall_iv == 0: continue

            for strike, row in iv_by_strike.iterrows():
                strike_iv_mean = row['IV_mean']
                if pd.isna(strike_iv_mean): continue

                if abs(strike_iv_mean - mean_overall_iv) > 2 * std_overall_iv and row['TradeQuantity_sum'] > 20:
                    signal = AlphaSignal(
                        timestamp=row['Last_Time_dt'], signal_type=SignalType.VOLATILITY_ARBITRAGE,
                        strength=SignalStrength.MODERATE, ticker=self.ticker,
                        option_symbol=row['Last_Symbol'], direction="NEUTRAL", confidence=70,
                        entry_price=row['Last_Trade_Price'], target_price=None, stop_price=None,
                        notional_value=row['NotionalValue_sum'], urgency_score=60, smart_money_score=75,
                        metadata={'strike_iv': strike_iv_mean, 'mean_expiry_iv': mean_overall_iv,
                                  'iv_deviation_ratio': (strike_iv_mean - mean_overall_iv) / (mean_overall_iv + 1e-6),
                                  'volume': row['TradeQuantity_sum'], 'trade_count': row['IV_count']},
                        trade_recommendation=f"Vol Arb: {'Sell' if strike_iv_mean > mean_overall_iv else 'Buy'} IV at {strike} for {self.ticker} {pd.to_datetime(expiry).strftime('%d%b%y')}"
                    ); signals.append(signal)
        return signals

    # --- _analyze_flow_toxicity (No column name changes needed if StandardOptionSymbol is used) ---
    def _analyze_flow_toxicity(self, signals: list[AlphaSignal], df: pd.DataFrame) -> list[AlphaSignal]:
        if df.empty: return signals
        total_volume = df['TradeQuantity'].sum()
        # Ensure Aggressor column is string type for .str.contains
        buy_volume = df[df['Aggressor'].astype(str).str.contains('Buy|Ask', na=False, case=False)]['TradeQuantity'].sum()
        sell_volume = df[df['Aggressor'].astype(str).str.contains('Sell|Bid', na=False, case=False)]['TradeQuantity'].sum()
        order_imbalance = abs(buy_volume - sell_volume) / (total_volume + 1e-6) if total_volume > 0 else 0
        for signal in signals:
            option_trades = df[df['StandardOptionSymbol'] == signal.option_symbol]
            if option_trades.empty: 
                signal.metadata['flow_toxicity'] = {'order_imbalance': order_imbalance, 'informed_score': 0, 'avg_trade_size':0, 'trade_size_consistency':0}
                continue
            avg_trade_size = option_trades['TradeQuantity'].mean()
            trade_size_std = option_trades['TradeQuantity'].std(ddof=0) # Use ddof=0 if it's population std
            informed_score = (avg_trade_size / (trade_size_std + 1e-6)) * 10 if pd.notna(trade_size_std) and trade_size_std > 0 else 50
            signal.smart_money_score = min(100, signal.smart_money_score + informed_score)
            signal.metadata['flow_toxicity'] = {
                'order_imbalance': order_imbalance, 'informed_score': informed_score,
                'avg_trade_size': avg_trade_size if pd.notna(avg_trade_size) else 0,
                'trade_size_consistency': 1 / (1 + trade_size_std) if pd.notna(trade_size_std) and trade_size_std > 0 else 1
            }
        return signals
        
    # --- _calculate_smart_money_score (No column changes needed, uses Time_dt) ---
    def _calculate_smart_money_score(self, trade: pd.Series) -> float:
        score = 50.0
        if pd.notna(trade['TradeQuantity']):
            if trade['TradeQuantity'] >= 100: score += 20
            elif trade['TradeQuantity'] >= 50: score += 10
        if pd.notna(trade['Aggressor']) and pd.notna(trade['Trade_Price']):
            if 'Ask' in trade['Aggressor'] and pd.notna(trade['Option_Ask']) and trade['Trade_Price'] <= trade['Option_Ask']: score += 10
            elif 'Bid' in trade['Aggressor'] and pd.notna(trade['Option_Bid']) and trade['Trade_Price'] >= trade['Option_Bid']: score += 10
        institutional_exchanges = ['CBOE', 'ISE', 'PHLX', 'AMEX', 'BOX', 'MIAX'] # Expanded list
        if pd.notna(trade['Exchange']) and trade['Exchange'] in institutional_exchanges: score += 10
        if pd.notna(trade['Time_dt']):
            trade_hour = trade['Time_dt'].hour
            if 8 <= trade_hour <= 9 or 14 <= trade_hour <= 15: # Adjusted for typical US market hours (CDT/CST consideration)
                score += 5 
        return min(100, score)

    # --- _determine_signal_strength (No column changes needed) ---
    def _determine_signal_strength(self, trade: pd.Series) -> SignalStrength:
        strength_score = 0; notional = 0
        if pd.notna(trade['TradeQuantity']) and pd.notna(trade['Trade_Price']):
             notional = trade['NotionalValue'] # Use pre-calculated NotionalValue
        if pd.notna(trade['TradeQuantity']):
            if trade['TradeQuantity'] >= 200: strength_score += 3
            elif trade['TradeQuantity'] >= 100: strength_score += 2
            else: strength_score += 1
        if notional >= 50000: strength_score += 2
        elif notional >= 25000: strength_score += 1
        if pd.notna(trade['Aggressor']) and ('Ask' in trade['Aggressor'] or 'Bid' in trade['Aggressor']): strength_score += 1
        if strength_score >= 5: return SignalStrength.VERY_STRONG
        elif strength_score >= 3: return SignalStrength.STRONG
        elif strength_score >= 2: return SignalStrength.MODERATE
        return SignalStrength.WEAK

    # --- _calculate_confidence (No column changes needed if StandardOptionSymbol and IV are present) ---
    def _calculate_confidence(self, trade: pd.Series, df: pd.DataFrame) -> float:
        confidence = 50.0;
        if df.empty or trade.empty: return confidence
        symbol_trades = df[df['StandardOptionSymbol'] == trade['StandardOptionSymbol']]
        if not symbol_trades.empty:
            symbol_volume = symbol_trades['TradeQuantity'].sum()
            if pd.notna(trade['TradeQuantity']) and symbol_volume > trade['TradeQuantity'] * 3: confidence += 20
            if len(symbol_trades) > 1:
                price_trend = symbol_trades['Trade_Price'].diff().mean()
                if pd.notna(price_trend) and pd.notna(trade['Aggressor']):
                    if price_trend > 0 and ('Buy' in trade['Aggressor'] or 'Ask' in trade['Aggressor']): confidence += 15
                    elif price_trend < 0 and ('Sell' in trade['Aggressor'] or 'Bid' in trade['Aggressor']): confidence += 15
        mean_iv_all = df['IV'].mean() # Mean IV of all trades in the input df
        if pd.notna(trade['IV']) and pd.notna(mean_iv_all) and trade['IV'] < mean_iv_all : confidence += 10
        return min(95, confidence)

    def _calculate_target(self, trade: pd.Series) -> Optional[float]:
        # MODIFIED: Use 'Option_Type'
        if pd.notna(trade['Trade_Price']):
            # Using a simple 50% profit target for options as an example
            if pd.notna(trade.get('Option_Type')):
                if 'C' in trade.get('Option_Type', '').upper(): # Call or Put check
                    return trade['Trade_Price'] * 1.5 
                elif 'P' in trade.get('Option_Type', '').upper():
                    return trade['Trade_Price'] * 1.5
        return None
    
    def _calculate_stop(self, trade: pd.Series) -> Optional[float]:
        if pd.notna(trade['Trade_Price']):
            return trade['Trade_Price'] * 0.7 # Example: 30% stop loss
        return None
        
    def _generate_trade_recommendation(self, trade: pd.Series, direction: str) -> str:
        # MODIFIED: Use 'Option_Type', 'Strike_Price', 'Expiration_Date'
        option_type_str = str(trade.get('Option_Type', 'Option')).upper()
        strike_str = str(trade.get('Strike_Price', "N/A"))
        # Format Expiration_Date if it's a datetime object, else use as string
        expiry_val = trade.get('Expiration_Date')
        if pd.notna(expiry_val) and isinstance(expiry_val, datetime):
            expiry_str = expiry_val.strftime('%d%b%y').upper()
        elif pd.notna(expiry_val):
            expiry_str = str(expiry_val)
        else:
            expiry_str = "N/A"

        trade_price_str = f"{trade.get('Trade_Price', 0):.2f}"

        if direction == "BULLISH":
            if 'C' in option_type_str:
                return f"Buy {self.ticker} {expiry_str} {strike_str} Call @ {trade_price_str} or better"
            else: # Assuming Puts for bullish if not Call (e.g. selling puts)
                return f"Sell {self.ticker} {expiry_str} {strike_str} Put @ {trade_price_str} or better"
        elif direction == "BEARISH":
            if 'P' in option_type_str:
                return f"Buy {self.ticker} {expiry_str} {strike_str} Put @ {trade_price_str} or better"
            else: # Assuming Calls for bearish if not Put (e.g. selling calls)
                return f"Sell {self.ticker} {expiry_str} {strike_str} Call @ {trade_price_str} or better"
        return f"Monitor {self.ticker} {expiry_str} {strike_str} {option_type_str} for entry"

    # _generate_accumulation_recommendation (No column name changes needed from its direct inputs)
    def _generate_accumulation_recommendation(self, symbol: str, direction: str, row: pd.Series) -> str:
        avg_price = row['Trade_Price_mean']
        total_volume = row['TradeQuantity_sum']
        return f"{direction} accumulation detected: Consider following institutional flow in {symbol}. Avg entry: ${avg_price:.2f}, Total volume: {total_volume}"

    # _rank_and_filter_signals (No column changes needed)
    def _rank_and_filter_signals(self, signals: list[AlphaSignal]) -> list[AlphaSignal]:
        for signal in signals:
            signal.metadata['composite_score'] = (
                signal.strength.value * 25 + signal.confidence * 0.5 +
                signal.urgency_score * 0.3 + signal.smart_money_score * 0.2 )
        ranked_signals = sorted(signals, key=lambda x: x.metadata['composite_score'], reverse=True)
        filtered_signals = [s for s in ranked_signals if s.strength.value >= SignalStrength.MODERATE.value and s.confidence >= 60]
        return filtered_signals[:20] 
    
    def _validate_data(self, df: pd.DataFrame):
        required_columns = [
            'Time_dt', 'StandardOptionSymbol', 'TradeQuantity', 'Trade_Price', 
            'Aggressor', 'IV', 'Delta', 'Underlying_Price', 'NotionalValue',
            # Added columns that are implicitly used or good to validate for robustness
            'Option_Ask', 'Option_Bid', 'Exchange', 
            'Expiration_Date', 'Strike_Price', 'Option_Type' 
        ]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns for AlphaExtractor: {missing}. Available columns: {df.columns.tolist()}")

# generate_signal_report (No changes needed, uses AlphaSignal object)
def generate_signal_report(signals: list[AlphaSignal]) -> str:
    if not signals: return "No significant alpha signals detected."
    report = "=== ALPHA SIGNALS REPORT ===\n\n"
    for i, signal in enumerate(signals[:10], 1):
        report += f"{i}. {signal.signal_type.value} - {signal.direction}\n"
        report += f"   Symbol: {signal.option_symbol}\n"
        report += f"   Strength: {signal.strength.name} | Confidence: {signal.confidence:.1f}%\n"
        report += f"   Entry: ${signal.entry_price:.2f} | Notional: ${signal.notional_value:,.0f}\n"
        report += f"   Smart Money Score: {signal.smart_money_score:.0f} | Urgency: {signal.urgency_score:.0f}\n"
        report += f"   \U0001F4CA {signal.trade_recommendation}\n" # Chart emoji
        report += "-" * 50 + "\n"
    return report