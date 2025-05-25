# analysis_modules/trade_pattern_detector.py
# Functions for identifying specific trading patterns like blocks, strategies, and sweeps.

import pandas as pd
import numpy as np
from datetime import timedelta
import config
import rtd_handler # For format_tos_rtd_symbol

def analyze_block_trades(df: pd.DataFrame, threshold=None): # Changed default to None
    """
    Identifies block trades from the DataFrame.
    Expects df to have 'Time' as a regular column.
    """
    if threshold is None: # Access config inside the function
        threshold = config.LARGE_TRADE_THRESHOLD_QTY

    if df.empty or 'TradeQuantity' not in df.columns: 
        return pd.DataFrame()
    
    df_c = df.copy()
    df_c['TradeQuantity'] = pd.to_numeric(df_c['TradeQuantity'], errors='coerce')
    
    numeric_threshold = pd.to_numeric(threshold, errors='coerce')
    if pd.isna(numeric_threshold): 
        numeric_threshold = 50 # Default fallback if config value is invalid or not provided

    block_trades = df_c[df_c['TradeQuantity'] >= numeric_threshold].copy()
    if block_trades.empty: 
        return pd.DataFrame()
    
    cols_to_include = ['Time', 'Option_Description_orig', 'TradeQuantity', 'Trade_Price', 'Aggressor',
                       'Underlying_Price', 'IV', 'Delta', 'Exchange', 'Option_Type', 'Strike_Price', 'Expiration_Date', 'Condition']
    
    block_trades_out = block_trades 
        
    existing_cols = [c for c in cols_to_include if c in block_trades_out.columns]
    return block_trades_out[existing_cols].sort_values(by='TradeQuantity', ascending=False)

def identify_common_strategies_from_blocks(block_trades_df: pd.DataFrame, active_ticker: str):
    """
    Identifies common strategies (Straddles, Strangles) from a DataFrame of block trades.
    Expects block_trades_df to have 'Time' as a regular column.
    """
    identified_strategies = []
    if block_trades_df.empty or block_trades_df.shape[0] < 2:
        return identified_strategies, block_trades_df

    block_trades_df['Time'] = pd.to_datetime(block_trades_df['Time'], errors='coerce')
    block_trades_df['Expiration_Date'] = pd.to_datetime(block_trades_df['Expiration_Date'], errors='coerce')
    # Ensure necessary columns for logic are not NaN after conversion
    block_trades_df.dropna(subset=['Time', 'Expiration_Date', 'Option_Type', 'Strike_Price', 'TradeQuantity', 'Exchange', 'Condition'], inplace=True)
    if block_trades_df.empty: # Check again after dropna
        return identified_strategies, block_trades_df


    spread_condition_trades = block_trades_df[block_trades_df['Condition'] == 'Spread'].sort_values(by=['Time', 'Option_Description_orig'])
    
    processed_indices = set() 

    for i, trade1 in spread_condition_trades.iterrows():
        if i in processed_indices:
            continue

        for j, trade2 in spread_condition_trades.iterrows():
            if j <= i or j in processed_indices: 
                continue

            time_diff_seconds = abs((trade1['Time'] - trade2['Time']).total_seconds())
            
            # Convert TradeQuantity to numeric for comparison, handling potential errors
            trade1_qty = pd.to_numeric(trade1['TradeQuantity'], errors='coerce')
            trade2_qty = pd.to_numeric(trade2['TradeQuantity'], errors='coerce')

            if pd.isna(trade1_qty) or pd.isna(trade2_qty): # Skip if quantities are not valid numbers
                continue

            if (time_diff_seconds <= config.STRATEGY_TIME_WINDOW_SECONDS and 
                trade1['Expiration_Date'] == trade2['Expiration_Date'] and
                trade1['Option_Type'] != trade2['Option_Type'] and 
                trade1_qty == trade2_qty and 
                trade1['Exchange'] == trade2['Exchange']): 
                
                strategy_details = {
                    'time': trade1['Time'], 'legs': [], 
                    'total_quantity': trade1_qty, # Use the numeric quantity
                    'combined_premium': None, 'underlying_at_trade': trade1.get('Underlying_Price'),
                    'active_ticker': active_ticker, 'condition': trade1.get('Condition') 
                }
                
                leg1_price = pd.to_numeric(trade1['Trade_Price'], errors='coerce')
                leg2_price = pd.to_numeric(trade2['Trade_Price'], errors='coerce')

                if pd.notna(leg1_price) and pd.notna(leg2_price):
                    strategy_details['combined_premium'] = leg1_price + leg2_price

                call_leg_data, put_leg_data = (trade1, trade2) if trade1['Option_Type'] == 'C' else (trade2, trade1)
                
                strategy_details['legs'].append(call_leg_data.to_dict())
                strategy_details['legs'].append(put_leg_data.to_dict())

                strike_price_call_f = float(call_leg_data['Strike_Price'])
                strike_price_put_f = float(put_leg_data['Strike_Price'])


                if strike_price_call_f == strike_price_put_f:
                    strategy_details['strategy_type'] = "Straddle"
                    strategy_details['description'] = (
                        f"{int(strategy_details['total_quantity'])} lot {active_ticker} "
                        f"{call_leg_data['Expiration_Date'].strftime('%d%b%y').upper()} "
                        f"{strike_price_call_f:.2f} Straddle" 
                    )
                else: 
                    strategy_details['strategy_type'] = "Strangle" 
                    strategy_details['description'] = (
                        f"{int(strategy_details['total_quantity'])} lot {active_ticker} "
                        f"{call_leg_data['Expiration_Date'].strftime('%d%b%y').upper()} "
                        f"{strike_price_put_f:.2f}P/{strike_price_call_f:.2f}C Strangle" 
                    )
                
                identified_strategies.append(strategy_details)
                processed_indices.add(i); processed_indices.add(j)
                break 
    
    remaining_block_trades_df = block_trades_df[~block_trades_df.index.isin(processed_indices)]
    return identified_strategies, remaining_block_trades_df

def detect_sweep_orders(cleaned_df: pd.DataFrame, active_ticker: str, log_status_to_ui_func):
    if cleaned_df.empty:
        log_status_to_ui_func(f"Sweep Detection: Input DataFrame empty for {active_ticker}.")
        return []

    df = cleaned_df.copy()
    required_cols = ['Time', 'Option_Description_orig', 'Expiration_Date', 'Strike_Price', 
                     'Option_Type', 'TradeQuantity', 'Trade_Price', 'Exchange', 'Aggressor']
    for col in required_cols:
        if col not in df.columns:
            log_status_to_ui_func(f"Sweep Detection: Missing required column '{col}' for {active_ticker}.")
            return []
            
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['TradeQuantity'] = pd.to_numeric(df['TradeQuantity'], errors='coerce')
    df['Trade_Price'] = pd.to_numeric(df['Trade_Price'], errors='coerce')
    df['Strike_Price'] = pd.to_numeric(df['Strike_Price'], errors='coerce')
    df['Expiration_Date'] = pd.to_datetime(df['Expiration_Date'], errors='coerce')

    df.dropna(subset=required_cols, inplace=True)
    if df.empty:
        log_status_to_ui_func(f"Sweep Detection: DataFrame empty after Time/Numeric conversion & NaN drop for {active_ticker}.")
        return []

    df.sort_values(by=['Option_Description_orig', 'Time'], inplace=True)
    
    aggressive_trades_df = df[df['Aggressor'].isin(['Buyer (At Ask)', 'Seller (At Bid)'])].copy()
    
    if aggressive_trades_df.empty:
        log_status_to_ui_func(f"Sweep Detection: No aggressive trades found for {active_ticker}.")
        return []

    detected_sweeps_list = []
    max_time_diff = pd.Timedelta(milliseconds=config.SWEEP_MAX_TIME_DIFF_MS)
    min_trades_for_sweep = config.SWEEP_MIN_TRADES
    min_exchanges_for_sweep = config.SWEEP_MIN_EXCHANGES

    for option_desc, group in aggressive_trades_df.groupby('Option_Description_orig'):
        trades_for_option = group.to_dict('records')
        num_trades = len(trades_for_option)
        i = 0
        while i < num_trades:
            current_sweep_trades = [trades_for_option[i]]
            current_aggressor_type = trades_for_option[i]['Aggressor']
            involved_exchanges = {trades_for_option[i]['Exchange']}
            
            first_trade_in_sweep = trades_for_option[i]
            exp_date = pd.Timestamp(first_trade_in_sweep.get('Expiration_Date')) if pd.notna(first_trade_in_sweep.get('Expiration_Date')) else pd.NaT
            strike = float(first_trade_in_sweep.get('Strike_Price')) if pd.notna(first_trade_in_sweep.get('Strike_Price')) else np.nan
            opt_type = str(first_trade_in_sweep.get('Option_Type')) if pd.notna(first_trade_in_sweep.get('Option_Type')) else None
            can_generate_tos_symbol = all(pd.notna(val) for val in [exp_date, strike, opt_type]) and opt_type is not None

            j = i + 1
            while j < num_trades:
                next_trade = trades_for_option[j]
                time_difference = next_trade['Time'] - current_sweep_trades[-1]['Time']

                if time_difference <= max_time_diff and \
                   next_trade['Aggressor'] == current_aggressor_type:
                    current_sweep_trades.append(next_trade)
                    involved_exchanges.add(next_trade['Exchange'])
                else:
                    break 
                j += 1

            if len(current_sweep_trades) >= min_trades_for_sweep and \
               len(involved_exchanges) >= min_exchanges_for_sweep:
                
                start_time = current_sweep_trades[0]['Time']
                end_time = current_sweep_trades[-1]['Time']
                total_quantity = sum(pd.to_numeric(trade['TradeQuantity'], errors='coerce') for trade in current_sweep_trades) # Ensure numeric sum
                total_notional_value = sum(pd.to_numeric(trade['TradeQuantity'], errors='coerce') * pd.to_numeric(trade['Trade_Price'], errors='coerce') for trade in current_sweep_trades) # Ensure numeric sum
                average_price = total_notional_value / total_quantity if total_quantity > 0 else 0.0
                aggressor_side_brief = 'Buy' if current_aggressor_type == 'Buyer (At Ask)' else 'Sell'
                
                tos_option_symbol = "N/A"
                if can_generate_tos_symbol:
                    tos_option_symbol = rtd_handler.format_tos_rtd_symbol(
                        active_ticker, exp_date, strike, opt_type
                    )
                    if tos_option_symbol is None: tos_option_symbol = option_desc 

                detected_sweeps_list.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'option_description_orig': option_desc,
                    'tos_symbol': tos_option_symbol,
                    'total_quantity': total_quantity,
                    'average_price': average_price,
                    'exchanges_involved': sorted(list(involved_exchanges)),
                    'aggressor_side': aggressor_side_brief,
                    'aggressor_detail': current_aggressor_type,
                    'number_of_legs': len(current_sweep_trades)
                })
                i = j 
            else:
                i += 1
    return detected_sweeps_list
