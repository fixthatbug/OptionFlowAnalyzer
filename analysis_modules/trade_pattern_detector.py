# analysis_modules/trade_pattern_detector.py
# Functions for identifying specific trading patterns like blocks, strategies, and sweeps.

import pandas as pd
import numpy as np
from datetime import timedelta
import config # Assuming config.py is in a path accessible by this module
import rtd_handler # For format_tos_rtd_symbol

def analyze_block_trades(df: pd.DataFrame, qty_threshold=None, notional_threshold=None): # Added notional_threshold
    """
    Identifies block trades from the DataFrame based on quantity OR notional value.
    Expects df to have 'Time', 'TradeQuantity', and 'NotionalValue' columns.
    """
    if qty_threshold is None: 
        qty_threshold = config.LARGE_TRADE_THRESHOLD_QTY
    # Add a new config for notional value threshold or use a default
    if notional_threshold is None:
        notional_threshold = getattr(config, 'LARGE_TRADE_NOTIONAL_THRESHOLD', 200000) # Default if not in config

    if df.empty: 
        return pd.DataFrame()
    
    df_c = df.copy()

    # Ensure required columns are numeric
    df_c['TradeQuantity'] = pd.to_numeric(df_c.get('TradeQuantity'), errors='coerce')
    df_c['NotionalValue'] = pd.to_numeric(df_c.get('NotionalValue'), errors='coerce') # NotionalValue from data_utils.py

    # Default fallback if config value is invalid or not provided for quantity
    numeric_qty_threshold = pd.to_numeric(qty_threshold, errors='coerce')
    if pd.isna(numeric_qty_threshold): 
        numeric_qty_threshold = 50 
    
    numeric_notional_threshold = pd.to_numeric(notional_threshold, errors='coerce')
    if pd.isna(numeric_notional_threshold):
        numeric_notional_threshold = 200000

    # Identify blocks by either quantity or notional value
    is_block_by_qty = df_c['TradeQuantity'] >= numeric_qty_threshold
    is_block_by_notional = df_c['NotionalValue'] >= numeric_notional_threshold
    
    block_trades = df_c[is_block_by_qty | is_block_by_notional].copy() # OR condition

    if block_trades.empty: 
        return pd.DataFrame()
    
    # Define columns to include in the output, ensuring StandardOptionSymbol is present
    cols_to_include = [
        'Time', 'StandardOptionSymbol', 'Option_Description_orig', 'TradeQuantity', 'Trade_Price', 'NotionalValue', 
        'Aggressor', 'TradeDirection', 'Underlying_Price', 'IV', 'Delta', 'Exchange', 
        'Option_Type', 'Strike_Price', 'Expiration_Date', 'Condition'
    ]
    
    # Ensure all listed columns exist in block_trades, adding them as NaN if not
    for col in cols_to_include:
        if col not in block_trades.columns:
            block_trades[col] = np.nan # Or some other appropriate default

    # Filter for existing columns to prevent errors if some are still missing after the above step
    existing_cols_in_block_trades = [c for c in cols_to_include if c in block_trades.columns]
    block_trades_out = block_trades[existing_cols_in_block_trades]
        
    # Sort by NotionalValue as primary, then TradeQuantity
    return block_trades_out.sort_values(by=['NotionalValue', 'TradeQuantity'], ascending=[False, False])

def identify_common_strategies_from_blocks(block_trades_df: pd.DataFrame, active_ticker: str):
    """
    Identifies common strategies (Straddles, Strangles) from a DataFrame of block trades.
    Now uses 'StandardOptionSymbol' if available, otherwise 'Option_Description_orig'.
    """
    identified_strategies = []
    if block_trades_df.empty or block_trades_df.shape[0] < 2:
        return identified_strategies, block_trades_df

    # Ensure essential columns are present and correctly typed
    # StandardOptionSymbol is the preferred identifier
    id_col = 'StandardOptionSymbol' if 'StandardOptionSymbol' in block_trades_df.columns else 'Option_Description_orig'

    required_cols_for_strat = ['Time', id_col, 'Expiration_Date', 'Option_Type', 'Strike_Price', 
                               'TradeQuantity', 'Exchange', 'Condition', 'Trade_Price']
    for col in required_cols_for_strat:
        if col not in block_trades_df.columns:
            # This function operates on block_trades_df, if a col is missing, it's an upstream issue.
            # For now, return empty if critical components are missing.
            return identified_strategies, block_trades_df


    block_trades_df['Time'] = pd.to_datetime(block_trades_df['Time'], errors='coerce')
    block_trades_df['Expiration_Date'] = pd.to_datetime(block_trades_df['Expiration_Date'], errors='coerce')
    block_trades_df['Strike_Price'] = pd.to_numeric(block_trades_df['Strike_Price'], errors='coerce')
    block_trades_df['TradeQuantity'] = pd.to_numeric(block_trades_df['TradeQuantity'], errors='coerce')
    block_trades_df['Trade_Price'] = pd.to_numeric(block_trades_df['Trade_Price'], errors='coerce')


    block_trades_df.dropna(subset=['Time', 'Expiration_Date', 'Option_Type', 'Strike_Price', 'TradeQuantity'], inplace=True)
    if block_trades_df.empty:
        return identified_strategies, block_trades_df

    # Filter for 'Spread' condition, then sort
    # The current logic only looks for straddles/strangles within trades marked as 'Spread' AND also being block trades.
    # This might be too restrictive. Consider if strategies should be identified more broadly.
    # For now, sticking to the existing structure of this function.
    spread_condition_trades = block_trades_df[block_trades_df['Condition'] == 'Spread'].sort_values(by=['Time', id_col])
    
    processed_indices = set() 

    for i, trade1 in spread_condition_trades.iterrows():
        if i in processed_indices:
            continue

        for j, trade2 in spread_condition_trades.iterrows():
            if j <= i or j in processed_indices: 
                continue

            time_diff_seconds = abs((trade1['Time'] - trade2['Time']).total_seconds())
            
            trade1_qty = trade1['TradeQuantity'] # Already numeric
            trade2_qty = trade2['TradeQuantity'] # Already numeric

            # Check if underlying symbols match (important if id_col is StandardOptionSymbol)
            # Assumes StandardOptionSymbol format: UNDERLYING_YYMMDD_TYPE_STRIKE
            underlying1 = trade1[id_col].split('_')[0] if id_col == 'StandardOptionSymbol' else active_ticker
            underlying2 = trade2[id_col].split('_')[0] if id_col == 'StandardOptionSymbol' else active_ticker


            if (time_diff_seconds <= config.STRATEGY_TIME_WINDOW_SECONDS and 
                underlying1 == underlying2 and # Ensure same underlying
                trade1['Expiration_Date'] == trade2['Expiration_Date'] and
                trade1['Option_Type'] != trade2['Option_Type'] and # Must be Call and Put
                trade1_qty == trade2_qty and 
                trade1.get('Exchange') == trade2.get('Exchange')): # Optional: same exchange condition?
                
                strategy_details = {
                    'time': trade1['Time'], 
                    'legs_original_data': [], # Store original trade data for legs
                    'total_quantity': trade1_qty,
                    'combined_premium': None, 
                    'underlying_at_trade': trade1.get('Underlying_Price'),
                    'active_ticker': active_ticker, # Or determined from StandardOptionSymbol
                    'condition': trade1.get('Condition'),
                    'symbol': f"Strategy_{trade1[id_col]}_{trade2[id_col]}", # A placeholder symbol for the strategy
                    'is_complex_strategy': True # Flag this as a strategy
                }
                
                leg1_price = trade1['Trade_Price']
                leg2_price = trade2['Trade_Price']

                if pd.notna(leg1_price) and pd.notna(leg2_price):
                    strategy_details['combined_premium'] = (leg1_price * trade1_qty) + (leg2_price * trade2_qty) # Total premium for strategy qty

                call_leg_data, put_leg_data = (trade1, trade2) if trade1['Option_Type'] == 'C' else (trade2, trade1)
                
                # Store relevant details for each leg, not the full dict to avoid deep copies and large objects if not needed
                strategy_details['legs_original_data'].append(call_leg_data.to_dict()) # Storing dicts might be verbose; consider specific fields
                strategy_details['legs_original_data'].append(put_leg_data.to_dict())
                
                # Add simplified leg info for briefing
                strategy_details['call_leg_symbol'] = call_leg_data[id_col]
                strategy_details['put_leg_symbol'] = put_leg_data[id_col]
                strategy_details['call_strike'] = float(call_leg_data['Strike_Price'])
                strategy_details['put_strike'] = float(put_leg_data['Strike_Price'])


                if strategy_details['call_strike'] == strategy_details['put_strike']:
                    strategy_details['strategy_type'] = "Straddle"
                    strategy_details['description'] = (
                        f"{int(strategy_details['total_quantity'])} lot {underlying1} " # Use determined underlying
                        f"{call_leg_data['Expiration_Date'].strftime('%d%b%y').upper()} "
                        f"{strategy_details['call_strike']:.2f} Straddle" 
                    )
                else: 
                    strategy_details['strategy_type'] = "Strangle" 
                    strategy_details['description'] = (
                        f"{int(strategy_details['total_quantity'])} lot {underlying1} " # Use determined underlying
                        f"{call_leg_data['Expiration_Date'].strftime('%d%b%y').upper()} "
                        f"{strategy_details['put_strike']:.2f}P/{strategy_details['call_strike']:.2f}C Strangle" 
                    )
                
                identified_strategies.append(strategy_details)
                processed_indices.add(i); processed_indices.add(j)
                break # Found a pair for trade1, move to next trade1
    
    remaining_block_trades_df = block_trades_df[~block_trades_df.index.isin(processed_indices)]
    return identified_strategies, remaining_block_trades_df

def detect_sweep_orders(cleaned_df: pd.DataFrame, active_ticker: str, log_status_to_ui_func):
    """
    Detects sweep orders from the cleaned DataFrame.
    Uses 'StandardOptionSymbol' for identifying same options and 'TradeDirection' for aggressor side.
    """
    if cleaned_df.empty:
        log_status_to_ui_func(f"Sweep Detection: Input DataFrame empty for {active_ticker}.")
        return []

    df = cleaned_df.copy()
    
    # Use StandardOptionSymbol if available, else Option_Description_orig
    id_col = 'StandardOptionSymbol' if 'StandardOptionSymbol' in df.columns else 'Option_Description_orig'

    required_cols = ['Time', id_col, 'Expiration_Date', 'Strike_Price', 
                     'Option_Type', 'TradeQuantity', 'Trade_Price', 'Exchange', 
                     'Aggressor', 'TradeDirection'] # Added TradeDirection
    for col in required_cols:
        if col not in df.columns:
            log_status_to_ui_func(f"Sweep Detection: Missing required column '{col}' for {active_ticker}.")
            return []
            
    # Ensure data types (some might have been done in data_utils, but good to re-verify or ensure for this specific function)
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['TradeQuantity'] = pd.to_numeric(df['TradeQuantity'], errors='coerce')
    df['Trade_Price'] = pd.to_numeric(df['Trade_Price'], errors='coerce')
    # Strike_Price and Expiration_Date should be correctly typed from data_utils if StandardOptionSymbol is used

    df.dropna(subset=['Time', id_col, 'TradeQuantity', 'Trade_Price', 'Exchange', 'Aggressor', 'TradeDirection'], inplace=True)
    if df.empty:
        log_status_to_ui_func(f"Sweep Detection: DataFrame empty after Time/Numeric conversion & NaN drop for {active_ticker}.")
        return []

    df.sort_values(by=[id_col, 'Time'], inplace=True)
    
    # Filter for aggressive trades using TradeDirection
    # TradeDirection: 1 for Buy, -1 for Sell, 0 for Neutral
    aggressive_trades_df = df[df['TradeDirection'] != 0].copy() 
    
    if aggressive_trades_df.empty:
        log_status_to_ui_func(f"Sweep Detection: No aggressive trades (based on TradeDirection) found for {active_ticker}.")
        return []

    detected_sweeps_list = []
    # Ensure config values are correctly accessed (e.g., config.SWEEP_MAX_TIME_DIFF_MS)
    max_time_diff = pd.Timedelta(milliseconds=getattr(config, 'SWEEP_MAX_TIME_DIFF_MS', 500))
    min_trades_for_sweep = getattr(config, 'SWEEP_MIN_TRADES', 3)
    min_exchanges_for_sweep = getattr(config, 'SWEEP_MIN_EXCHANGES', 2)

    # Group by the chosen ID column (StandardOptionSymbol or Option_Description_orig)
    for option_identifier_val, group in aggressive_trades_df.groupby(id_col):
        trades_for_option = group.to_dict('records')
        num_trades = len(trades_for_option)
        i = 0
        while i < num_trades:
            # Start a potential sweep with the current trade
            # The aggressor side is determined by TradeDirection of the first trade
            current_aggressor_direction = trades_for_option[i]['TradeDirection']
            
            current_sweep_trades = [trades_for_option[i]]
            involved_exchanges = {trades_for_option[i]['Exchange']}
            
            first_trade_in_sweep = trades_for_option[i]

            j = i + 1
            while j < num_trades:
                next_trade = trades_for_option[j]
                time_difference = next_trade['Time'] - current_sweep_trades[-1]['Time']

                # Check for same aggressor direction and time window
                if time_difference <= max_time_diff and \
                   next_trade['TradeDirection'] == current_aggressor_direction:
                    current_sweep_trades.append(next_trade)
                    involved_exchanges.add(next_trade['Exchange'])
                else:
                    # Break if time diff is too large OR aggressor side changes
                    break 
                j += 1

            # Check if the collected sequence qualifies as a sweep
            if len(current_sweep_trades) >= min_trades_for_sweep and \
               len(involved_exchanges) >= min_exchanges_for_sweep:
                
                start_time = current_sweep_trades[0]['Time']
                end_time = current_sweep_trades[-1]['Time']
                total_quantity = sum(pd.to_numeric(trade['TradeQuantity'], errors='coerce') for trade in current_sweep_trades)
                total_notional_value = sum(pd.to_numeric(trade['TradeQuantity'], errors='coerce') * pd.to_numeric(trade['Trade_Price'], errors='coerce') for trade in current_sweep_trades)
                average_price = (total_notional_value / total_quantity) if total_quantity > 0 else 0.0
                aggressor_side_brief = 'Buy' if current_aggressor_direction == 1 else 'Sell'
                
                # Use StandardOptionSymbol for tos_symbol if it was used as id_col
                # Otherwise, try to format it if Option_Description_orig was used
                tos_option_symbol_for_sweep = option_identifier_val # Default to the group key

                if id_col == 'Option_Description_orig': # If grouped by original desc, try to get components for ToS symbol
                    exp_date_sweep = pd.Timestamp(first_trade_in_sweep.get('Expiration_Date')) if pd.notna(first_trade_in_sweep.get('Expiration_Date')) else pd.NaT
                    strike_sweep = float(first_trade_in_sweep.get('Strike_Price')) if pd.notna(first_trade_in_sweep.get('Strike_Price')) else np.nan
                    opt_type_sweep = str(first_trade_in_sweep.get('Option_Type')) if pd.notna(first_trade_in_sweep.get('Option_Type')) else None
                    underlying_for_sweep = first_trade_in_sweep.get('ParsedUnderlying', active_ticker) # Get ParsedUnderlying if available

                    if all(pd.notna(val) for val in [exp_date_sweep, strike_sweep, opt_type_sweep]) and opt_type_sweep is not None:
                        formatted_tos = rtd_handler.format_tos_rtd_symbol(
                            underlying_for_sweep, exp_date_sweep, strike_sweep, opt_type_sweep
                        )
                        if formatted_tos: tos_option_symbol_for_sweep = formatted_tos
                
                sweep_info = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'option_symbol': option_identifier_val, # This is the key used for grouping (e.g. StandardOptionSymbol)
                    'tos_symbol': tos_option_symbol_for_sweep, # This is the .TICKER... symbol for RTD
                    'total_quantity': total_quantity,
                    'average_price': average_price,
                    'exchanges_involved': sorted(list(involved_exchanges)),
                    'aggressor_side': aggressor_side_brief,
                    # 'aggressor_detail': first_trade_in_sweep['Aggressor'], # Original Aggressor string
                    'number_of_legs': len(current_sweep_trades),
                    'underlying_at_start': first_trade_in_sweep.get('Underlying_Price', np.nan) # Add underlying price at sweep start
                }
                # Add option characteristics if id_col was StandardOptionSymbol (can parse from it)
                # Or if they were directly available on first_trade_in_sweep
                sweep_info['strike_price'] = first_trade_in_sweep.get('Strike_Price', np.nan)
                sweep_info['expiration_date'] = first_trade_in_sweep.get('Expiration_Date', pd.NaT)
                sweep_info['option_type'] = first_trade_in_sweep.get('Option_Type', None)
                sweep_info['initial_iv'] = first_trade_in_sweep.get('IV', np.nan) # IV at start of sweep
                sweep_info['initial_delta'] = first_trade_in_sweep.get('Delta', np.nan) # Delta at start of sweep


                detected_sweeps_list.append(sweep_info)
                i = j # Move past the trades included in this sweep
            else:
                i += 1 # Move to the next trade to start a new potential sweep
                
    if detected_sweeps_list:
        log_status_to_ui_func(f"Sweep Detection: Found {len(detected_sweeps_list)} potential sweeps for {active_ticker}.")
    else:
        log_status_to_ui_func(f"Sweep Detection: No sweeps met all criteria for {active_ticker}.")
        
    return detected_sweeps_list
