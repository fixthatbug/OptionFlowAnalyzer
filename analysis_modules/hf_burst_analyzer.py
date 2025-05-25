# analysis_modules/hf_burst_analyzer.py
# Contains functions for High-Frequency (HF) burst analysis.

import pandas as pd
import numpy as np
from datetime import timedelta, datetime # Added datetime for DTE calculation
import config
import rtd_handler 
from analysis_modules.flow_calculator import hf_calculate_ofi_agg 
from analysis_modules.analysis_helpers import categorize_dte # For DTE calculation

def analyze_high_frequency_flow(df_input: pd.DataFrame, 
                                time_resample_freq=config.HF_TIME_RESAMPLE_FREQ, 
                                rolling_window_periods=config.HF_ROLLING_WINDOW_FOR_STATS, 
                                z_score_ofi=config.HF_Z_SCORE_THRESHOLD_OFI, 
                                z_score_agg=config.HF_Z_SCORE_THRESHOLD_AGG, 
                                min_duration_windows=config.HF_MIN_BURST_DURATION_WINDOWS, 
                                min_ofi_val_abs=config.HF_MIN_OFI_VALUE_FOR_BURST_ABS, 
                                min_agg_qty_abs=config.HF_MIN_AGG_QTY_FOR_BURST_ABS):
    """
    Analyzes high-frequency flow to detect bursts in OFI and Aggression,
    and extracts detailed information about each burst.
    """
    active_ticker_for_hf = df_input.attrs.get('active_ticker', 'TICKER_HF_UNKNOWN')
    log_status_to_ui_func = df_input.attrs.get('log_status_to_ui_func_from_main', lambda msg: print(f"HF Log ({active_ticker_for_hf}): {msg}"))

    if df_input.empty or not isinstance(df_input.index, pd.DatetimeIndex):
        log_status_to_ui_func(f"Input DataFrame is empty or not time-indexed.")
        return pd.DataFrame(), pd.DataFrame(), [] 

    # Ensure df_input has 'Time' as a column if it's also the index, for easier filtering later
    df_full_trades = df_input.copy()
    if 'Time' not in df_full_trades.columns and isinstance(df_full_trades.index, pd.DatetimeIndex) and df_full_trades.index.name == 'Time':
        df_full_trades['Time'] = df_full_trades.index


    # Resampling data preparation
    df_for_resample = df_input.copy()
    for col in ['TradeQuantity', 'Trade_Price']: # Ensure these are numeric for hf_calculate_ofi_agg
        if col in df_for_resample.columns: 
            df_for_resample[col] = pd.to_numeric(df_for_resample[col], errors='coerce')
        else: 
            log_status_to_ui_func(f"Resample Prep: Missing essential column '{col}'.")
            return pd.DataFrame(), pd.DataFrame(), []
    df_for_resample.dropna(subset=['TradeQuantity', 'Trade_Price', 'Aggressor'], inplace=True)
    if df_for_resample.empty: 
        log_status_to_ui_func(f"Resample Prep: DataFrame empty after NaN drop.")
        return pd.DataFrame(), pd.DataFrame(), []

    try:
        hf_metrics = df_for_resample.resample(time_resample_freq).apply(hf_calculate_ofi_agg)
    except Exception as e: 
        log_status_to_ui_func(f"Resampling error: {e}")
        return pd.DataFrame(), pd.DataFrame(), []
    hf_metrics = hf_metrics.fillna(0)

    # Z-Score Calculations (same as before)
    hf_metrics['OFI_Mean'] = hf_metrics['NetDollarOFI'].rolling(window=rolling_window_periods, min_periods=max(1, rolling_window_periods // 2)).mean().fillna(0)
    hf_metrics['OFI_Std'] = hf_metrics['NetDollarOFI'].rolling(window=rolling_window_periods, min_periods=max(1, rolling_window_periods // 2)).std().fillna(0)
    hf_metrics['OFI_ZScore'] = np.where(hf_metrics['OFI_Std'] == 0, 0, (hf_metrics['NetDollarOFI'] - hf_metrics['OFI_Mean']) / hf_metrics['OFI_Std'])
    hf_metrics['OFI_Burst_Signal'] = (abs(hf_metrics['OFI_ZScore']) >= z_score_ofi) & (abs(hf_metrics['NetDollarOFI']) >= min_ofi_val_abs)

    hf_metrics['Agg_Mean'] = hf_metrics['NetAggressorQty'].rolling(window=rolling_window_periods, min_periods=max(1, rolling_window_periods // 2)).mean().fillna(0)
    hf_metrics['Agg_Std'] = hf_metrics['NetAggressorQty'].rolling(window=rolling_window_periods, min_periods=max(1, rolling_window_periods // 2)).std().fillna(0)
    hf_metrics['Agg_ZScore'] = np.where(hf_metrics['Agg_Std'] == 0, 0, (hf_metrics['NetAggressorQty'] - hf_metrics['Agg_Mean']) / hf_metrics['Agg_Std'])
    hf_metrics['Agg_Burst_Signal'] = (abs(hf_metrics['Agg_ZScore']) >= z_score_agg) & (abs(hf_metrics['NetAggressorQty']) >= min_agg_qty_abs)

    burst_events_list = []
    # This hf_options_details_list is for the overall summary of options active in ANY burst
    # It can be populated from the enriched per-burst data if needed, or kept as is.
    # For now, let's focus on enriching burst_events_list.
    hf_options_summary_list = [] 


    def find_burst_sequences(signal_series, value_series, burst_type_label, current_active_ticker_local):
        nonlocal hf_options_summary_list # To append overall top options from bursts

        signal_series_shifted = signal_series.shift(1, fill_value=False)
        starts = signal_series.index[signal_series & ~signal_series_shifted]
        ends = signal_series.index[~signal_series & signal_series_shifted]
        if not signal_series.empty and signal_series.iloc[-1]: 
            if not ends.empty and ends[-1] == signal_series.index[-1]:
                pass
            else:
                ends = ends.append(pd.Index([signal_series.index[-1]]))
            
        for start_time_burst in starts: # Renamed to avoid conflict
            potential_ends = ends[ends >= start_time_burst]
            if not potential_ends.empty:
                end_time_burst = potential_ends[0]
                burst_period_values = value_series.loc[start_time_burst:end_time_burst]
                duration_windows = len(burst_period_values)

                if duration_windows >= min_duration_windows:
                    peak_value_abs = burst_period_values.abs().max()
                    peak_value_signed = burst_period_values[burst_period_values.abs() == peak_value_abs].iloc[0] if not burst_period_values[burst_period_values.abs() == peak_value_abs].empty else 0.0
                    
                    total_value_in_burst = burst_period_values.sum()
                    try: 
                        duration_seconds = (end_time_burst - start_time_burst).total_seconds() + pd.Timedelta(time_resample_freq).total_seconds()
                        if duration_seconds < 0 : duration_seconds = pd.Timedelta(time_resample_freq).total_seconds() 
                    except Exception: 
                        duration_seconds = duration_windows * pd.Timedelta(time_resample_freq).total_seconds() # Fallback
                    
                    # --- Enhanced Detail Extraction for THIS Burst ---
                    trades_in_current_burst = df_full_trades[
                        (df_full_trades['Time'] >= start_time_burst) & 
                        (df_full_trades['Time'] <= end_time_burst + pd.Timedelta(milliseconds=1)) # Inclusive of end_time_burst window
                    ].copy()
                    
                    ul_price_start = trades_in_current_burst['Underlying_Price'].iloc[0] if not trades_in_current_burst.empty else np.nan
                    ul_price_end = trades_in_current_burst['Underlying_Price'].iloc[-1] if not trades_in_current_burst.empty else np.nan
                    ul_price_min = trades_in_current_burst['Underlying_Price'].min() if not trades_in_current_burst.empty else np.nan
                    ul_price_max = trades_in_current_burst['Underlying_Price'].max() if not trades_in_current_burst.empty else np.nan

                    top_options_details_in_burst = []
                    if not trades_in_current_burst.empty:
                        # Ensure necessary columns are present for grouping and calculations
                        required_cols_for_burst_opts = ['Option_Description_orig', 'Expiration_Date', 'Strike_Price', 'Option_Type', 'TradeQuantity', 'NetAggVolumeTrade', 'IV', 'Delta', 'Time']
                        if not all(col in trades_in_current_burst.columns for col in required_cols_for_burst_opts):
                            log_status_to_ui_func(f"Burst Detail: Missing one or more required columns for top option analysis in burst at {start_time_burst}")
                        else:
                            # Calculate Notional Value for sorting top options
                            trades_in_current_burst['NotionalValue'] = pd.to_numeric(trades_in_current_burst['TradeQuantity'], errors='coerce') * pd.to_numeric(trades_in_current_burst['Trade_Price'], errors='coerce')
                            
                            grouped_by_option = trades_in_current_burst.groupby(['Option_Description_orig', 'Expiration_Date', 'Strike_Price', 'Option_Type'])
                            
                            top_options_data = grouped_by_option.agg(
                                TotalQuantityInBurst=('TradeQuantity', 'sum'),
                                NotionalValueInBurst=('NotionalValue', 'sum'),
                                NetAggQtyInBurst=('NetAggVolumeTrade', 'sum'),
                                AvgIVInBurst=('IV', 'mean'),
                                AvgDeltaInBurst=('Delta', 'mean'),
                                FirstTradeTimeInBurst=('Time', 'min'),
                                LastTradeTimeInBurst=('Time', 'max')
                            ).sort_values(by='NotionalValueInBurst', ascending=False).head(config.HF_BURST_TOP_N_OPTIONS_PER_BURST)

                            for idx_tuple, opt_burst_summary in top_options_data.iterrows():
                                opt_desc_orig, exp_date, strike, opt_type = idx_tuple
                                
                                # Get IV at first and last trade of this option within the burst
                                option_trades_in_burst = trades_in_current_burst[
                                    (trades_in_current_burst['Option_Description_orig'] == opt_desc_orig) &
                                    (trades_in_current_burst['Expiration_Date'] == exp_date) &
                                    (trades_in_current_burst['Strike_Price'] == strike) &
                                    (trades_in_current_burst['Option_Type'] == opt_type)
                                ].sort_values(by='Time')

                                iv_start_trade = np.nan
                                iv_end_trade = np.nan
                                iv_change_in_burst = np.nan

                                if not option_trades_in_burst.empty:
                                    iv_start_trade = option_trades_in_burst['IV'].iloc[0]
                                    if len(option_trades_in_burst) > 1:
                                        iv_end_trade = option_trades_in_burst['IV'].iloc[-1]
                                        if pd.notna(iv_start_trade) and pd.notna(iv_end_trade):
                                            iv_change_in_burst = iv_end_trade - iv_start_trade
                                    elif pd.notna(iv_start_trade) : # Only one trade for this option in burst
                                        iv_end_trade = iv_start_trade


                                tos_sym = rtd_handler.format_tos_rtd_symbol(current_active_ticker_local, exp_date, strike, opt_type)
                                if tos_sym is None: tos_sym = opt_desc_orig
                                
                                dte_val = (pd.Timestamp(exp_date) - pd.Timestamp(start_time_burst)).days if pd.notna(exp_date) else "N/A"


                                top_options_details_in_burst.append({
                                    'tos_symbol': tos_sym,
                                    'strike': strike,
                                    'type': opt_type,
                                    'dte_at_burst': dte_val,
                                    'total_qty_in_burst': opt_burst_summary['TotalQuantityInBurst'],
                                    'net_agg_qty_in_burst': opt_burst_summary['NetAggQtyInBurst'],
                                    'avg_iv_in_burst': opt_burst_summary['AvgIVInBurst'],
                                    'iv_start_trade': iv_start_trade,
                                    'iv_end_trade': iv_end_trade,
                                    'iv_change_in_burst': iv_change_in_burst,
                                    'avg_delta_in_burst': opt_burst_summary['AvgDeltaInBurst']
                                })
                                
                                # Also populate the overall hf_options_summary_list for options_of_interest
                                hf_options_summary_list.append({
                                    'symbol': tos_sym,
                                    'reason': f"Active in HF {burst_type_label} {('Buying' if peak_value_signed > 0 else 'Selling')} Burst",
                                    'metric': opt_burst_summary['TotalQuantityInBurst'], # Or NetAggQtyInBurst
                                    'time': start_time_burst,
                                    'agg': f"Burst NetAggQty: {opt_burst_summary['NetAggQtyInBurst']}",
                                    'is_complex_strategy': False 
                                })
                    # --- End Enhanced Detail Extraction ---

                    burst_events_list.append({
                        'StartTime': start_time_burst, 
                        'EndTime': end_time_burst, 
                        'DurationWindows': duration_windows,
                        'DurationSeconds': duration_seconds, 
                        'BurstType': burst_type_label,
                        'PeakValue': peak_value_signed, 
                        'TotalValueInBurst': total_value_in_burst,
                        'Direction': 'Buying' if peak_value_signed > 0 else ('Selling' if peak_value_signed < 0 else 'Neutral'),
                        'UnderlyingPrice_Start': ul_price_start,
                        'UnderlyingPrice_End': ul_price_end,
                        'UnderlyingPrice_MinInBurst': ul_price_min,
                        'UnderlyingPrice_MaxInBurst': ul_price_max,
                        'TopOptionsInBurst': top_options_details_in_burst # Add the new detailed list
                    })
    
    find_burst_sequences(hf_metrics['OFI_Burst_Signal'], hf_metrics['NetDollarOFI'], 'NetDollarOFI', active_ticker_for_hf)
    find_burst_sequences(hf_metrics['Agg_Burst_Signal'], hf_metrics['NetAggressorQty'], 'NetAggressorQty', active_ticker_for_hf)

    identified_bursts_df = pd.DataFrame(burst_events_list)
    if not identified_bursts_df.empty:
        identified_bursts_df = identified_bursts_df.sort_values(by='StartTime').reset_index(drop=True)

    # Deduplicate and take top N for the overall hf_options_summary_list
    if hf_options_summary_list:
        temp_df_hf_opts = pd.DataFrame(hf_options_summary_list)
        # Sum metrics for options appearing in multiple bursts if needed, or just take first appearance by time then sort by metric
        # For simplicity, let's group by symbol and aggregate metrics, then sort
        final_hf_options_summary = temp_df_hf_opts.groupby('symbol').agg(
            reason=('reason', 'first'), # Take first reason
            metric=('metric', 'sum'),   # Sum total quantity across all bursts it appeared in
            time=('time', 'min'),       # Time of first burst it appeared in
            agg=('agg', 'first'),       # Example, could be more complex aggregation
            is_complex_strategy=('is_complex_strategy', 'first')
        ).sort_values(by='metric', ascending=False).head(config.HF_BURST_TOP_N_OPTIONS).reset_index() # Use overall top N
        hf_options_summary_list = final_hf_options_summary.to_dict('records')


    return hf_metrics, identified_bursts_df, hf_options_summary_list
