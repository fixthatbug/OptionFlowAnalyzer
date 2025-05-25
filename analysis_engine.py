# analysis_engine.py (Refactored as Orchestrator)
# Orchestrates options flow analysis by calling specialized modules.

import pandas as pd
import numpy as np
from datetime import datetime 
import config
import rtd_handler 

from analysis_modules.analysis_helpers import (
    categorize_moneyness, 
    categorize_dte, 
    get_sort_key 
)
from analysis_modules.flow_calculator import hf_calculate_ofi_agg
from analysis_modules.hf_burst_analyzer import analyze_high_frequency_flow
from analysis_modules.trade_pattern_detector import (
    analyze_block_trades,
    identify_common_strategies_from_blocks,
    detect_sweep_orders
)
from analysis_modules.report_generator import ( # Still needed for re-export if main_app calls these via analysis_engine
    generate_trade_briefing, 
    _get_dynamic_strategy_suggestions, 
    generate_detailed_txt_report
)

# --- Stance Determination Helper ---
def _calculate_market_stance_and_rationale(results: dict, active_ticker: str) -> tuple:
    """
    Calculates the market stance, conviction, and primary rationale based on analysis results.
    """
    bullish_score = 0
    bearish_score = 0
    rationale_components = []

    # Thresholds from config (ensure they are defined in config.py)
    ofi_strong_thresh = config.OFI_STRONG_THRESHOLD_DOLLAR
    ofi_mod_thresh = ofi_strong_thresh * 0.3
    agg_strong_thresh = config.AGGRESSION_STRONG_THRESHOLD_CONTRACTS
    agg_mod_thresh = agg_strong_thresh * 0.3
    sweep_qty_strong_thresh = config.SWEEP_STRONG_QTY_THRESHOLD # NEW - Add to config.py
    sweep_qty_mod_thresh = sweep_qty_strong_thresh * 0.3 # NEW - Add to config.py
    hf_burst_peak_strong_scalar_agg = 0.5 # Scalar for AGGRESSION_STRONG_THRESHOLD_CONTRACTS
    hf_burst_peak_mod_scalar_agg = 0.15
    hf_burst_peak_strong_scalar_ofi = 0.5 # Scalar for OFI_STRONG_THRESHOLD_DOLLAR
    hf_burst_peak_mod_scalar_ofi = 0.15


    # 1. OFI Analysis
    ofi_binned_summary = results.get('ofi_binned_summary', pd.DataFrame())
    net_ofi_sum_dollar = ofi_binned_summary['Net_OFI_Value'].sum() if not ofi_binned_summary.empty and 'Net_OFI_Value' in ofi_binned_summary.columns else 0.0
    
    if net_ofi_sum_dollar > ofi_strong_thresh:
        bullish_score += 2
        rationale_components.append(f"Very Strong Net OFI (${net_ofi_sum_dollar:,.0f})")
    elif net_ofi_sum_dollar > ofi_mod_thresh:
        bullish_score += 1
        rationale_components.append(f"Moderate Net OFI (${net_ofi_sum_dollar:,.0f})")
    elif net_ofi_sum_dollar < -ofi_strong_thresh:
        bearish_score += 2
        rationale_components.append(f"Very Strong Negative Net OFI (${net_ofi_sum_dollar:,.0f})")
    elif net_ofi_sum_dollar < -ofi_mod_thresh:
        bearish_score += 1
        rationale_components.append(f"Moderate Negative Net OFI (${net_ofi_sum_dollar:,.0f})")

    # 2. Net Aggressor Quantity
    net_agg_qty_overall = ofi_binned_summary['Net_Aggressor_Qty'].sum() if not ofi_binned_summary.empty and 'Net_Aggressor_Qty' in ofi_binned_summary.columns else 0.0
    if net_agg_qty_overall > agg_strong_thresh:
        bullish_score += 2
        rationale_components.append(f"Strong Net Aggressor Quantity ({int(net_agg_qty_overall)} contracts)")
    elif net_agg_qty_overall > agg_mod_thresh:
        bullish_score += 1
        rationale_components.append(f"Moderate Net Aggressor Quantity ({int(net_agg_qty_overall)} contracts)")
    elif net_agg_qty_overall < -agg_strong_thresh:
        bearish_score += 2
        rationale_components.append(f"Strong Negative Net Aggressor Quantity ({int(net_agg_qty_overall)} contracts)")
    elif net_agg_qty_overall < -agg_mod_thresh:
        bearish_score += 1
        rationale_components.append(f"Moderate Negative Net Aggressor Quantity ({int(net_agg_qty_overall)} contracts)")

    # 3. HF Burst Analysis
    hf_summary = results.get('hf_summary_stats', {})
    num_bull_hf_bursts = 0
    num_bear_hf_bursts = 0
    if hf_summary.get('strongest_agg_buy_burst_peak', 0) > agg_strong_thresh * hf_burst_peak_strong_scalar_agg or \
       hf_summary.get('strongest_ofi_buy_burst_peak', 0) > ofi_strong_thresh * hf_burst_peak_strong_scalar_ofi:
        bullish_score += 2
        rationale_components.append("Strong HF Buying Bursts")
        num_bull_hf_bursts = hf_summary.get('num_agg_bursts',0) + hf_summary.get('num_ofi_bursts',0) # Simplified count
    elif hf_summary.get('strongest_agg_buy_burst_peak', 0) > agg_mod_thresh * hf_burst_peak_mod_scalar_agg or \
         hf_summary.get('strongest_ofi_buy_burst_peak', 0) > ofi_mod_thresh * hf_burst_peak_mod_scalar_ofi:
        bullish_score += 1
        rationale_components.append("Moderate HF Buying Bursts")
        num_bull_hf_bursts = hf_summary.get('num_agg_bursts',0) + hf_summary.get('num_ofi_bursts',0)

    if hf_summary.get('strongest_agg_sell_burst_peak', 0) < -agg_strong_thresh * hf_burst_peak_strong_scalar_agg or \
       hf_summary.get('strongest_ofi_sell_burst_peak', 0) < -ofi_strong_thresh * hf_burst_peak_strong_scalar_ofi:
        bearish_score += 2
        rationale_components.append("Strong HF Selling Bursts")
        num_bear_hf_bursts = hf_summary.get('num_agg_bursts',0) + hf_summary.get('num_ofi_bursts',0) # Simplified
    elif hf_summary.get('strongest_agg_sell_burst_peak', 0) < -agg_mod_thresh * hf_burst_peak_mod_scalar_agg or \
         hf_summary.get('strongest_ofi_sell_burst_peak', 0) < -ofi_mod_thresh * hf_burst_peak_mod_scalar_ofi:
        bearish_score += 1
        rationale_components.append("Moderate HF Selling Bursts")
        num_bear_hf_bursts = hf_summary.get('num_agg_bursts',0) + hf_summary.get('num_ofi_bursts',0)
    
    # 4. Sweep Order Analysis
    identified_sweeps = results.get('identified_sweep_orders', [])
    net_buy_sweep_qty = sum(s['total_quantity'] for s in identified_sweeps if s['aggressor_side'] == 'Buy')
    net_sell_sweep_qty = sum(s['total_quantity'] for s in identified_sweeps if s['aggressor_side'] == 'Sell')

    if net_buy_sweep_qty > sweep_qty_strong_thresh:
        bullish_score += 2
        rationale_components.append(f"Strong Net Buy Sweep Volume ({int(net_buy_sweep_qty)} contracts)")
    elif net_buy_sweep_qty > sweep_qty_mod_thresh:
        bullish_score += 1
        rationale_components.append(f"Moderate Net Buy Sweep Volume ({int(net_buy_sweep_qty)} contracts)")
    
    if net_sell_sweep_qty > sweep_qty_strong_thresh: # Comparing absolute quantity for selling
        bearish_score += 2
        rationale_components.append(f"Strong Net Sell Sweep Volume ({int(net_sell_sweep_qty)} contracts)")
    elif net_sell_sweep_qty > sweep_qty_mod_thresh:
        bearish_score += 1
        rationale_components.append(f"Moderate Net Sell Sweep Volume ({int(net_sell_sweep_qty)} contracts)")

    # 5. OTM Aggression (from flow_by_moneyness)
    flow_by_moneyness = results.get('flow_by_moneyness', pd.DataFrame())
    if not flow_by_moneyness.empty and 'NetAggressiveVolumeQty' in flow_by_moneyness.columns:
        # OTM Call Buying
        if ('C', 'OTM') in flow_by_moneyness.index and flow_by_moneyness.loc[('C', 'OTM'), 'NetAggressiveVolumeQty'] > agg_mod_thresh * 0.5: # Adjusted threshold
            bullish_score += 1
            rationale_components.append(f"Noticeable OTM Call Buying Aggression ({int(flow_by_moneyness.loc[('C', 'OTM'), 'NetAggressiveVolumeQty'])} contracts)")
        # OTM Put Selling (Bullish)
        if ('P', 'OTM') in flow_by_moneyness.index and flow_by_moneyness.loc[('P', 'OTM'), 'NetAggressiveVolumeQty'] < -agg_mod_thresh * 0.25: # Selling Puts means negative AggQty for puts
            bullish_score += 1
            rationale_components.append(f"Noticeable OTM Put Selling Aggression ({int(abs(flow_by_moneyness.loc[('P', 'OTM'), 'NetAggressiveVolumeQty']))} contracts)")
        # OTM Put Buying
        if ('P', 'OTM') in flow_by_moneyness.index and flow_by_moneyness.loc[('P', 'OTM'), 'NetAggressiveVolumeQty'] > agg_mod_thresh * 0.5:
            bearish_score += 1
            rationale_components.append(f"Noticeable OTM Put Buying Aggression ({int(flow_by_moneyness.loc[('P', 'OTM'), 'NetAggressiveVolumeQty'])} contracts)")
        # OTM Call Selling (Bearish)
        if ('C', 'OTM') in flow_by_moneyness.index and flow_by_moneyness.loc[('C', 'OTM'), 'NetAggressiveVolumeQty'] < -agg_mod_thresh * 0.25:
            bearish_score += 1
            rationale_components.append(f"Noticeable OTM Call Selling Aggression ({int(abs(flow_by_moneyness.loc[('C', 'OTM'), 'NetAggressiveVolumeQty']))} contracts)")

    # Determine Stance
    stance = "NEUTRAL - OBSERVE"
    conviction = "Low"
    net_score = bullish_score - bearish_score

    # Define thresholds for stance based on net_score
    # These thresholds will need tuning.
    strong_conviction_threshold = 4 # Example
    moderate_conviction_threshold = 2 # Example

    if net_score >= strong_conviction_threshold:
        stance = "STRONG BULLISH"
        conviction = "High"
    elif net_score >= moderate_conviction_threshold:
        stance = "MODERATELY BULLISH"
        conviction = "Medium"
    elif net_score <= -strong_conviction_threshold:
        stance = "STRONG BEARISH"
        conviction = "High"
    elif net_score <= -moderate_conviction_threshold:
        stance = "MODERATELY BEARISH"
        conviction = "Medium"
    
    # Refine rationale
    if not rationale_components:
        primary_rationale_text = "Flow data balanced or lacks strong individual signals."
        if stance == "NEUTRAL - OBSERVE" and net_buy_sweep_qty == 0 and net_sell_sweep_qty == 0 and num_bull_hf_bursts == 0 and num_bear_hf_bursts == 0:
             primary_rationale_text = "Overall flow is quiet with no significant aggressive activity detected."

    elif len(rationale_components) == 1:
        primary_rationale_text = f"{rationale_components[0]} is the primary driver."
    else:
        primary_rationale_text = "; ".join(rationale_components[:3]) # Top 3 reasons
        if len(rationale_components) > 3:
            primary_rationale_text += " along with other supporting factors."
    
    # Handle contradictions more explicitly in rationale if possible (future enhancement)
    # For now, the score difference itself implies the net direction.
    # If bullish_score and bearish_score are both high and similar, that's a contradiction.
    if stance == "NEUTRAL - OBSERVE" and bullish_score > 0 and bearish_score > 0 and abs(net_score) < moderate_conviction_threshold :
        primary_rationale_text = f"Mixed signals: Bullish factors (score: {bullish_score}) countered by Bearish factors (score: {bearish_score}). Key drivers: {'; '.join(rationale_components[:2]) if rationale_components else 'N/A'}."
        conviction = "Low (Contradictory)"


    return stance, conviction, primary_rationale_text


# --- Main Holistic Analysis Function ---
def perform_holistic_flow_analysis(df_input: pd.DataFrame, 
                                   active_ticker: str, 
                                   time_window: str = '15min', 
                                   log_status_to_ui_func=None):
    if df_input is None or df_input.empty:
        if log_status_to_ui_func: log_status_to_ui_func(f"Holistic Analysis: No data provided for {active_ticker}.")
        return {"error": "No data provided for analysis."}, []
    
    if log_status_to_ui_func is None:
        log_status_to_ui_func = lambda msg: print(f"Analysis Log ({active_ticker}): {msg}")

    results = {}
    options_of_interest_details = [] 
    
    df = df_input.copy()
    df.attrs['active_ticker'] = active_ticker 
    df.attrs['log_status_to_ui_func_from_main'] = log_status_to_ui_func 

    log_status_to_ui_func(f"Starting holistic analysis for {active_ticker} with {len(df)} initial rows.")

    if 'Time' not in df.columns: 
        log_status_to_ui_func("Holistic Analysis: 'Time' column missing.")
        return {"error": "'Time' column missing."}, []
    if not pd.api.types.is_datetime64_any_dtype(df['Time']):
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df.dropna(subset=['Time'], inplace=True) 
    if df.empty: 
        log_status_to_ui_func("Holistic Analysis: DataFrame empty after Time handling.")
        return {"error": "DataFrame empty after Time handling."}, []

    df_for_resample = df.copy()
    if not isinstance(df_for_resample.index, pd.DatetimeIndex) or df_for_resample.index.name != 'Time':
        df_for_resample = df_for_resample.set_index('Time', drop=False).sort_index() 
    else:
        if 'Time' not in df_for_resample.columns: df_for_resample['Time'] = df_for_resample.index 
        df_for_resample = df_for_resample.sort_index()
    
    df_columnar = df.reset_index(drop=True) 

    num_cols = ['Delta', 'TradeQuantity', 'Strike_Price', 'Underlying_Price', 'Option_Bid', 'Option_Ask', 'Trade_Price', 'IV']
    for col in num_cols: 
        if col in df_columnar.columns: 
            df_columnar[col] = pd.to_numeric(df_columnar[col], errors='coerce')
        elif col in ['TradeQuantity', 'Trade_Price', 'Delta', 'Strike_Price']: 
            log_status_to_ui_func(f"Holistic Analysis: Essential numeric column '{col}' missing from df_columnar.")
            return {"error": f"Essential numeric column '{col}' missing."}, []
        else: 
            df_columnar[col] = np.nan
            
    if 'Expiration_Date' in df_columnar.columns: 
        df_columnar['Expiration_Date'] = pd.to_datetime(df_columnar['Expiration_Date'], errors='coerce')
    else: 
        log_status_to_ui_func("Holistic Analysis: 'Expiration_Date' column missing from df_columnar.")
        return {"error": "'Expiration_Date' column missing."}, []
    
    if 'Condition' not in df_columnar.columns: df_columnar['Condition'] = None 

    crit_cat_cols = ['Option_Type', 'Aggressor', 'Exchange', 'Option_Description_orig']
    for col in crit_cat_cols: 
        if col not in df_columnar.columns: 
            log_status_to_ui_func(f"Holistic Analysis: Essential categorical column '{col}' missing from df_columnar.")
            return {"error": f"Essential categorical column '{col}' missing."}, []
    
    cols_for_nan_check = ['TradeQuantity', 'Trade_Price', 'Aggressor', 'Option_Type', 'Strike_Price', 
                          'Delta', 'Expiration_Date', 'Exchange', 'Option_Description_orig', 'Time']
    df_columnar.dropna(subset=[col for col in cols_for_nan_check if col in df_columnar.columns], inplace=True)
    if df_columnar.empty: 
        log_status_to_ui_func("Holistic Analysis: df_columnar empty after dropping NaNs in critical columns.")
        return {"error": "df_columnar empty after critical NaN drop."}, []

    log_status_to_ui_func(f"Data prepped. Rows in df_columnar: {len(df_columnar)}. Rows in df_for_resample: {len(df_for_resample)}.")

    df_columnar['BuyVolumeAgg'] = np.where(df_columnar['Aggressor'] == 'Buyer (At Ask)', df_columnar['TradeQuantity'], 0)
    df_columnar['SellVolumeAgg'] = np.where(df_columnar['Aggressor'] == 'Seller (At Bid)', df_columnar['TradeQuantity'], 0)
    df_columnar['NetAggVolumeTrade'] = df_columnar['BuyVolumeAgg'] - df_columnar['SellVolumeAgg']

    log_status_to_ui_func("Analyzing high-frequency flow...")
    df_for_resample.attrs['active_ticker'] = active_ticker 
    df_for_resample.attrs['log_status_to_ui_func_from_main'] = log_status_to_ui_func
    hf_metrics_df, identified_bursts_df, hf_options_details = analyze_high_frequency_flow(df_for_resample.copy()) 
    results['high_frequency_metrics'] = hf_metrics_df
    results['identified_hf_bursts'] = identified_bursts_df
    options_of_interest_details.extend(hf_options_details)
    log_status_to_ui_func(f"HF analysis complete. {len(identified_bursts_df)} bursts, {len(hf_options_details)} HF options identified.")

    hf_summary_stats = {}
    if not identified_bursts_df.empty:
        ofi_bursts = identified_bursts_df[identified_bursts_df['BurstType'] == 'NetDollarOFI']
        agg_bursts = identified_bursts_df[identified_bursts_df['BurstType'] == 'NetAggressorQty']
        hf_summary_stats['num_ofi_bursts'] = len(ofi_bursts)
        hf_summary_stats['num_agg_bursts'] = len(agg_bursts)
        hf_summary_stats['strongest_ofi_buy_burst_peak'] = ofi_bursts[ofi_bursts['Direction'] == 'Buying']['PeakValue'].max(skipna=True) if not ofi_bursts[ofi_bursts['Direction'] == 'Buying'].empty else 0
        hf_summary_stats['strongest_ofi_sell_burst_peak'] = ofi_bursts[ofi_bursts['Direction'] == 'Selling']['PeakValue'].min(skipna=True) if not ofi_bursts[ofi_bursts['Direction'] == 'Selling'].empty else 0
        hf_summary_stats['strongest_agg_buy_burst_peak'] = agg_bursts[agg_bursts['Direction'] == 'Buying']['PeakValue'].max(skipna=True) if not agg_bursts[agg_bursts['Direction'] == 'Buying'].empty else 0
        hf_summary_stats['strongest_agg_sell_burst_peak'] = agg_bursts[agg_bursts['Direction'] == 'Selling']['PeakValue'].min(skipna=True) if not agg_bursts[agg_bursts['Direction'] == 'Selling'].empty else 0
    results['hf_summary_stats'] = {k: (v if pd.notna(v) else 0) for k, v in hf_summary_stats.items()}

    ofi_binned_list = []
    if not df_for_resample.empty and isinstance(df_for_resample.index, pd.DatetimeIndex):
        # Ensure 'Aggressor' column is present for hf_calculate_ofi_agg if df_for_resample was modified
        if 'Aggressor' not in df_for_resample.columns and 'Aggressor' in df_columnar.columns:
             # This merge might be complex if indices don't align perfectly or Time is not unique
             # A safer way is to ensure df_for_resample has all needed columns from the start
            log_status_to_ui_func("OFI Binning: 'Aggressor' column missing in df_for_resample. Attempting to use original df's Aggressor.")
            # This is tricky. df_for_resample was set_index('Time', drop=False) from 'df'. 'df' should have 'Aggressor'.
            # So df_for_resample should have it. Add a check.
        if 'Aggressor' not in df_for_resample.columns:
            log_status_to_ui_func("CRITICAL: 'Aggressor' column missing in df_for_resample for OFI binning. Skipping.")
        else:
            binned_agg_results = df_for_resample.resample(time_window).apply(hf_calculate_ofi_agg).fillna(0)
            for period_name, row in binned_agg_results.iterrows():
                ofi_binned_list.append({
                    'Time_Window': period_name, 'Aggressive_Buy_Value': row['BuyDollarVolume'],
                    'Aggressive_Sell_Value': row['SellDollarVolume'], 'Net_OFI_Value': row['NetDollarOFI'],
                    'Net_Aggressor_Qty': row['NetAggressorQty'], 'Total_Volume_Window': row['TotalQuantity']
                })
    else:
        log_status_to_ui_func("OFI Binning: df_for_resample is empty or not time-indexed. Skipping OFI binning.")
        
    ofi_df = pd.DataFrame(ofi_binned_list)
    if not ofi_df.empty: ofi_df = ofi_df.set_index('Time_Window')
    results['ofi_binned_summary'] = ofi_df[ofi_df['Total_Volume_Window'] > 0] if not ofi_df.empty else pd.DataFrame()
    log_status_to_ui_func("OFI binned summary calculated.")

    all_block_trades_df = analyze_block_trades(df_columnar.copy(), config.LARGE_TRADE_THRESHOLD_QTY)
    log_status_to_ui_func(f"Block trade analysis complete. Found {len(all_block_trades_df)} blocks.")
    
    identified_strategies_list, individual_block_trades_df = identify_common_strategies_from_blocks(
        all_block_trades_df.copy(), active_ticker
    )
    results['identified_common_strategies'] = identified_strategies_list
    results['individual_block_trades_summary'] = individual_block_trades_df 
    log_status_to_ui_func(f"Common strategy ID complete. Found {len(identified_strategies_list)} strategies.")

    log_status_to_ui_func(f"Detecting sweep orders for {active_ticker}...")
    identified_sweep_orders = detect_sweep_orders(df_columnar.copy(), active_ticker, log_status_to_ui_func)
    results['identified_sweep_orders'] = identified_sweep_orders
    log_status_to_ui_func(f"Sweep detection finished. Found {len(identified_sweep_orders)} sweeps.")

    # --- Stance Calculation ---
    # Pass necessary parts of 'results' collected so far
    stance_inputs = {
        'ofi_binned_summary': results.get('ofi_binned_summary'),
        'hf_summary_stats': results.get('hf_summary_stats'),
        'identified_sweep_orders': results.get('identified_sweep_orders'),
        'flow_by_moneyness': None # This will be calculated next
    }
    
    # Calculate flow by moneyness/DTE (uses df_columnar which has necessary 'Time' column)
    df_cats = df_columnar.copy() 
    df_cats['Moneyness'] = df_cats.apply(lambda r: categorize_moneyness(r['Delta'], r['Option_Type']), axis=1)
    df_cats['DTE_Category'] = df_cats.apply(lambda r: categorize_dte(r['Expiration_Date'], r['Time']), axis=1)

    agg_funcs_flow_cat = {
        'TotalVolumeQty': ('TradeQuantity', 'sum'),
        'BuyDollarVolume': ('Trade_Price', lambda x: (df_cats.loc[x.index, 'TradeQuantity'] * x * (df_cats.loc[x.index, 'Aggressor'] == 'Buyer (At Ask)')).sum()),
        'SellDollarVolume': ('Trade_Price', lambda x: (df_cats.loc[x.index, 'TradeQuantity'] * x * (df_cats.loc[x.index, 'Aggressor'] == 'Seller (At Bid)')).sum()),
        'AggressiveBuyVolumeQty': ('BuyVolumeAgg', 'sum'),
        'AggressiveSellVolumeQty': ('SellVolumeAgg', 'sum')
    }
    
    if not df_cats.empty:
        flow_by_moneyness_df = df_cats.groupby(['Option_Type', 'Moneyness']).agg(**agg_funcs_flow_cat).fillna(0)
        flow_by_moneyness_df['NetDollarOFI'] = flow_by_moneyness_df['BuyDollarVolume'] - flow_by_moneyness_df['SellDollarVolume']
        flow_by_moneyness_df['NetAggressiveVolumeQty'] = flow_by_moneyness_df['AggressiveBuyVolumeQty'] - flow_by_moneyness_df['AggressiveSellVolumeQty']
        results['flow_by_moneyness'] = flow_by_moneyness_df
        stance_inputs['flow_by_moneyness'] = results['flow_by_moneyness'] # Update for stance calc

        flow_by_dte_df = df_cats.groupby(['Option_Type', 'DTE_Category']).agg(**agg_funcs_flow_cat).fillna(0)
        flow_by_dte_df['NetDollarOFI'] = flow_by_dte_df['BuyDollarVolume'] - flow_by_dte_df['SellDollarVolume']
        flow_by_dte_df['NetAggressiveVolumeQty'] = flow_by_dte_df['AggressiveBuyVolumeQty'] - flow_by_dte_df['AggressiveSellVolumeQty']
        results['flow_by_dte'] = flow_by_dte_df
    else:
        results['flow_by_moneyness'] = pd.DataFrame(); results['flow_by_dte'] = pd.DataFrame()
    log_status_to_ui_func("Flow by moneyness/DTE calculated.")

    # Now call stance calculation
    market_stance, stance_conviction, stance_rationale = _calculate_market_stance_and_rationale(stance_inputs, active_ticker)
    results['market_stance'] = market_stance
    results['stance_conviction'] = stance_conviction
    results['stance_primary_rationale'] = stance_rationale
    log_status_to_ui_func(f"Market stance calculated: {market_stance} (Conviction: {stance_conviction})")


    # Populate Options of Interest Details (after all analyses that might generate interest)
    if identified_strategies_list:
        for strategy in identified_strategies_list:
            options_of_interest_details.append({
                'symbol': strategy['description'], 'reason': f"Large {strategy['strategy_type']} Identified",
                'metric': strategy['total_quantity'], 'time': strategy['time'],
                'agg': 'Spread Execution', 'is_complex_strategy': True, 
                'details_dict': strategy 
            })

    if not individual_block_trades_df.empty:
        for _, row in individual_block_trades_df.iterrows():
            time_of_trade = pd.Timestamp(row.get('Time')) if pd.notna(row.get('Time')) else pd.NaT
            if all(pd.notna(row[c]) for c in ['Expiration_Date', 'Strike_Price', 'Option_Type']):
                tos_sym = rtd_handler.format_tos_rtd_symbol(active_ticker, row['Expiration_Date'], row['Strike_Price'], row['Option_Type'])
                if tos_sym: 
                    options_of_interest_details.append({
                        'symbol': tos_sym, 'reason': 'Individual Block Trade', 
                        'metric': row['TradeQuantity'], 'time': time_of_trade, 
                        'agg': row.get('Aggressor', 'N/A'), 
                        'details_dict': row.to_dict() 
                        })
    
    if results.get('identified_sweep_orders'):
        for sweep in results['identified_sweep_orders']:
            options_of_interest_details.append({
                'symbol': sweep['tos_symbol'], 
                'reason': f"{sweep['aggressor_side']} Sweep Order ({sweep['number_of_legs']} legs)",
                'metric': sweep['total_quantity'], 'time': sweep['start_time'],       
                'agg': sweep['aggressor_detail'],  'is_complex_strategy': False,      
                'details_dict': sweep 
            })
            
    if not df_cats.empty: # df_cats is df_columnar with Moneyness/DTE
        current_ooi_symbols = {item['symbol'] for item in options_of_interest_details if not item.get('is_complex_strategy', False) and isinstance(item.get('symbol'), str) and item.get('symbol','').startswith('.')}
        
        top_vol_opts = df_cats.groupby(['Option_Description_orig', 'Expiration_Date', 'Strike_Price', 'Option_Type'])['TradeQuantity'].sum().nlargest(config.TOP_N_VOLUME_OPTIONS).reset_index()
        for _, r in top_vol_opts.iterrows():
            if all(pd.notna(r[c]) for c in ['Expiration_Date', 'Strike_Price', 'Option_Type']):
                tos_sym = rtd_handler.format_tos_rtd_symbol(active_ticker, r['Expiration_Date'], r['Strike_Price'], r['Option_Type'])
                if tos_sym and tos_sym not in current_ooi_symbols:
                    options_of_interest_details.append({'symbol': tos_sym, 'reason': 'High Total Volume', 
                                                        'metric': r['TradeQuantity'], 'time': pd.NaT, 'agg': 'N/A',
                                                        'is_complex_strategy': False, 'details_dict': r.to_dict() })
        results['most_active_options_by_volume'] = top_vol_opts[['Option_Description_orig', 'TradeQuantity']]

        net_agg_opt = df_cats.groupby(['Option_Description_orig', 'Option_Type', 'Expiration_Date', 'Strike_Price']).agg(NetAggVolQty=('NetAggVolumeTrade', 'sum')).reset_index()
        
        for reason_prefix, qty_col, is_buy, opt_filter_type in [
            ("Top Net Aggressive Call Buy Qty", 'NetAggVolQty', True, 'C'),
            ("Top Net Aggressive Put Buy Qty", 'NetAggVolQty', True, 'P'),
            ("Top Net Aggressive Call Sell Qty", 'NetAggVolQty', False, 'C'),
            ("Top Net Aggressive Put Sell Qty", 'NetAggVolQty', False, 'P')
        ]:
            filtered_agg = net_agg_opt[net_agg_opt['Option_Type'] == opt_filter_type]
            if is_buy:
                top_agg = filtered_agg[filtered_agg[qty_col] > 0].nlargest(config.TOP_N_AGGRESSIVE_OPTIONS, qty_col)
                agg_desc = "Net Buy"
            else:
                top_agg = filtered_agg[filtered_agg[qty_col] < 0].nsmallest(config.TOP_N_AGGRESSIVE_OPTIONS, qty_col)
                agg_desc = "Net Sell"

            for _, r in top_agg.iterrows():
                if all(pd.notna(r[c]) for c in ['Expiration_Date', 'Strike_Price', 'Option_Type']):
                    tos_sym = rtd_handler.format_tos_rtd_symbol(active_ticker, r['Expiration_Date'], r['Strike_Price'], r['Option_Type'])
                    if tos_sym and tos_sym not in current_ooi_symbols:
                        options_of_interest_details.append({'symbol': tos_sym, 'reason': reason_prefix, 
                                                            'metric': r[qty_col], 'time': pd.NaT, 'agg': agg_desc,
                                                            'is_complex_strategy': False, 'details_dict': r.to_dict()})
    else: results['most_active_options_by_volume'] = pd.DataFrame()
    log_status_to_ui_func("Top volume and aggressive options identified.")

    if 'IV' in df_cats.columns and not df_cats.empty: # From df_columnar which has IV
        results['iv_by_moneyness'] = df_cats.groupby(['Option_Type', 'Moneyness'])['IV'].mean().dropna()
        results['iv_by_dte'] = df_cats.groupby(['Option_Type', 'DTE_Category'])['IV'].mean().dropna()
    else:
        results['iv_by_moneyness'] = pd.Series(dtype=float); results['iv_by_dte'] = pd.Series(dtype=float)

    options_of_interest_details.sort(key=get_sort_key) 
    
    unique_options_map = {}
    final_options_of_interest_symbols_list = []
    temp_symbol_set = set() 

    for item in options_of_interest_details:
        item_symbol = item.get('symbol', '') 
        if not item.get('is_complex_strategy', False) and isinstance(item_symbol, str) and item_symbol.startswith('.'):
            if item_symbol not in temp_symbol_set:
                 if len(temp_symbol_set) < config.MAX_OPTIONS_FOR_RTD: 
                    final_options_of_interest_symbols_list.append(item_symbol)
                    temp_symbol_set.add(item_symbol)
        
        map_key = item_symbol 
        if map_key not in unique_options_map: 
            if len(unique_options_map) < config.MAX_OPTIONS_FOR_BRIEFING_MAP: 
                 unique_options_map[map_key] = item 
        
        if len(temp_symbol_set) >= config.MAX_OPTIONS_FOR_RTD and \
           len(unique_options_map) >= config.MAX_OPTIONS_FOR_BRIEFING_MAP: 
            break
            
    results['options_of_interest_details_map'] = unique_options_map
    log_status_to_ui_func(f"Final options of interest compiled. {len(final_options_of_interest_symbols_list)} for RTD, {len(unique_options_map)} for briefing map.")
    
    return results, final_options_of_interest_symbols_list

# Re-export reporting functions for main_app.py to call via analysis_engine if needed
# from analysis_modules.report_generator import generate_trade_briefing, _get_dynamic_strategy_suggestions, generate_detailed_txt_report
