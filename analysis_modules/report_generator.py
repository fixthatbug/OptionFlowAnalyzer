# analysis_modules/report_generator.py
# Functions for generating trade briefings and detailed text reports.

import pandas as pd
import numpy as np
from datetime import datetime
import config
import rtd_handler 
from analysis_modules.analysis_helpers import categorize_moneyness, categorize_dte, get_sort_key 

def _format_percentage(value, default_na="N/A"):
    if pd.notna(value) and isinstance(value, (int, float)):
        return f"{value*100:.1f}%"
    return default_na

def _format_signed_percentage_change(value, default_na="N/A"):
    if pd.notna(value) and isinstance(value, (int, float)):
        return f"{value*100:+.1f}%" # Added plus sign for positive changes
    return default_na

def _format_float(value, precision=2, default_na="N/A"):
    if pd.notna(value) and isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return default_na

def _format_int(value, default_na="N/A"):
    if pd.notna(value) and isinstance(value, (int, float)): # Allow float if it's a whole number
        return f"{int(value):,}" # Added comma for thousands
    return default_na


def _get_dynamic_strategy_suggestions(stance, primary_option_symbol, rtd_data_df_indexed, active_ticker, options_of_interest_symbols, holistic_analysis_results):
    suggestions = []
    option_name_readable = primary_option_symbol if primary_option_symbol else "a key option"
    rtd_iv_str = "N/A (Check RTD)"
    rtd_oi_str = "N/A (Check RTD)"
    is_iv_high = None 

    if primary_option_symbol and isinstance(rtd_data_df_indexed, pd.DataFrame) and not rtd_data_df_indexed.empty and primary_option_symbol in rtd_data_df_indexed.index:
        live_data = rtd_data_df_indexed.loc[primary_option_symbol]
        iv_val_str = str(live_data.get('IV (%)', 'N/A')).replace('%','')
        try:
            iv_numeric = float(iv_val_str) 
            rtd_iv_str = _format_percentage(iv_numeric)
            if iv_numeric > config.IV_HIGH_THRESHOLD: is_iv_high = True 
            elif iv_numeric < config.IV_LOW_THRESHOLD: is_iv_high = False 
        except (ValueError, TypeError): pass 
        
        oi_val = live_data.get('Open Int', 'N/A')
        try: 
            rtd_oi_str = _format_int(float(str(oi_val).replace(',',''))) if pd.notna(oi_val) and str(oi_val).replace(',','').replace('.','',1).isdigit() else str(oi_val)
        except (ValueError, TypeError): 
            rtd_oi_str = str(oi_val) 
        
        parsed_opt = rtd_handler.parse_tos_option_symbol(primary_option_symbol)
        if parsed_opt:
            exp_date_readable = datetime(parsed_opt['exp_year'], parsed_opt['exp_month'], parsed_opt['exp_day']).strftime('%b %d \'%y')
            strike_f = float(parsed_opt['strike_price']) if pd.notna(parsed_opt['strike_price']) else np.nan
            option_name_readable = f"{active_ticker} {exp_date_readable} {_format_float(strike_f)}{parsed_opt['option_type']}" if pd.notna(strike_f) else primary_option_symbol

    suggestions.append(f"**Strategy Ideas for {option_name_readable} (Live IV: {rtd_iv_str}, OI: {rtd_oi_str}):**")

    if stance.endswith("BULLISH"):
        suggestions.append("  * **Long Call:** Simplest bullish bet. Max profit unlimited; max loss premium.")
        if is_iv_high is False: suggestions.append("    * ✔️ *Favorable if IV is low/moderate.*")
        elif is_iv_high is True: suggestions.append("    * ⚠️ *Costly if IV is high; risk of IV crush.*")
        
        suggestions.append("  * **Bull Call Spread (Debit):** Buy call, sell higher strike call. Reduces cost, defines risk/reward.")
        if is_iv_high is True: suggestions.append("    * ✔️ *Good for high IV; benefits from skew.*")
        
        suggestions.append("  * **Bull Put Spread (Credit):** Sell put, buy lower strike put. Profit if stock stays above short put. Collects premium.")
        if is_iv_high is True: suggestions.append("    * ✔️ *Good for high IV premium collection.*")
        elif is_iv_high is False: suggestions.append("    * ⚠️ *Less premium if IV is low.*")

    elif stance.endswith("BEARISH"):
        suggestions.append("  * **Long Put:** Simplest bearish bet.")
        if is_iv_high is False: suggestions.append("    * ✔️ *Favorable if IV is low/moderate.*")
        elif is_iv_high is True: suggestions.append("    * ⚠️ *Costly if IV is high.*")

        suggestions.append("  * **Bear Put Spread (Debit):** Buy put, sell lower strike put.")
        if is_iv_high is True: suggestions.append("    * ✔️ *Good for reducing cost in high IV.*")
        
        suggestions.append("  * **Bear Call Spread (Credit):** Sell call, buy higher strike call.")
        if is_iv_high is True: suggestions.append("    * ✔️ *Good for high IV premium collection.*")

    elif "NEUTRAL" in stance or "OBSERVE" in stance :
        suggestions.append("  * **Iron Condor (Credit):** Sell OTM call spread & OTM put spread. Profits if stock stays in range.")
        if is_iv_high is True: suggestions.append("    * ✔️ *Ideal for high IV environments.*")
        else: suggestions.append("    * ⚠️ *Lower premium & tighter profit range if IV is low.*")
        
        suggestions.append("  * **Long Straddle/Strangle (Debit, if expecting volatility spike):** Buy call & put.")
        if is_iv_high is False: suggestions.append("    * ✔️ *Cheaper to establish if IV is low before catalyst.*")
        elif is_iv_high is True: suggestions.append("    * ⚠️ *Very expensive if IV is already high; needs large move.*")
    
    if len(suggestions) <= 1: 
        suggestions.append("* No specific strategy variations generated based on current stance and IV. Consult general options education.*")
    return "\n".join(suggestions)


def generate_trade_briefing(holistic_analysis_results, rtd_data_df, active_ticker, options_of_interest_symbols):
    briefing_parts = []
    analysis_date_str = "N/A"
    
    first_valid_timestamp = None
    if 'ofi_binned_summary' in holistic_analysis_results:
        ofi_binned_df_local = holistic_analysis_results.get('ofi_binned_summary', pd.DataFrame())
        if isinstance(ofi_binned_df_local, pd.DataFrame) and not ofi_binned_df_local.empty and isinstance(ofi_binned_df_local.index, pd.DatetimeIndex):
            if not ofi_binned_df_local.index.empty:
                first_valid_timestamp = ofi_binned_df_local.index[0]
    if pd.notna(first_valid_timestamp) and isinstance(first_valid_timestamp, pd.Timestamp):
        analysis_date_str = first_valid_timestamp.strftime("%Y-%m-%d")

    briefing_parts.append(f"## Actionable Trade Briefing & Strategy Recommendation")
    briefing_parts.append(f"**Ticker:** {active_ticker} | **Flow Analysis Date:** {analysis_date_str} | **RTD Snapshot:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    briefing_parts.append("---")

    if holistic_analysis_results.get("error"):
        briefing_parts.append(f"\n**ERROR IN HISTORICAL ANALYSIS:** {holistic_analysis_results['error']}")
        return "\n".join(briefing_parts)

    stance = holistic_analysis_results.get('market_stance', "NEUTRAL - OBSERVE (Stance calculation error)")
    conviction = holistic_analysis_results.get('stance_conviction', "Low (Conviction calculation error)")
    primary_rationale = holistic_analysis_results.get('stance_primary_rationale', "Rationale not available or calculation error.")

    briefing_parts.append(f"\n**I. Overall Stance & Conviction:**")
    briefing_parts.append(f"* **Market Stance for {active_ticker}:** {stance} (Conviction: {conviction})")
    briefing_parts.append(f"* **Primary Rationale:** {primary_rationale}")

    briefing_parts.append("\n\n**II. Detailed Supporting Analysis & Observations:**")
    ofi_binned_summary_df = holistic_analysis_results.get('ofi_binned_summary', pd.DataFrame())
    net_ofi_sum_dollar = ofi_binned_summary_df['Net_OFI_Value'].sum() if not ofi_binned_summary_df.empty and 'Net_OFI_Value' in ofi_binned_summary_df.columns else 0.0
    net_agg_qty_overall = ofi_binned_summary_df['Net_Aggressor_Qty'].sum() if not ofi_binned_summary_df.empty and 'Net_Aggressor_Qty' in ofi_binned_summary_df.columns else 0.0
    briefing_parts.append(f"* **A. Overall Flow Diagnosis (Historical):** Net OFI: ${_format_float(net_ofi_sum_dollar,0)}. Net Agg Qty (all trades): {_format_int(net_agg_qty_overall)} contracts.")
    
    flow_by_moneyness = holistic_analysis_results.get('flow_by_moneyness', pd.DataFrame())
    if not flow_by_moneyness.empty: 
        briefing_parts.append(f"    * Flow by Moneyness (Net Agg Qty | Net OFI):\n{flow_by_moneyness[['NetAggressiveVolumeQty', 'NetDollarOFI']].to_string(float_format='%.2f')}")
    
    flow_by_dte = holistic_analysis_results.get('flow_by_dte', pd.DataFrame())
    if not flow_by_dte.empty:
        briefing_parts.append(f"\n    * Flow by DTE (Net Agg Qty | Net OFI):\n{flow_by_dte[['NetAggressiveVolumeQty', 'NetDollarOFI']].to_string(float_format='%.2f')}")

    briefing_parts.append("\n* **B. High-Frequency Insights (Top Bursts Detailed):**") 
    identified_bursts_df = holistic_analysis_results.get('identified_hf_bursts', pd.DataFrame())
    if not identified_bursts_df.empty:
        sorted_bursts_for_briefing = identified_bursts_df.copy()
        if 'PeakValue' in sorted_bursts_for_briefing.columns:
            sorted_bursts_for_briefing['AbsPeakValue'] = sorted_bursts_for_briefing['PeakValue'].abs()
            sorted_bursts_for_briefing = sorted_bursts_for_briefing.sort_values(by='AbsPeakValue', ascending=False)
        
        num_bursts_to_detail = config.HF_BURST_BRIEFING_TOP_N_DISPLAY 
        
        briefing_parts.append(f"    * **Detailed Look at Top {min(num_bursts_to_detail, len(sorted_bursts_for_briefing))} Significant HF Bursts:**")

        for i, (_, burst_row) in enumerate(sorted_bursts_for_briefing.head(num_bursts_to_detail).iterrows()):
            value_unit = "$" if "OFI" in burst_row.get('BurstType','') else "contracts"
            start_time_dt = pd.Timestamp(burst_row.get('StartTime')) 
            start_time_str = start_time_dt.strftime('%H:%M:%S') if pd.notna(start_time_dt) else 'N/A'
            
            ul_price_start_burst = burst_row.get('UnderlyingPrice_Start', np.nan)
            ul_price_end_burst = burst_row.get('UnderlyingPrice_End', np.nan)
            ul_price_min_burst = burst_row.get('UnderlyingPrice_MinInBurst', np.nan) 
            ul_price_max_burst = burst_row.get('UnderlyingPrice_MaxInBurst', np.nan)

            ul_price_start_str = _format_float(ul_price_start_burst)
            ul_price_end_str = _format_float(ul_price_end_burst)
            ul_price_range_str = f" (Range: {_format_float(ul_price_min_burst)}-{_format_float(ul_price_max_burst)})" if pd.notna(ul_price_min_burst) and pd.notna(ul_price_max_burst) else ""
            
            briefing_parts.append(
                f"        * **Burst {i+1} ({burst_row.get('BurstType','N/A')} - {burst_row.get('Direction','N/A')}):** "
                f"Start: {start_time_str}, Dur: {_format_float(burst_row.get('DurationSeconds',0),1)}s, Peak: {_format_int(burst_row.get('PeakValue',0))} {value_unit}. "
                f"UL Price: {ul_price_start_str} -> {ul_price_end_str}{ul_price_range_str}."
            )
            
            top_options_in_burst = burst_row.get('TopOptionsInBurst', [])
            if top_options_in_burst:
                briefing_parts.append("            * Key Contracts in this Burst:")
                for opt_detail in top_options_in_burst:
                    iv_change_str = _format_signed_percentage_change(opt_detail.get('iv_change_in_burst'))
                    avg_iv_str = _format_percentage(opt_detail.get('avg_iv_in_burst'))
                    iv_start_str = _format_percentage(opt_detail.get('iv_start_trade'))
                    iv_end_str = _format_percentage(opt_detail.get('iv_end_trade'))
                    avg_delta_str = _format_float(opt_detail.get('avg_delta_in_burst'), 2)

                    briefing_parts.append(
                        f"                - {opt_detail.get('tos_symbol','N/A')} (DTE: {opt_detail.get('dte_at_burst','N/A')}): "
                        f"Qty: {_format_int(opt_detail.get('total_qty_in_burst',0))}, NetAgg: {_format_int(opt_detail.get('net_agg_qty_in_burst',0))}. "
                        f"IV: {iv_start_str} -> {iv_end_str} (Chg: {iv_change_str}, Avg: {avg_iv_str}). "
                        f"AvgDelta: {avg_delta_str}"
                    )
            else:
                briefing_parts.append("            * No specific top options detailed for this burst.")
            briefing_parts.append("") 
        
        if len(identified_bursts_df) > num_bursts_to_detail:
             briefing_parts.append(f"        * ... and {len(identified_bursts_df)-num_bursts_to_detail} more bursts identified (see detailed report).")
    else: 
        briefing_parts.append("    * No significant HF bursts met criteria.")

    briefing_parts.append("\n* **C. Standout Option Trades & Strategies (Historical Flow):**")
    
    identified_sweeps = holistic_analysis_results.get('identified_sweep_orders', [])
    if identified_sweeps:
        briefing_parts.append("    * **Noteworthy Sweep Orders Detected (Top 3 by Quantity):**")
        sorted_sweeps = sorted(identified_sweeps, key=lambda x: x.get('total_quantity', 0), reverse=True)
        for i, sweep in enumerate(sorted_sweeps[:3]):
            start_time_dt = pd.Timestamp(sweep.get('start_time'))
            end_time_dt = pd.Timestamp(sweep.get('end_time'))
            briefing_parts.append(
                f"        * **{sweep.get('aggressor_side','N/A')} Sweep:** {_format_int(sweep.get('total_quantity',0))} contracts of "
                f"{sweep.get('tos_symbol','N/A')} from {start_time_dt.strftime('%H:%M:%S')} to "
                f"{end_time_dt.strftime('%H:%M:%S')} across "
                f"{len(sweep.get('exchanges_involved',[]))} exchanges. Avg Price: ${_format_float(sweep.get('average_price',0))}."
            )
        if len(sorted_sweeps) > 3: briefing_parts.append(f"        * ... and {len(sorted_sweeps)-3} more sweeps.")

    identified_strategies = holistic_analysis_results.get('identified_common_strategies', [])
    if identified_strategies:
        briefing_parts.append("    * **Noteworthy Complex Strategies Identified (Top 3 by Quantity):**")
        sorted_strategies = sorted(identified_strategies, key=lambda x: x.get('total_quantity',0), reverse=True) 
        for i, strategy in enumerate(sorted_strategies[:3]): 
            strat_time_dt = pd.Timestamp(strategy.get('time'))
            strat_time_str = strat_time_dt.strftime('%H:%M:%S') if pd.notna(strat_time_dt) else "N/A"
            underlying_price_at_trade = strategy.get('underlying_at_trade', np.nan)
            underlying_price_str = _format_float(underlying_price_at_trade)
            
            briefing_parts.append(
                f"        * **{strategy.get('strategy_type','Unknown Strategy')} ({strategy.get('condition', 'N/A')}):** {strategy.get('description','N/A')} "
                f"at {strat_time_str} (UL: {underlying_price_str})."
            )
            for leg_idx, leg in enumerate(strategy.get('legs', [])):
                leg_iv_str = _format_percentage(leg.get('IV'))
                briefing_parts.append(
                    f"            - Leg: {_format_int(leg.get('TradeQuantity',0))} {leg.get('Option_Description_orig','Leg')} @ ${_format_float(leg.get('Trade_Price',0))} (IV: {leg_iv_str})"
                )
            if strategy.get('combined_premium') is not None:
                 briefing_parts.append(f"            - Approx. Combined Premium: ${_format_float(strategy['combined_premium'])}")
            interpretation = "This large "
            if strategy.get('strategy_type') == "Straddle" or strategy.get('strategy_type') == "Strangle":
                interpretation += f"{strategy.get('strategy_type','strategy').lower()} suggests a significant bet on volatility for {strategy.get('active_ticker','ticker')}."
            else: interpretation += f"{strategy.get('strategy_type','strategy').lower()} suggests a specific strategic play."
            briefing_parts.append(f"            - Interpretation: {interpretation}")

    individual_block_trades_df = holistic_analysis_results.get('individual_block_trades_summary', pd.DataFrame())
    if not individual_block_trades_df.empty:
        briefing_parts.append("    * **Noteworthy Individual Block Trades (Not part of above strategies/sweeps, Top 3 by Qty):**")
        for i, (_, row) in enumerate(individual_block_trades_df.head(3).iterrows()): 
            trade_time_dt = pd.to_datetime(row.get('Time'))
            trade_time_str = trade_time_dt.strftime('%H:%M:%S') if pd.notna(trade_time_dt) else "N/A"
            exp_date_dt = pd.to_datetime(row.get('Expiration_Date')) if pd.notna(row.get('Expiration_Date')) else pd.NaT

            iv_str = _format_percentage(row.get('IV'))
            delta_str = _format_float(row.get('Delta'), 2)
            
            moneyness = categorize_moneyness(row.get('Delta'), row.get('Option_Type')) 
            dte_cat = categorize_dte(exp_date_dt, trade_time_dt) 
            
            intent_detail = f"{row.get('Aggressor','N/A')} of {_format_int(row.get('TradeQuantity','N/A'))} {row.get('Option_Description_orig','N/A')} @ ${_format_float(row.get('Trade_Price',0))}."
            intent_detail += f" (IV: {iv_str}, Delta: {delta_str}, {moneyness}, {dte_cat}, UL: ${_format_float(row.get('Underlying_Price',0))} at {trade_time_str}, Cond: {row.get('Condition', 'N/A')})"
            briefing_parts.append(f"        * {intent_detail}")
    elif not identified_strategies and not identified_sweeps: 
        briefing_parts.append(f"    * No block trades (qty >= {config.LARGE_TRADE_THRESHOLD_QTY}), common strategies, or sweeps readily identified in this dataset.")

    briefing_parts.append("\n\n**III. Key Options of Interest & Live RTD Snapshot (Max 20 for RTD list):**")
    options_details_map = holistic_analysis_results.get('options_of_interest_details_map', {})
    if not options_details_map:
        briefing_parts.append("* No specific options of high interest identified from historical flow.")
    else:
        rtd_df_indexed = rtd_data_df.set_index('ToS Symbol') if isinstance(rtd_data_df, pd.DataFrame) and 'ToS Symbol' in rtd_data_df.columns and not rtd_data_df.empty and rtd_data_df.index.name != 'ToS Symbol' else pd.DataFrame()
        display_count = 0
        
        sorted_options_for_display = sorted(list(options_details_map.values()), key=get_sort_key)

        for item_details in sorted_options_for_display:
            if display_count >= config.MAX_OPTIONS_FOR_BRIEFING_DISPLAY: break  
            item_symbol_or_desc = item_details.get('symbol', 'N/A Symbol') 
            reason = item_details.get('reason', 'General Interest')
            metric_val = item_details.get('metric')
            metric_str = _format_int(metric_val) if isinstance(metric_val, (int, float)) else str(metric_val) if pd.notna(metric_val) else ""
            briefing_parts.append(f"* `{item_symbol_or_desc}` (Reason: {reason}, Metric: {metric_str})")

            details_dict_for_item = item_details.get('details_dict', {})

            if "Sweep Order" in reason:
                sweep_data = details_dict_for_item
                start_time_dt_sweep = pd.Timestamp(sweep_data.get('start_time'))
                end_time_dt_sweep = pd.Timestamp(sweep_data.get('end_time'))
                briefing_parts.append(
                    f"    * Sweep Details: {_format_int(sweep_data.get('total_quantity',0))} contracts from "
                    f"{start_time_dt_sweep.strftime('%H:%M:%S')} to {end_time_dt_sweep.strftime('%H:%M:%S')} "
                    f"across {len(sweep_data.get('exchanges_involved',[]))} exchanges. Avg Price: ${_format_float(sweep_data.get('average_price',0))}."
                )
            elif item_details.get('is_complex_strategy', False):
                strategy_data = details_dict_for_item 
                legs_info = []
                for leg in strategy_data.get('legs',[]):
                    legs_info.append(f"{_format_int(leg.get('TradeQuantity',0))} {leg.get('Option_Description_orig','Leg')} @ ${_format_float(leg.get('Trade_Price',0))}")
                combined_premium_val = strategy_data.get('combined_premium')
                combined_premium_str = _format_float(combined_premium_val)
                briefing_parts.append(f"    * Strategy Details: {', '.join(legs_info)}. Approx. Combined Premium: ${combined_premium_str}")
            elif isinstance(item_symbol_or_desc, str) and item_symbol_or_desc.startswith('.'): 
                if not rtd_df_indexed.empty and item_symbol_or_desc in rtd_df_indexed.index:
                    live = rtd_df_indexed.loc[item_symbol_or_desc]
                    live_iv_str = _format_percentage(live.get('IV (%)'))
                    live_delta_str = _format_float(live.get('Delta'),2)
                    live_last_str = _format_float(live.get('Last'))
                    live_bid_str = _format_float(live.get('Bid'))
                    live_ask_str = _format_float(live.get('Ask'))
                    live_vol_str = _format_int(live.get('Volume'))
                    live_oi_str = _format_int(live.get('Open Int'))


                    briefing_parts.append(f"    * RTD: Last: {live_last_str}, Bid: {live_bid_str}, Ask: {live_ask_str}, Vol: {live_vol_str}, OI: {live_oi_str}, IV: {live_iv_str}, Delta: {live_delta_str}")
                else: briefing_parts.append("    * RTD: Live data not available for this symbol in current snapshot.")
            display_count += 1

    briefing_parts.append("\n\n**IV. Illustrative Strategy Considerations (Based on Stance & Primary Option):**")
    primary_opt_for_strat = options_of_interest_symbols[0] if options_of_interest_symbols else None
    rtd_df_indexed_for_strat = rtd_data_df.set_index('ToS Symbol') if isinstance(rtd_data_df, pd.DataFrame) and 'ToS Symbol' in rtd_data_df.columns and not rtd_data_df.empty and rtd_data_df.index.name != 'ToS Symbol' else pd.DataFrame()
    strategy_text = _get_dynamic_strategy_suggestions(stance, primary_opt_for_strat, rtd_df_indexed_for_strat, active_ticker, options_of_interest_symbols, holistic_analysis_results)
    briefing_parts.append(strategy_text)

    briefing_parts.append("\n\n**V. My \"Wall Street\" Perspective & Refined Strategic Considerations:**")
    perspective = f"For {active_ticker}, the current stance is **{stance}** (Conviction: {conviction}). " 
    perspective += f"{primary_rationale} " 
    
    if options_details_map and (stance == "NEUTRAL - OBSERVE" or "MODERATELY" in stance):
        top_interest_items_for_perspective = sorted(list(options_details_map.values()), key=get_sort_key) 
        if top_interest_items_for_perspective:
            top_item = top_interest_items_for_perspective[0] 
            if "Sweep Order" in top_item.get('reason',''):
                 perspective += f" Noteworthy is the {top_item['reason']} for {top_item['symbol']} (Qty: {_format_int(top_item.get('metric',0))}), suggesting urgent positioning. "
            elif top_item.get('is_complex_strategy'):
                perspective += f" Significant complex trades like the identified {top_item.get('strategy_details',{}).get('strategy_type','Unknown Strategy')} ({top_item['symbol']}) suggest sophisticated positioning. "

    briefing_parts.append(perspective)
    briefing_parts.append("\n**Key Risks & Management:** Options trading involves substantial risk. Always manage risk exposure. This analysis is a snapshot based on historical flow; live market conditions and your trading plan dictate actual trades.")
    briefing_parts.append("\n**Next Steps:** Actively monitor RTD for 'Options of Interest.' Correlate with live underlying price action, volume, and key technical levels. Await entry/exit triggers aligned with your specific trading plan and risk tolerance.")
    return "\n".join(briefing_parts)

def generate_detailed_txt_report(holistic_analysis_results, active_ticker, rtd_data_df=None):
    report_lines = []
    analysis_date_str = "N/A"
    first_valid_timestamp = None
    if 'ofi_binned_summary' in holistic_analysis_results: 
        ofi_binned_df_local = holistic_analysis_results.get('ofi_binned_summary', pd.DataFrame())
        if isinstance(ofi_binned_df_local, pd.DataFrame) and not ofi_binned_df_local.empty and isinstance(ofi_binned_df_local.index, pd.DatetimeIndex):
            if not ofi_binned_df_local.index.empty:
                first_valid_timestamp = ofi_binned_df_local.index[0]
    if pd.notna(first_valid_timestamp) and isinstance(first_valid_timestamp, pd.Timestamp):
        analysis_date_str = first_valid_timestamp.strftime("%Y-%m-%d")

    report_lines.append(f"--- Detailed Flow Analysis Report for: {active_ticker} ---")
    report_lines.append(f"Analysis Date (from data): {analysis_date_str}") 
    report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    stance = holistic_analysis_results.get('market_stance', "N/A")
    conviction = holistic_analysis_results.get('stance_conviction', "N/A")
    rationale = holistic_analysis_results.get('stance_primary_rationale', "N/A")
    report_lines.append(f"\n--- Calculated Market Stance ---")
    report_lines.append(f"Stance: {stance} (Conviction: {conviction})")
    report_lines.append(f"Rationale: {rationale}")

    report_lines.append("\n--- Overall Summary ---")
    ofi_binned_summary = holistic_analysis_results.get('ofi_binned_summary', pd.DataFrame())
    if not ofi_binned_summary.empty:
        report_lines.append(f"Net OFI (Total): ${_format_float(ofi_binned_summary['Net_OFI_Value'].sum(),0)}")
        report_lines.append(f"Net Aggressor Qty (Total): {_format_int(ofi_binned_summary['Net_Aggressor_Qty'].sum())}")
    
    report_lines.append("\n--- Identified Sweep Orders ---")
    identified_sweeps = holistic_analysis_results.get('identified_sweep_orders', [])
    if identified_sweeps:
        for i, sweep in enumerate(identified_sweeps):
            start_time_dt_sweep = pd.Timestamp(sweep.get('start_time'))
            end_time_dt_sweep = pd.Timestamp(sweep.get('end_time'))
            report_lines.append(
                f"  Sweep {i+1}: {sweep.get('aggressor_side','N/A')} of {_format_int(sweep.get('total_quantity',0))} {sweep.get('tos_symbol','N/A')} "
                f"from {start_time_dt_sweep.strftime('%H:%M:%S')} to {end_time_dt_sweep.strftime('%H:%M:%S')} "
                f"across {len(sweep.get('exchanges_involved',[]))} exchanges. Avg Price: ${_format_float(sweep.get('average_price',0))}. "
                f"Legs: {sweep.get('number_of_legs')}"
            )
    else:
        report_lines.append("  None identified.")

    report_lines.append("\n--- Identified Common Strategies ---")
    identified_strategies = holistic_analysis_results.get('identified_common_strategies', [])
    if identified_strategies:
        for i, strategy in enumerate(identified_strategies):
            report_lines.append(f"  Strategy {i+1}: {strategy.get('description', 'N/A')}")
            for leg_idx, leg in enumerate(strategy.get('legs', [])):
                 leg_iv_str = _format_percentage(leg.get('IV'))
                 report_lines.append(f"    - Leg {leg_idx+1}: {_format_int(leg.get('TradeQuantity',0))} {leg.get('Option_Description_orig','N/A')} @ ${_format_float(leg.get('Trade_Price',0))} (IV: {leg_iv_str})")
            if strategy.get('combined_premium') is not None:
                 report_lines.append(f"    - Approx. Combined Premium: ${_format_float(strategy['combined_premium'])}")
    else:
        report_lines.append("  None identified.")

    report_lines.append("\n--- Individual Block Trades (Not part of above strategies/sweeps) ---")
    individual_blocks = holistic_analysis_results.get('individual_block_trades_summary', pd.DataFrame())
    if not individual_blocks.empty:
        blocks_to_display = individual_blocks.copy()
        if 'Time' in blocks_to_display.columns:
            blocks_to_display['Time'] = pd.to_datetime(blocks_to_display['Time'], errors='coerce')
        formatters = {}
        cols_to_format_float = ['Trade_Price', 'Underlying_Price', 'IV', 'Delta', 'Strike_Price']
        for col_f in cols_to_format_float:
            if col_f in blocks_to_display.columns:
                if col_f == 'IV': formatters[col_f] = lambda x: _format_percentage(x, default_na="N/A")
                elif col_f == 'Delta': formatters[col_f] = lambda x: _format_float(x, 4, "N/A") # More precision for delta
                else: formatters[col_f] = lambda x: _format_float(x, 2, "N/A")
        if 'TradeQuantity' in blocks_to_display.columns:
            formatters['TradeQuantity'] = lambda x: _format_int(x, "N/A")
        if 'Time' in blocks_to_display.columns: 
            formatters['Time'] = lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else "N/A"
        report_lines.append(blocks_to_display.to_string(index=False, formatters=formatters))
    else:
        report_lines.append("  No individual block trades met criteria (after strategy/sweep extraction).")
    
    report_lines.append("\n--- High-Frequency Bursts (Detailed) ---") 
    hf_bursts_df = holistic_analysis_results.get('identified_hf_bursts', pd.DataFrame())
    if not hf_bursts_df.empty:
        for idx, burst_row in hf_bursts_df.iterrows(): 
            start_time_dt_burst = pd.Timestamp(burst_row.get('StartTime'))
            start_time_str = start_time_dt_burst.strftime('%H:%M:%S') if pd.notna(start_time_dt_burst) else 'N/A'
            
            ul_price_start_burst = burst_row.get('UnderlyingPrice_Start', np.nan)
            ul_price_end_burst = burst_row.get('UnderlyingPrice_End', np.nan)
            ul_price_min_burst = burst_row.get('UnderlyingPrice_MinInBurst', np.nan)
            ul_price_max_burst = burst_row.get('UnderlyingPrice_MaxInBurst', np.nan)

            ul_price_start_str = _format_float(ul_price_start_burst)
            ul_price_end_str = _format_float(ul_price_end_burst)
            ul_price_range_str = f" (Range: {_format_float(ul_price_min_burst)}-{_format_float(ul_price_max_burst)})" if pd.notna(ul_price_min_burst) and pd.notna(ul_price_max_burst) else ""
            
            report_lines.append(
                f"  Burst {idx+1} ({burst_row.get('BurstType','N/A')} - {burst_row.get('Direction','N/A')}): "
                f"Start: {start_time_str}, Dur: {_format_float(burst_row.get('DurationSeconds',0),1)}s, Peak: {_format_int(burst_row.get('PeakValue',0))} "
                f"{'$' if 'OFI' in burst_row.get('BurstType','') else 'contracts'}. "
                f"UL Price: {ul_price_start_str} -> {ul_price_end_str}{ul_price_range_str}."
            )
            top_options_in_burst = burst_row.get('TopOptionsInBurst', [])
            if top_options_in_burst:
                report_lines.append("    Key Contracts in this Burst:")
                for opt_detail in top_options_in_burst:
                    iv_start_str = _format_percentage(opt_detail.get('iv_start_trade'))
                    iv_end_str = _format_percentage(opt_detail.get('iv_end_trade'))
                    iv_change_str = _format_signed_percentage_change(opt_detail.get('iv_change_in_burst'))
                    avg_iv_str = _format_percentage(opt_detail.get('avg_iv_in_burst'))
                    avg_delta_str = _format_float(opt_detail.get('avg_delta_in_burst'), 2)

                    report_lines.append(
                        f"      - {opt_detail.get('tos_symbol','N/A')} (DTE: {opt_detail.get('dte_at_burst','N/A')}): "
                        f"Qty: {_format_int(opt_detail.get('total_qty_in_burst',0))}, NetAgg: {_format_int(opt_detail.get('net_agg_qty_in_burst',0))}, "
                        f"AvgIV: {avg_iv_str}, IV: {iv_start_str}->{iv_end_str} (Chg: {iv_change_str}), AvgDelta: {avg_delta_str}"
                    )
            else:
                report_lines.append("    No specific top options detailed for this burst.")
            report_lines.append("") 
    else:
        report_lines.append("  No significant HF bursts identified.")

    for key in ['flow_by_moneyness', 'flow_by_dte', 'iv_by_moneyness', 'iv_by_dte', 'most_active_options_by_volume']:
        report_lines.append(f"\n--- {key.replace('_', ' ').title()} ---")
        data_item = holistic_analysis_results.get(key)
        if isinstance(data_item, (pd.DataFrame, pd.Series)) and not data_item.empty:
            report_lines.append(data_item.to_string(float_format='%.2f'))
        elif data_item is not None:
             report_lines.append(str(data_item))
        else:
            report_lines.append("  Data not available.")
            
    report_lines.append("\n--- End of Report ---")
    return "\n".join(report_lines)
