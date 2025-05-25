# main_app.py
# Main application script for the Options Data Parser & Analyzer.

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import pandas as pd
import os
from datetime import datetime
import numpy as np 

# Import from our modules
import config
import data_utils
import rtd_handler
import analysis_engine 
import ui_builder

# --- Global Variables ---
current_theme = "light"
root = None
style = None
widgets = {} 
tab_ui_widgets = {} 
main_notebook = None

app_data = {
    "cleaned_dfs": {},
    "analysis_outputs": {}, 
    "options_of_interest": {}, 
    "rtd_data": {} 
}

# --- Enhanced Status Logging Function ---
def log_status_to_ui(message, active_ticker_for_log=None, also_to_analysis_area=True):
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    status_label_widget = widgets.get("status_label")
    if status_label_widget and status_label_widget.winfo_exists():
        status_label_widget.config(text=full_message)
    if also_to_analysis_area and active_ticker_for_log:
        try:
            current_tab_ui_local = tab_ui_widgets.get(active_ticker_for_log)
            if current_tab_ui_local:
                analysis_area_widget_local = current_tab_ui_local.get("analysis_results_area")
                if analysis_area_widget_local and analysis_area_widget_local.winfo_exists():
                    analysis_area_widget_local.insert(tk.END, full_message + "\n")
                    analysis_area_widget_local.see(tk.END)
        except Exception: pass 
    if root and root.winfo_exists(): root.update_idletasks()

# --- Helper Functions for Differential Processing ---
def get_last_processed_trade_signature(active_ticker):
    if not os.path.exists(config.LAST_TRADE_META_DIRECTORY):
        os.makedirs(config.LAST_TRADE_META_DIRECTORY)
    meta_file_path = os.path.join(config.LAST_TRADE_META_DIRECTORY, f"{active_ticker}_last_trade.txt")
    try:
        with open(meta_file_path, "r", encoding="utf-8") as f: return f.read().strip()
    except FileNotFoundError: return None

def save_last_processed_trade_signature(active_ticker, last_raw_trade_line):
    if not os.path.exists(config.LAST_TRADE_META_DIRECTORY):
        os.makedirs(config.LAST_TRADE_META_DIRECTORY)
    meta_file_path = os.path.join(config.LAST_TRADE_META_DIRECTORY, f"{active_ticker}_last_trade.txt")
    try:
        with open(meta_file_path, "w", encoding="utf-8") as f: f.write(last_raw_trade_line)
    except Exception as e:
        if tk._default_root and tk._default_root.winfo_exists():
             messagebox.showerror("Metadata Error", f"Could not save last trade signature for {active_ticker}: {e}")

# --- Generic Report Saving Function ---
def _save_text_report(active_ticker, report_content, report_type_suffix):
    if not report_content:
        log_status_to_ui(f"No content to save for {report_type_suffix} report for {active_ticker}.", active_ticker, False)
        return
    try:
        if not os.path.exists(config.ANALYSIS_TXT_OUTPUT_DIRECTORY):
            os.makedirs(config.ANALYSIS_TXT_OUTPUT_DIRECTORY)
        report_datetime_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        txt_filename = f"{active_ticker}_{report_datetime_str}_{report_type_suffix.replace(' ', '_')}.txt"
        full_txt_path = os.path.join(config.ANALYSIS_TXT_OUTPUT_DIRECTORY, txt_filename)
        with open(full_txt_path, "w", encoding="utf-8") as f: f.write(report_content)
        log_status_to_ui(f"{report_type_suffix} report saved: {txt_filename}", active_ticker, False)
    except Exception as e:
        error_msg = f"Error saving {report_type_suffix} report for {active_ticker}: {e}"
        log_status_to_ui(error_msg, active_ticker, False)
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("TXT Report Error", error_msg)

# --- Event Handlers / Callbacks ---
def get_active_ticker_and_ui_widgets():
    global main_notebook, tab_ui_widgets
    if not main_notebook or not main_notebook.tabs(): return None, None
    try:
        current_tab_widget_path = main_notebook.select() 
        for ticker, ui_elements_dict in tab_ui_widgets.items():
            if "tab_frame" in ui_elements_dict and str(ui_elements_dict["tab_frame"]) == current_tab_widget_path:
                return ticker, ui_elements_dict
        return None, None 
    except tk.TclError: return None, None

def _on_rtd_data_ready_for_final_briefing(active_ticker, fetched_rtd_dataframe):
    global app_data, root
    if not active_ticker:
        log_status_to_ui("RTD callback error: No active ticker.")
        return

    log_status_to_ui(f"RTD data received for {active_ticker}. Generating final outputs...", active_ticker)
    app_data['rtd_data'][active_ticker] = fetched_rtd_dataframe 
    current_tab_ui = tab_ui_widgets.get(active_ticker)
    if not current_tab_ui:
        log_status_to_ui(f"Briefing error for {active_ticker}: UI elements not found.", active_ticker); return

    briefing_area_widget = current_tab_ui.get("trade_briefing_area")
    if not briefing_area_widget or not briefing_area_widget.winfo_exists(): 
        log_status_to_ui(f"Briefing error for {active_ticker}: Briefing area widget missing.", active_ticker); return
        
    ui_builder.clear_text_area(briefing_area_widget)
    holistic_analysis_results = app_data['analysis_outputs'].get(active_ticker)
    options_of_interest_symbols_for_briefing = app_data['options_of_interest'].get(active_ticker, [])

    if not holistic_analysis_results or "error" in holistic_analysis_results:
        error_detail = holistic_analysis_results.get('error', 'Unknown analysis error') if holistic_analysis_results else "Historical analysis data not found"
        briefing_area_widget.insert(tk.END, f"Error: {error_detail} for briefing.\nPlease run 'Process & Analyze' first.\n")
        log_status_to_ui(f"Output generation error for {active_ticker}: {error_detail}.", active_ticker); return
    
    log_status_to_ui(f"Generating UI Actionable Briefing for {active_ticker}...", active_ticker)
    trade_briefing_text = analysis_engine.generate_trade_briefing(
        holistic_analysis_results, fetched_rtd_dataframe, active_ticker, options_of_interest_symbols_for_briefing
    )
    briefing_area_widget.insert(tk.END, trade_briefing_text); briefing_area_widget.yview_moveto(0.0) 
    log_status_to_ui(f"UI Actionable Briefing for {active_ticker} generated.", active_ticker)
    _save_text_report(active_ticker, trade_briefing_text, "Actionable_Briefing_Report")
    
    final_status_msg = f"Outputs for {active_ticker} generated (UI briefing & Briefing TXT report)."
    log_status_to_ui(final_status_msg, active_ticker, False)
    if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("Outputs Ready", final_status_msg)

# MOVED handle_fetch_rtd_for_active_tab_button EARLIER
def handle_fetch_rtd_for_active_tab_button():
    global root, app_data, tab_ui_widgets 
    active_ticker, current_tab_ui = get_active_ticker_and_ui_widgets()
    if not active_ticker or not current_tab_ui:
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Error", "No active ticker tab selected."); return
    if rtd_handler.excel_wb_global is None: # Check if Excel is connected
        log_status_to_ui("Excel not connected. Please ensure Excel is connected via 'Process & Analyze' or manually.", active_ticker, False)
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Excel Error", "Not connected to Excel. The 'Process & Analyze' flow attempts connection, or ensure it's manually connected if you skipped that."); return

    rtd_opt_symbols_to_fetch = []
    for i in range(3): 
        entry_widget = current_tab_ui.get(f"rtd_option_entry_{i+1}")
        if entry_widget and entry_widget.winfo_exists() and entry_widget.get().strip():
            rtd_opt_symbols_to_fetch.append(entry_widget.get().strip())
    if not rtd_opt_symbols_to_fetch:
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("Info", "No option symbols entered in RTD fields for current tab."); return
    
    rtd_results_area_widget = current_tab_ui.get("rtd_results_area")
    if not rtd_results_area_widget or not rtd_results_area_widget.winfo_exists():
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("UI Error", "RTD results area not found for current tab."); return
    try:
        rtd_wait_sec = int(widgets["rtd_wait_time_entry"].get()); rtd_wait_ms = rtd_wait_sec * 1000
        if rtd_wait_ms < 1000: rtd_wait_ms = 1000 
    except ValueError:
        rtd_wait_ms = 3000
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showwarning("RTD Wait Time", "Invalid RTD wait time. Using default 3000ms.")
    log_status_to_ui(f"Fetching ad-hoc RTD for {active_ticker} ({len(rtd_opt_symbols_to_fetch)} options)...", active_ticker)
    def _on_adhoc_rtd_ready(ticker, rtd_df):
        app_data['rtd_data'][ticker] = rtd_df 
        log_status_to_ui(f"Ad-hoc RTD data fetched for {ticker}.", ticker, False)
        if rtd_results_area_widget.winfo_exists(): 
            ui_builder.clear_text_area(rtd_results_area_widget)
            rtd_results_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Fetched RTD Data for {ticker} (Ad-hoc):\n")
            if not rtd_df.empty: rtd_results_area_widget.insert(tk.END, rtd_df.to_string() + "\n")
            else: rtd_results_area_widget.insert(tk.END, "No data returned from RTD fetch for these symbols.\n")
            rtd_results_area_widget.see(tk.END)
    rtd_handler.fetch_rtd_data_for_options(root, active_ticker, rtd_opt_symbols_to_fetch,
        rtd_results_area_widget, _on_adhoc_rtd_ready, rtd_wait_ms,
        lambda msg: log_status_to_ui(msg, active_ticker, False))

def handle_process_and_analyze_active_tab():
    global app_data, root
    active_ticker, current_tab_ui = get_active_ticker_and_ui_widgets()
    if not active_ticker or not current_tab_ui:
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Error", "No active ticker tab selected or UI elements not found."); return

    data_text_area_widget = current_tab_ui.get("data_text_area")
    if not data_text_area_widget or not data_text_area_widget.winfo_exists():
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("UI Error", f"Data text area not found for ticker {active_ticker}."); return
    raw_data_full_paste = data_text_area_widget.get("1.0", tk.END).strip()
    if not raw_data_full_paste:
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Error", f"No raw data pasted for ticker {active_ticker}."); return

    log_status_to_ui(f"Starting processing for {active_ticker}...", active_ticker)
    analysis_area_widget = current_tab_ui.get("analysis_results_area")
    briefing_area_widget = current_tab_ui.get("trade_briefing_area")
    ui_builder.clear_text_area(analysis_area_widget); ui_builder.clear_text_area(briefing_area_widget)
    analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Processing data for {active_ticker}...\n")

    excel_connected_for_this_run = False
    if rtd_handler.excel_wb_global is None: 
        log_status_to_ui(f"Attempting to connect to Excel workbook: {config.EXCEL_WORKBOOK_NAME}...", active_ticker)
        if rtd_handler.connect_to_excel(config.EXCEL_WORKBOOK_NAME, lambda msg: log_status_to_ui(msg, active_ticker, False)):
            excel_connected_for_this_run = True
            log_status_to_ui(f"Successfully connected to Excel: {rtd_handler.excel_wb_global.name}", active_ticker)
        else:
            log_status_to_ui(f"Failed to connect to Excel. RTD data will not be live.", active_ticker, False)
    else:
        excel_connected_for_this_run = True 
        log_status_to_ui(f"Already connected to Excel: {rtd_handler.excel_wb_global.name}", active_ticker)

    last_known_trade_signature = get_last_processed_trade_signature(active_ticker)
    raw_lines_full_paste = raw_data_full_paste.split('\n')
    new_raw_lines_to_process = []
    last_raw_line_of_current_paste_for_saving = raw_lines_full_paste[-1].strip() if raw_lines_full_paste else None

    if last_known_trade_signature:
        found_last_trade_idx = -1
        for i, line in enumerate(raw_lines_full_paste):
            if line.strip() == last_known_trade_signature: found_last_trade_idx = i; break
        if found_last_trade_idx != -1:
            new_raw_lines_to_process = [line for line in raw_lines_full_paste[found_last_trade_idx+1:] if line.strip()]
            log_status_to_ui(f"Found last processed trade. Identified {len(new_raw_lines_to_process)} potential new lines.", active_ticker)
        else:
            new_raw_lines_to_process = [line for line in raw_lines_full_paste if line.strip()]
            log_status_to_ui("Previous last trade not found in paste. Processing all pasted lines as new.", active_ticker)
            if tk._default_root and tk._default_root.winfo_exists():
                messagebox.showwarning("Data Sync", "Could not find the last known trade. Processing all lines. Duplicates will be handled.")
    else:
        new_raw_lines_to_process = [line for line in raw_lines_full_paste if line.strip()]
        log_status_to_ui("No previous trade signature. Processing all lines as new.", active_ticker)

    newly_cleaned_df = pd.DataFrame()
    if new_raw_lines_to_process:
        raw_data_for_cleaning = "\n".join(new_raw_lines_to_process)
        log_status_to_ui(f"Cleaning {len(new_raw_lines_to_process)} new raw lines...", active_ticker)
        df_new_cleaned, parsed, skipped, malformed = data_utils.process_raw_data_string(
            raw_data_for_cleaning, config.PRICE_FIELD_INDEX,
            config.EXPECTED_FIELD_COUNT_FULL, config.EXPECTED_FIELD_COUNT_NO_CONDITION)
        summary_msg_new = f"New Data Parsing:\n - Valid New Trades: {parsed}\n - N/A Price Skipped: {skipped}\n - Malformed: {malformed}\n"
        analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {summary_msg_new}")
        log_status_to_ui(f"New data parsing complete: {parsed} valid, {skipped} skipped, {malformed} malformed.", active_ticker)
        if df_new_cleaned is not None and not df_new_cleaned.empty: newly_cleaned_df = df_new_cleaned
    else:
        analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] No new raw lines identified to process.\n")
        log_status_to_ui("No new raw lines identified to process.", active_ticker)

    if not os.path.exists(config.PERSISTENT_CLEANED_DATA_DIRECTORY): os.makedirs(config.PERSISTENT_CLEANED_DATA_DIRECTORY)
    master_df_path = os.path.join(config.PERSISTENT_CLEANED_DATA_DIRECTORY, f"{active_ticker}_master_cleaned_trades.csv")
    existing_master_df = pd.DataFrame()
    if os.path.exists(master_df_path):
        try:
            existing_master_df = pd.read_csv(master_df_path, parse_dates=['Time', 'Expiration_Date'])
            log_status_to_ui(f"Loaded {len(existing_master_df)} trades from master file.", active_ticker)
        except Exception as e:
            if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("File Error", f"Could not load master data: {e}.")
            log_status_to_ui(f"Error loading master data for {active_ticker}.", active_ticker, False)

    consolidated_cleaned_df = pd.DataFrame()
    if not newly_cleaned_df.empty:
        newly_cleaned_df['Time'] = pd.to_datetime(newly_cleaned_df['Time'], errors='coerce')
        newly_cleaned_df['Expiration_Date'] = pd.to_datetime(newly_cleaned_df['Expiration_Date'], errors='coerce')
        consolidated_cleaned_df = pd.concat([existing_master_df, newly_cleaned_df], ignore_index=True)
        key_cols_for_dedupe = ['Time', 'Option_Description_orig', 'TradeQuantity', 'Trade_Price', 'Exchange', 'Aggressor', 'Condition']
        key_cols_for_dedupe = [col for col in key_cols_for_dedupe if col in consolidated_cleaned_df.columns]
        original_row_count = len(consolidated_cleaned_df)
        if key_cols_for_dedupe: consolidated_cleaned_df.drop_duplicates(subset=key_cols_for_dedupe, keep='last', inplace=True)
        deduped_count = original_row_count - len(consolidated_cleaned_df)
        if deduped_count > 0: log_status_to_ui(f"Removed {deduped_count} duplicates from consolidated data.", active_ticker)
        try:
            consolidated_cleaned_df.to_csv(master_df_path, index=False)
            if last_raw_line_of_current_paste_for_saving and not newly_cleaned_df.empty :
                 save_last_processed_trade_signature(active_ticker, last_raw_line_of_current_paste_for_saving)
            log_status_to_ui(f"Master file updated. Total unique trades: {len(consolidated_cleaned_df)}.", active_ticker)
        except Exception as e:
            log_status_to_ui(f"Error saving updated master data: {e}", active_ticker, False)
            consolidated_cleaned_df = newly_cleaned_df if not newly_cleaned_df.empty else existing_master_df 
    elif not existing_master_df.empty:
        consolidated_cleaned_df = existing_master_df
        log_status_to_ui("No new trades. Using existing master data for analysis.", active_ticker)
    else: 
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("No Data", f"No data to process or analyze for {active_ticker}.")
        log_status_to_ui(f"No data for {active_ticker} to analyze.", active_ticker, False); return

    app_data['cleaned_dfs'][active_ticker] = consolidated_cleaned_df 
    if consolidated_cleaned_df.empty:
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("Analysis Info", f"No data available to analyze for {active_ticker}.")
        log_status_to_ui(f"No data to analyze for {active_ticker}.", active_ticker, False)
        ui_builder.clear_text_area(briefing_area_widget); briefing_area_widget.insert(tk.END, "No data available for analysis or briefing.")
        return

    log_status_to_ui(f"Running holistic analysis on {len(consolidated_cleaned_df)} trades...", active_ticker)
    analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] --- Running Holistic Flow Analysis on Consolidated Data ---\n")
    time_window_pandas_freq = widgets["otm_time_window_var"].get() 
    holistic_analysis_results, options_of_interest_symbols = analysis_engine.perform_holistic_flow_analysis(
        consolidated_cleaned_df.copy(), active_ticker, time_window=time_window_pandas_freq)
    app_data['analysis_outputs'][active_ticker] = holistic_analysis_results
    app_data['options_of_interest'][active_ticker] = options_of_interest_symbols
    log_status_to_ui(f"Holistic flow analysis complete for {active_ticker}.", active_ticker)

    if "error" in holistic_analysis_results:
        analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Error: {holistic_analysis_results['error']}\n")
        log_status_to_ui(f"Analysis error: {holistic_analysis_results['error']}", active_ticker, False)
    else: 
        analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Displaying analysis results...\n")
        for key, value in holistic_analysis_results.items():
            analysis_area_widget.insert(tk.END, f"\n--- {key.replace('_', ' ').title()} ---\n")
            if isinstance(value, list) and key == "identified_common_strategies": 
                if not value: analysis_area_widget.insert(tk.END, "None identified.\n")
                else:
                    for i, strategy_item in enumerate(value):
                        analysis_area_widget.insert(tk.END, f"  Strategy {i+1}: {strategy_item.get('description', 'N/A')}\n")
                        for leg_idx, leg in enumerate(strategy_item.get('legs', [])):
                            analysis_area_widget.insert(tk.END, f"    - Leg: {leg.get('TradeQuantity')} {leg.get('Option_Description_orig','N/A')} @ ${float(leg.get('Trade_Price',0)):.2f}\n")
            elif isinstance(value, (pd.DataFrame, pd.Series)):
                if key == 'high_frequency_metrics' and isinstance(value, pd.DataFrame) and len(value) > 100:
                    analysis_area_widget.insert(tk.END, f"(Displaying head and tail of {len(value)} rows)\n")
                    analysis_area_widget.insert(tk.END, value.head().to_string(index=(not isinstance(value, pd.Series))) + "\n...\n")
                    analysis_area_widget.insert(tk.END, value.tail().to_string(index=(not isinstance(value, pd.Series))) + "\n")
                else:
                    analysis_area_widget.insert(tk.END, value.to_string(index=(not isinstance(value, pd.Series))) + "\n")
            elif isinstance(value, dict):
                 for sub_key, sub_val in value.items():
                      analysis_area_widget.insert(tk.END, f"  {str(sub_key).replace('_', ' ').title()}: {sub_val}\n")
            else: analysis_area_widget.insert(tk.END, str(value) + "\n")
    analysis_area_widget.yview_moveto(0.0) 

    if options_of_interest_symbols:
        log_status_to_ui(f"Identified {len(options_of_interest_symbols)} options of interest. Populating RTD fields.", active_ticker)
        analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] --- Identified {len(options_of_interest_symbols)} Options of Interest (ToS Symbols for RTD) ---\n")
        for opt_sym in options_of_interest_symbols: analysis_area_widget.insert(tk.END, f" - {opt_sym}\n")
        for i, opt_symbol in enumerate(options_of_interest_symbols[:3]): 
            entry_widget = current_tab_ui.get(f"rtd_option_entry_{i+1}")
            if entry_widget and entry_widget.winfo_exists():
                entry_widget.delete(0, tk.END); entry_widget.insert(0, opt_symbol)
        
        if rtd_handler.excel_wb_global : 
            try:
                rtd_wait_sec = int(widgets["rtd_wait_time_entry"].get()); rtd_wait_ms = rtd_wait_sec * 1000
                if rtd_wait_ms < 1000: rtd_wait_ms = 1000 
            except ValueError:
                if tk._default_root and tk._default_root.winfo_exists(): messagebox.showwarning("RTD Wait Time", "Invalid RTD wait time. Using default 3000ms.")
                rtd_wait_ms = 3000
            log_status_to_ui(f"Attempting to fetch RTD for {len(options_of_interest_symbols)} options (wait: {rtd_wait_ms/1000:.1f}s)...", active_ticker)
            analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Attempting to fetch RTD data...\n")
            rtd_handler.fetch_rtd_data_for_options(root, active_ticker, options_of_interest_symbols, 
                current_tab_ui.get("rtd_results_area"), _on_rtd_data_ready_for_final_briefing, rtd_wait_ms,
                lambda msg: log_status_to_ui(msg, active_ticker, False))
        else: 
            log_status_to_ui("Excel not connected. Skipping live RTD.", active_ticker)
            analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Excel not connected. Cannot fetch live RTD data.\n")
            if tk._default_root and tk._default_root.winfo_exists(): messagebox.showwarning("RTD Info", "Excel not connected. Outputs will use placeholder RTD.")
            rtd_df_placeholder_data = []
            if options_of_interest_symbols:
                for sym in options_of_interest_symbols:
                    entry = {'ToS Symbol': sym}; 
                    for metric_display_name in rtd_handler.RTD_HEADERS[1:]: entry[metric_display_name] = "N/A (No RTD)"
                    rtd_df_placeholder_data.append(entry)
            rtd_df_placeholder = pd.DataFrame(rtd_df_placeholder_data)
            if rtd_df_placeholder.empty and options_of_interest_symbols: rtd_df_placeholder = pd.DataFrame(columns=rtd_handler.RTD_HEADERS)
            app_data['rtd_data'][active_ticker] = rtd_df_placeholder 
            
            log_status_to_ui(f"Generating UI Briefing for {active_ticker} with placeholder RTD...", active_ticker)
            trade_briefing_text = analysis_engine.generate_trade_briefing(
                holistic_analysis_results, app_data['rtd_data'][active_ticker], active_ticker, options_of_interest_symbols)
            briefing_area_widget.insert(tk.END, trade_briefing_text); briefing_area_widget.yview_moveto(0.0)
            log_status_to_ui(f"UI Briefing for {active_ticker} generated.", active_ticker)
            _save_text_report(active_ticker, trade_briefing_text, "Actionable_Briefing_Report")
            log_status_to_ui(f"Processing complete for {active_ticker} (RTD skipped). Briefing TXT report generated.", active_ticker, False)
    else: 
        log_status_to_ui("No specific options of interest identified.", active_ticker)
        analysis_area_widget.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] No specific options of interest identified.\n")
        briefing_area_widget.insert(tk.END, "\n\n--- Actionable Trade Briefing ---\nNo specific options of interest to generate detailed briefing.")
        minimal_report_content = f"Analysis for {active_ticker} complete. No specific options of interest identified based on current criteria."
        _save_text_report(active_ticker, minimal_report_content, "Analysis_Summary_Report")
        log_status_to_ui(f"Analysis complete for {active_ticker} (no specific options of interest). Summary TXT report generated.", active_ticker, False)
    
    if data_text_area_widget and data_text_area_widget.winfo_exists():
        data_text_area_widget.delete('1.0', tk.END)
        log_status_to_ui(f"Raw data input area cleared for {active_ticker}. Ready for next paste.", active_ticker)

def handle_toggle_theme():
    global current_theme, style, root, widgets, tab_ui_widgets, main_notebook
    if current_theme == "light":
        current_theme = "dark"
    else:
        current_theme = "light"
    
    all_widgets_to_theme = widgets.copy(); all_widgets_to_theme["root"] = root
    if main_notebook: all_widgets_to_theme["main_notebook"] = main_notebook
    for ticker, tab_elements_dict in tab_ui_widgets.items():
        if tab_elements_dict:
            for widget_key, widget_obj in tab_elements_dict.items():
                if isinstance(widget_obj, tk.Widget): all_widgets_to_theme[f"tab_{ticker}_{widget_key}"] = widget_obj
    ui_builder.apply_theme_to_widgets(root, style, current_theme, all_widgets_to_theme)
    log_status_to_ui(f"Theme switched to {current_theme}.", None, False)

def add_new_ticker_tab():
    global main_notebook, tab_ui_widgets, style, root, current_theme, widgets, app_data
    # This function now correctly references handle_fetch_rtd_for_active_tab_button as it's defined earlier
    tab_command_callbacks = { 'fetch_rtd_tab_specific': handle_fetch_rtd_for_active_tab_button }
    
    ticker = simpledialog.askstring("New Ticker", "Enter Ticker Symbol:", parent=root)
    if not ticker: return 
    ticker = ticker.strip().upper()
    if not ticker: 
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Error", "Ticker symbol cannot be empty."); return
    
    for existing_ticker, ui_elements in tab_ui_widgets.items():
        if existing_ticker == ticker:
            for i, tab_id_str in enumerate(main_notebook.tabs()):
                 tab_frame_path_in_notebook = main_notebook.nametowidget(tab_id_str)
                 if "tab_frame" in ui_elements and ui_elements["tab_frame"] == tab_frame_path_in_notebook:
                    main_notebook.select(i)
                    if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("Info", f"Tab for ticker {ticker} already exists. Switched to it.")
                    return

    tab_frame = ttk.Frame(main_notebook); main_notebook.add(tab_frame, text=ticker)
    created_widgets_for_tab = ui_builder.create_ticker_tab_content(tab_frame, ticker, tab_command_callbacks)
    created_widgets_for_tab["tab_frame"] = tab_frame; tab_ui_widgets[ticker] = created_widgets_for_tab

    app_data['cleaned_dfs'][ticker] = pd.DataFrame() 
    app_data['analysis_outputs'][ticker] = {}; app_data['options_of_interest'][ticker] = []
    app_data['rtd_data'][ticker] = {}
    main_notebook.select(tab_frame) 
    
    all_widgets_for_current_tab_theme = widgets.copy(); all_widgets_for_current_tab_theme["root"] = root
    if main_notebook: all_widgets_for_current_tab_theme["main_notebook"] = main_notebook
    if created_widgets_for_tab: 
        for key, widg in created_widgets_for_tab.items():
            if isinstance(widg, tk.Widget): all_widgets_for_current_tab_theme[f"tab_{ticker}_{key}"] = widg
    ui_builder.apply_theme_to_widgets(root, style, current_theme, all_widgets_for_current_tab_theme)
    log_status_to_ui(f"Created and switched to tab for {ticker}.", ticker)

def handle_close_active_tab():
    global main_notebook, tab_ui_widgets, app_data, widgets
    if not main_notebook or not main_notebook.tabs():
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("Info", "No tabs to close."); return
    try:
        selected_tab_path = main_notebook.select(); selected_tab_index = main_notebook.index(selected_tab_path) 
        active_ticker_to_close = None
        for ticker_key, ui_elements in tab_ui_widgets.items():
            if "tab_frame" in ui_elements and str(ui_elements["tab_frame"]) == selected_tab_path:
                active_ticker_to_close = ticker_key; break
        if not active_ticker_to_close: 
            try: active_ticker_to_close = main_notebook.tab(selected_tab_index, "text")
            except tk.TclError:
                if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Error", "Could not determine ticker for selected tab (TclError)."); return
        if not active_ticker_to_close: 
            if tk._default_root and tk._default_root.winfo_exists(): messagebox.showerror("Error", "Could not determine ticker for selected tab."); return
        if tk._default_root and tk._default_root.winfo_exists() and messagebox.askyesno("Confirm Close", f"Are you sure you want to close the tab for {active_ticker_to_close}?"):
            main_notebook.forget(selected_tab_index) 
            if active_ticker_to_close in tab_ui_widgets: del tab_ui_widgets[active_ticker_to_close]
            for key_app_data in ["cleaned_dfs", "analysis_outputs", "options_of_interest", "rtd_data"]:
                if active_ticker_to_close in app_data[key_app_data]:
                    del app_data[key_app_data][active_ticker_to_close]
            log_status_to_ui(f"Closed tab for {active_ticker_to_close}.", None, False)
    except tk.TclError: 
        if tk._default_root and tk._default_root.winfo_exists(): messagebox.showinfo("Info", "No tab currently selected to close.")

# --- Main Application Setup ---
def main():
    global root, style, widgets, main_notebook, app_data, tab_ui_widgets, current_theme
    root = tk.Tk(); root.title("Options Flow & RTD Analyzer v3.3 (Streamlined Workflow)"); root.geometry("1050x950") 
    style = ttk.Style()
    available_themes = style.theme_names()
    if 'clam' in available_themes: style.theme_use('clam')
    elif 'alt' in available_themes: style.theme_use('alt')
    elif 'vista' in available_themes and os.name == 'nt': style.theme_use('vista')
    
    top_control_frame = ttk.Frame(root, padding=5); top_control_frame.pack(fill=tk.X, pady=(5,0), padx=5)
    widgets["top_control_frame"] = top_control_frame
    
    widgets["add_ticker_button"] = ttk.Button(top_control_frame, text="Add Ticker Tab", command=add_new_ticker_tab)
    widgets["add_ticker_button"].pack(side=tk.LEFT, padx=2)
    widgets["run_analysis_button"] = ttk.Button(top_control_frame, text="Process & Analyze Active Tab", command=handle_process_and_analyze_active_tab)
    widgets["run_analysis_button"].pack(side=tk.LEFT, padx=2) 
    widgets["close_tab_button"] = ttk.Button(top_control_frame, text="Close Active Tab", command=handle_close_active_tab)
    widgets["close_tab_button"].pack(side=tk.LEFT, padx=2)
    
    excel_info_frame = ttk.Frame(top_control_frame); excel_info_frame.pack(side=tk.LEFT, padx=10) 
    widgets["excel_info_frame"] = excel_info_frame
    widgets["excel_wb_label"] = ttk.Label(excel_info_frame, text="Target Excel WB:"); 
    widgets["excel_wb_label"].pack(side=tk.LEFT, padx=(0,2))
    widgets["excel_wb_entry"] = ttk.Entry(excel_info_frame, width=25); 
    widgets["excel_wb_entry"].insert(0, config.EXCEL_WORKBOOK_NAME); 
    widgets["excel_wb_entry"].config(state='readonly') 
    widgets["excel_wb_entry"].pack(side=tk.LEFT, padx=2)
    
    widgets["rtd_wait_time_label"] = ttk.Label(excel_info_frame, text="RTD Wait (s):")
    widgets["rtd_wait_time_label"].pack(side=tk.LEFT, padx=(5,0))
    widgets["rtd_wait_time_entry"] = ttk.Entry(excel_info_frame, width=4)
    widgets["rtd_wait_time_entry"].insert(0, "3") 
    widgets["rtd_wait_time_entry"].pack(side=tk.LEFT, padx=2)

    ttk.Separator(top_control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
    global_filter_frame = ttk.Frame(top_control_frame); global_filter_frame.pack(side=tk.LEFT, padx=5)
    widgets["global_filter_frame"] = global_filter_frame
    widgets["otm_time_window_label"] = ttk.Label(global_filter_frame, text="Bin Window:")
    widgets["otm_time_window_label"].pack(side=tk.LEFT, padx=(0,2))
    widgets["otm_time_window_var"] = tk.StringVar(root)
    otm_time_window_choices = ["1min", "5min", "15min", "30min", "1H"] 
    widgets["otm_time_window_var"].set("15min")
    widgets["otm_time_window_menu"] = ttk.OptionMenu(global_filter_frame, widgets["otm_time_window_var"], widgets["otm_time_window_var"].get(), *otm_time_window_choices)
    widgets["otm_time_window_menu"].pack(side=tk.LEFT)
    
    widgets["theme_toggle_button"] = ttk.Button(top_control_frame, text="Toggle Theme", command=handle_toggle_theme)
    widgets["theme_toggle_button"].pack(side=tk.RIGHT, padx=5)

    main_notebook = ttk.Notebook(root); main_notebook.pack(expand=True, fill='both', padx=5, pady=(0,5))
    widgets["main_notebook"] = main_notebook
    widgets["status_label"] = ttk.Label(root, text="Ready.", relief=tk.SUNKEN, anchor='w', padding=(3,3))
    widgets["status_label"].pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0,5))
    
    if not os.path.exists(config.PERSISTENT_CLEANED_DATA_DIRECTORY): os.makedirs(config.PERSISTENT_CLEANED_DATA_DIRECTORY)
    if not os.path.exists(config.LAST_TRADE_META_DIRECTORY): os.makedirs(config.LAST_TRADE_META_DIRECTORY)
    if not os.path.exists(config.ANALYSIS_TXT_OUTPUT_DIRECTORY): os.makedirs(config.ANALYSIS_TXT_OUTPUT_DIRECTORY)

    all_initial_widgets = widgets.copy(); all_initial_widgets["root"] = root
    all_initial_widgets["main_notebook"] = main_notebook
    ui_builder.apply_theme_to_widgets(root, style, current_theme, all_initial_widgets)
    log_status_to_ui("Application started. Ready.", None, False)
    root.mainloop()

if __name__ == '__main__':
    main()
