# rtd_handler.py
# Functions related to xlwings, connecting to Excel, and fetching RTD data.

import xlwings as xw
import re
import tkinter as tk 
from tkinter import messagebox 
from datetime import datetime
import pandas as pd
import numpy as np
import config # For EXCEL_RTD_SHEET_NAME

excel_wb_global = None

RTD_METRICS_ORDERED = {
    "LAST": "Last", "BID": "Bid", "ASK": "Ask", "VOLUME": "Volume",
    "OPEN_INT": "Open Int", "IMPLIED_VOLATILITY": "IV (%)", 
    "DELTA": "Delta", "GAMMA": "Gamma", "THETA": "Theta", "VEGA": "Vega"
}
RTD_HEADERS = ["ToS Symbol"] + list(RTD_METRICS_ORDERED.values())

def format_tos_rtd_symbol(underlying_ticker, expiration_date, strike_price, option_type):
    if not all([underlying_ticker, expiration_date, strike_price is not None, option_type]):
        return None
    try:
        if not isinstance(expiration_date, (datetime, pd.Timestamp)):
            exp_date_dt = pd.to_datetime(expiration_date, errors='coerce')
            if pd.NaT == exp_date_dt: return None
        else:
            exp_date_dt = expiration_date

        year_yy = exp_date_dt.strftime('%y')
        month_mm = exp_date_dt.strftime('%m')
        day_dd = exp_date_dt.strftime('%d')
        
        current_strike_price = float(strike_price)
        if current_strike_price == int(current_strike_price): 
            strike_str = str(int(current_strike_price))
        else: 
            strike_str = str(current_strike_price)

        return f".{underlying_ticker.strip().upper()}{year_yy}{month_mm}{day_dd}{option_type.strip().upper()}{strike_str}"
    except Exception as e:
        return None

def parse_tos_option_symbol(symbol_str):
    if not symbol_str or not isinstance(symbol_str, str) or not symbol_str.startswith('.'):
        return None
    match = re.match(r"^\.(?P<ticker>[A-Z0-9/]+)(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})(?P<type>[CP])(?P<strike>\d+\.?\d*)$", symbol_str.upper())
    if match:
        data = match.groupdict()
        try:
            return {
                "ticker": data["ticker"], "exp_year": int(data["year"]) + 2000, 
                "exp_month": int(data["month"]), "exp_day": int(data["day"]), 
                "option_type": data["type"], "strike_price": float(data["strike"])
            }
        except ValueError: 
            return None
    return None

def connect_to_excel(workbook_name, status_update_func=None):
    global excel_wb_global
    if excel_wb_global is not None: # Already connected
        if status_update_func: status_update_func(f"Already connected to Excel: {excel_wb_global.name}")
        return True

    if not workbook_name:
        msg = "Error: Excel Workbook Name cannot be empty."
        if status_update_func: status_update_func(msg)
        if tk._default_root and tk._default_root.winfo_exists(): 
            messagebox.showerror("Input Error", "Excel Workbook Name cannot be empty.")
        return False
    try:
        excel_wb_global = xw.Book(workbook_name) 
        msg = f"Connected to Excel Workbook: {excel_wb_global.name}"
        if status_update_func: status_update_func(msg)
        if tk._default_root and tk._default_root.winfo_exists():
             messagebox.showinfo("Excel Connection", f"Successfully connected to Workbook: {excel_wb_global.name}")
        return True
    except Exception as e_specific:
        try:
            if not xw.apps.count > 0: 
                raise Exception("No Excel application instance found. Please ensure Excel is running.")
            if not xw.books:
                 raise Exception("Excel is running, but no workbooks are open.")
            excel_wb_global = xw.books.active 
            if not excel_wb_global:
                raise Exception("No active Excel workbook found, though Excel app is running and has open books.")
            msg = f"Connected to ACTIVE Excel Workbook: {excel_wb_global.name} (Verify this is correct)"
            if status_update_func: status_update_func(msg)
            if tk._default_root and tk._default_root.winfo_exists():
                 messagebox.showwarning("Excel Connection", msg)
            return True
        except Exception as e_active:
            msg = f"Could not connect to Excel. Error: {e_active} (Original error for specific name: {e_specific})"
            if status_update_func: status_update_func(f"Error: {msg}")
            if tk._default_root and tk._default_root.winfo_exists():
                messagebox.showerror("Excel Connection Error", msg)
            excel_wb_global = None; return False

def get_or_create_rtd_sheet(): # MODIFIED: No longer takes ticker_symbol
    """Gets or creates the single, predefined RTD sheet."""
    global excel_wb_global
    if excel_wb_global is None: return None
    
    sheet_name = config.EXCEL_RTD_SHEET_NAME # Use predefined sheet name from config

    try:
        sheet = excel_wb_global.sheets[sheet_name]
        sheet.clear_contents() 
    except Exception: 
        if excel_wb_global.sheets:
            sheet = excel_wb_global.sheets.add(name=sheet_name, after=excel_wb_global.sheets[-1])
        else: 
            sheet = excel_wb_global.sheets.add(name=sheet_name)
            
    sheet.range("A1").value = RTD_HEADERS
    sheet.range("A1").expand('right').font.bold = True
    sheet.autofit('columns') 
    return sheet

def fetch_rtd_data_for_options(root_tk_or_none, active_ticker, options_of_interest_symbols,
                               rtd_results_text_widget_or_none, 
                               on_fetch_complete_callback, 
                               user_defined_delay_ms,
                               status_update_func=None): 
    global excel_wb_global 

    if excel_wb_global is None: # Should be handled by main_app now before calling this
        msg = "Error: Not connected to any Excel workbook for RTD."
        if status_update_func: status_update_func(msg)
        if tk._default_root and tk._default_root.winfo_exists() and rtd_results_text_widget_or_none: messagebox.showerror("Excel Error", msg)
        if on_fetch_complete_callback: on_fetch_complete_callback(active_ticker, pd.DataFrame(columns=RTD_HEADERS)) 
        return

    target_sheet = get_or_create_rtd_sheet() # MODIFIED: Uses the single sheet
    if target_sheet is None:
        msg = f"Error: Could not get or create the master RTD Excel sheet ('{config.EXCEL_RTD_SHEET_NAME}')."
        if status_update_func: status_update_func(msg)
        if tk._default_root and tk._default_root.winfo_exists() and rtd_results_text_widget_or_none: messagebox.showerror("Excel Error", msg)
        if on_fetch_complete_callback: on_fetch_complete_callback(active_ticker, pd.DataFrame(columns=RTD_HEADERS))
        return

    if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
        # rtd_results_text_widget_or_none.delete('1.0', tk.END) # Clearing might be done by main_app logger now
        rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Populating RTD formulas for {len(options_of_interest_symbols)} options for {active_ticker} into sheet '{target_sheet.name}'...\n")
        if root_tk_or_none and root_tk_or_none.winfo_exists(): root_tk_or_none.update_idletasks()
    
    for i, tos_symbol in enumerate(options_of_interest_symbols):
        if not tos_symbol or not isinstance(tos_symbol, str) or not tos_symbol.startswith('.'):
            if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
                rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Skipping invalid ToS symbol: '{tos_symbol}'\n")
            continue
        
        row_num_excel = i + 2 
        target_sheet.range(f"A{row_num_excel}").value = tos_symbol
        
        for col_idx_metric, rtd_field_code in enumerate(RTD_METRICS_ORDERED.keys()):
            excel_col_letter = xw.utils.col_name(col_idx_metric + 2) 
            formula_cell_ref = f"{excel_col_letter}{row_num_excel}"
            try:
                target_sheet.range(formula_cell_ref).formula = f'=RTD("tos.rtd",,"{rtd_field_code}","{tos_symbol}")'
            except Exception as e_formula:
                if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
                    rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Error writing formula for {tos_symbol} {rtd_field_code}: {e_formula}\n")

    if status_update_func: status_update_func(f"Formulas written. Forcing Excel calculation for {active_ticker}...")
    if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
        rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Forcing Excel calculation...\n")
        if root_tk_or_none and root_tk_or_none.winfo_exists(): root_tk_or_none.update_idletasks()

    try:
        excel_wb_global.app.calculate() 
        if status_update_func: status_update_func(f"Excel calculation triggered for {active_ticker}.")
        if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
            rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Excel calculation triggered. Waiting for updates...\n")
    except Exception as e_calc:
        msg = f"Error forcing Excel calculation: {e_calc}"
        if status_update_func: status_update_func(msg)
        if tk._default_root and tk._default_root.winfo_exists() and rtd_results_text_widget_or_none: messagebox.showerror("Excel Error", msg)
        if on_fetch_complete_callback: on_fetch_complete_callback(active_ticker, pd.DataFrame(columns=RTD_HEADERS))
        return
    
    wait_time_seconds_float = user_defined_delay_ms / 1000.0
    if status_update_func: status_update_func(f"Waiting {wait_time_seconds_float:.1f}s for RTD updates for {active_ticker}...")

    def _read_values_and_callback():
        if status_update_func: status_update_func(f"Reading RTD values for {active_ticker} from sheet '{target_sheet.name}'...")
        if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
             rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Reading RTD values...\n")
        
        fetched_data_df = pd.DataFrame(columns=RTD_HEADERS) 
        if not options_of_interest_symbols: 
            if on_fetch_complete_callback: on_fetch_complete_callback(active_ticker, fetched_data_df)
            return

        last_row_to_read_excel = len(options_of_interest_symbols) + 1 
        last_col_letter_excel = xw.utils.col_name(len(RTD_HEADERS)) 
        data_range_str = f"A2:{last_col_letter_excel}{last_row_to_read_excel}"
        
        try:
            data_values_from_excel = target_sheet.range(data_range_str).options(pd.DataFrame, header=False, index=False, numbers=float, empty='').value 
            if data_values_from_excel is not None and not data_values_from_excel.empty:
                data_values_from_excel.columns = RTD_HEADERS 
                fetched_data_df = data_values_from_excel
        except Exception as e_read:
            if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
                rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] Error reading bulk RTD data: {e_read}\n")

        if 'IV (%)' in fetched_data_df.columns:
            def clean_iv_rtd(val):
                if pd.isna(val): return np.nan 
                s_val = str(val).strip()
                if not s_val: return np.nan 
                is_percentage_format = s_val.endswith('%')
                try:
                    num_val = float(s_val.rstrip('%')) if is_percentage_format else float(s_val)
                    return num_val / 100.0 if is_percentage_format or (abs(num_val) > 1.5) else num_val
                except ValueError: return np.nan 
            fetched_data_df['IV (%)'] = fetched_data_df['IV (%)'].apply(clean_iv_rtd)

        if rtd_results_text_widget_or_none and rtd_results_text_widget_or_none.winfo_exists():
            rtd_results_text_widget_or_none.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] --- RTD Fetch for {active_ticker} Complete. {len(fetched_data_df)} options processed. ---\n")
            if not fetched_data_df.empty:
                rtd_results_text_widget_or_none.insert(tk.END, "Sample of fetched RTD data (first 5 rows):\n")
                rtd_results_text_widget_or_none.insert(tk.END, fetched_data_df.head().to_string() + "\n")
            else:
                rtd_results_text_widget_or_none.insert(tk.END, "No data rows returned from Excel for RTD.\n")
            rtd_results_text_widget_or_none.see(tk.END) # Scroll to end

        if status_update_func: status_update_func(f"RTD data fetch for {active_ticker} complete.")
        if on_fetch_complete_callback:
            on_fetch_complete_callback(active_ticker, fetched_data_df)

    if root_tk_or_none and isinstance(root_tk_or_none, tk.Tk) and root_tk_or_none.winfo_exists():
        root_tk_or_none.after(user_defined_delay_ms, _read_values_and_callback)
    else:
        if on_fetch_complete_callback:
            _read_values_and_callback()
