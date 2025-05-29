# data_utils.py - Enhanced version with TOS-specific processing
"""
Enhanced data parsing, processing, and cleaning utilities for Options Flow Analyzer
Now includes specialized TOS Options Time & Sales processing with incremental updates
"""

import pandas as pd
import numpy as np
import re
import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
from io import StringIO
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# --- Assumed constants from config.py ---
PRICE_FIELD_INDEX = 3
EXPECTED_FIELD_COUNT_FULL = 10
EXPECTED_FIELD_COUNT_NO_CONDITION = 9
# --- Helper functions (safe_float_conversion, safe_int_conversion, parse_option_description, parse_market_at_trade) ---
# (Keep your existing helper functions as provided in the previous turn)
# Make sure parse_option_description returns: Expiration_Date, Strike_Price, Option_Type
# Make sure parse_market_at_trade returns: Option_Bid, Option_Ask

def safe_float_conversion(value_str, is_percentage=False):
    if pd.isna(value_str) or isinstance(value_str, str) and value_str.strip().upper() == "N/A": return np.nan
    try:
        cleaned_value_str = str(value_str).strip()
        if is_percentage:
            if cleaned_value_str.endswith('%'): return float(cleaned_value_str.rstrip('%')) / 100.0
            else: 
                val = float(cleaned_value_str)
                return val / 100.0 if abs(val) > 1.01 else val 
        return float(cleaned_value_str)
    except (ValueError, TypeError): return np.nan

def safe_int_conversion(value_str):
    if pd.isna(value_str) or isinstance(value_str, str) and value_str.strip().upper() == "N/A": return np.nan 
    try: return int(str(value_str).replace(',', '').strip())
    except (ValueError, TypeError): return np.nan 

def parse_option_description(desc_str):
    if pd.isna(desc_str): return pd.NaT, np.nan, None
    desc_str = str(desc_str).strip()
    option_type = None
    strike_price = np.nan
    expiration_date = pd.NaT
    exp_date_str_to_parse = None
    match = re.match(r"^(?P<date_str>\d{1,2}\s+[A-Z]{3}\s+\d{2})\s+(?P<strike>\d+\.?\d*)\s*(?P<type>[CP])$", desc_str.upper().strip())
    if match:
        parts = match.groupdict()
        exp_date_str_to_parse = parts['date_str']
        strike_price = safe_float_conversion(parts['strike'])
        option_type = parts['type'] 
    else:
        strike_type_match_fallback = re.search(r"(\d+\.?\d*)\s*([CP])$", desc_str.upper().strip())
        if strike_type_match_fallback:
            strike_price = safe_float_conversion(strike_type_match_fallback.group(1))
            option_type = strike_type_match_fallback.group(2)
            exp_date_str_to_parse = desc_str[:strike_type_match_fallback.start()].strip()
        elif desc_str and desc_str[-1] in ['C', 'P']:
             option_type = desc_str[-1]
             exp_date_str_to_parse = desc_str[:-1].strip()
        else:
            exp_date_str_to_parse = desc_str 
    if exp_date_str_to_parse:
        try: 
            expiration_date = pd.to_datetime(exp_date_str_to_parse, format='%d %b %y')
        except (ValueError, TypeError): 
            try: 
                expiration_date = pd.to_datetime(exp_date_str_to_parse)
            except (ValueError, TypeError): pass
    return expiration_date, strike_price, option_type

def parse_market_at_trade(market_str):
    if pd.isna(market_str) or not isinstance(market_str, str) or market_str.strip().upper() == "N/AXN/A" or 'x' not in market_str.lower(): return np.nan, np.nan
    match = re.match(r"^\s*([\d\.\-]+|N/A)?\s*[xX]\s*([\d\.\-]+|N/A)?\s*$", market_str.strip(), re.IGNORECASE)
    if not match: return np.nan, np.nan
    bid_str, ask_str = match.groups()
    bid = safe_float_conversion(bid_str) if bid_str else np.nan
    ask = safe_float_conversion(ask_str) if ask_str else np.nan
    return bid, ask

def process_raw_data_string(raw_data_string):
    lines = raw_data_string.strip().split('\n')
    list_of_trade_records = []
    skipped_na_price_lines = 0
    parsed_lines = 0
    malformed_lines = 0

    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue
        fields = line.split('\t')
        if len(fields) > PRICE_FIELD_INDEX and fields[PRICE_FIELD_INDEX].strip().upper() == "N/A":
            skipped_na_price_lines += 1
            continue 
        record = None
        if len(fields) == EXPECTED_FIELD_COUNT_NO_CONDITION:
            record = {'Time_str': fields[0],'Option_Description_str': fields[1],'TradeQuantity_str': fields[2],'Trade_Price_str': fields[3],'Exchange_str': fields[4],'Market_at_trade_str': fields[5],'Delta_str': fields[6],'IV_str': fields[7],'Underlying_Price_str': fields[8],'Condition_str': None}
        elif len(fields) >= EXPECTED_FIELD_COUNT_FULL:
            record = {'Time_str': fields[0],'Option_Description_str': fields[1],'TradeQuantity_str': fields[2],'Trade_Price_str': fields[3],'Exchange_str': fields[4],'Market_at_trade_str': fields[5],'Delta_str': fields[6],'IV_str': fields[7],'Underlying_Price_str': fields[8],'Condition_str': fields[9].strip() if len(fields) > 9 and fields[9].strip() else None}
        
        if record:
            for key, value in record.items():
                if isinstance(value, str): record[key] = value.strip()
            list_of_trade_records.append(record)
            parsed_lines += 1
        else: 
            print(f"Warning: Line {line_number} (non-N/A price) has {len(fields)} fields, expected {EXPECTED_FIELD_COUNT_NO_CONDITION} or >= {EXPECTED_FIELD_COUNT_FULL}. Line: '{line}'")
            malformed_lines +=1
            
    print(f"Info: From data string, processed {parsed_lines} valid trade lines, skipped {skipped_na_price_lines} N/A price lines, {malformed_lines} lines were malformed.")
    
    if not list_of_trade_records: 
        return pd.DataFrame(), parsed_lines, skipped_na_price_lines, malformed_lines

    df = pd.DataFrame(list_of_trade_records)
    
    # --- TIME PROCESSING ---
    # Create Time_dt (datetime object) needed by AlphaExtractor
    df['Time_dt'] = pd.to_datetime(df['Time_str'], errors='coerce')
    # Create 'Time' (string formatted) for display/CSV as per your example
    df['Time'] = df['Time_dt'].apply(lambda x: f"{x.month}/{x.day}/{x.year} {x.hour:02d}:{x.minute:02d}" if pd.notna(x) else "")

    # --- PARSE OPTION DESCRIPTION & MARKET DATA ---
    parsed_desc_cols = df['Option_Description_str'].apply(lambda x: pd.Series(parse_option_description(x), index=['Expiration_Date', 'Strike_Price', 'Option_Type']))
    df = pd.concat([df, parsed_desc_cols], axis=1)
    
    parsed_market_cols = df['Market_at_trade_str'].apply(lambda x: pd.Series(parse_market_at_trade(x), index=['Option_Bid', 'Option_Ask']))
    df = pd.concat([df, parsed_market_cols], axis=1)

    # --- RENAME & CONVERT BASE COLUMNS ---
    # Rename Option_Description_str to StandardOptionSymbol for AlphaExtractor
    df.rename(columns={
        'Option_Description_str': 'StandardOptionSymbol', 
        'Condition_str': 'Condition', 
        'Exchange_str': 'Exchange'
    }, inplace=True)

    # Convert raw string columns for numeric data
    df['IV'] = df['IV_str'].apply(lambda x: safe_float_conversion(x, is_percentage=True))
    df['TradeQuantity'] = df['TradeQuantity_str'].apply(safe_int_conversion)
    df['Trade_Price'] = df['Trade_Price_str'].apply(safe_float_conversion)
    df['Delta'] = df['Delta_str'].apply(safe_float_conversion)
    df['Underlying_Price'] = df['Underlying_Price_str'].apply(safe_float_conversion)

    # Ensure base numeric types for calculation (before NotionalValue)
    # Note: Strike_Price, Option_Bid, Option_Ask are already float/NaN from their parsing functions
    df['TradeQuantity'] = pd.to_numeric(df['TradeQuantity'], errors='coerce')
    df['Trade_Price'] = pd.to_numeric(df['Trade_Price'], errors='coerce')
    
    # --- CALCULATE NOTIONAL VALUE ---
    # Requires TradeQuantity and Trade_Price to be numeric
    df['NotionalValue'] = df['TradeQuantity'].astype(float) * df['Trade_Price'] * 100.0
    df['NotionalValue'] = df['NotionalValue'].fillna(0.0) # Fill NaN notional values, e.g. if price or qty was NaN

    # --- DERIVE AGGRESSOR ---
    aggressor_conditions = [
        (df['Trade_Price'].notna() & df['Option_Ask'].notna() & (df['Trade_Price'] == df['Option_Ask']) & (df['Trade_Price'] > 0)),
        (df['Trade_Price'].notna() & df['Option_Bid'].notna() & (df['Trade_Price'] == df['Option_Bid']) & (df['Trade_Price'] > 0)),
        (df['Trade_Price'].notna() & df['Option_Bid'].notna() & df['Option_Ask'].notna() & (df['Trade_Price'] > df['Option_Bid']) & (df['Trade_Price'] < df['Option_Ask']))
    ]
    aggressor_choices = ['Buyer (At Ask)', 'Seller (At Bid)', 'Between Bid/Ask']
    df['Aggressor'] = np.select(aggressor_conditions, aggressor_choices, default='Unknown/No Clear Market')
    
    # --- FINAL TYPE ENSURE & COLUMN SELECTION ---
    # Ensure final numeric types for columns that will be returned
    final_numeric_cols = [
        'Strike_Price', 'Trade_Price', 'Option_Bid', 'Option_Ask', 
        'Delta', 'IV', 'Underlying_Price', 'NotionalValue'
    ]
    for col in final_numeric_cols:
        if col in df.columns: # Check if column exists (it should if parsing was successful)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'TradeQuantity' in df.columns:
        df['TradeQuantity'] = df['TradeQuantity'].astype('Int64', errors='ignore') # Use nullable Int64

    # Define all columns that should be in the DataFrame returned to the caller (e.g., AlphaExtractor)
    # This includes everything AlphaExtractor needs and other useful parsed fields.
    columns_for_return_df = [
        'Time_dt',              # datetime object for calculations
        'Time',                 # String formatted time for display
        'StandardOptionSymbol', # Renamed from Option_Description_str
        'Expiration_Date',      # Parsed
        'Strike_Price',         # Parsed
        'Option_Type',          # Parsed
        'TradeQuantity',
        'Trade_Price',
        'Option_Bid',           # Parsed
        'Option_Ask',           # Parsed
        'Delta',
        'IV',
        'Underlying_Price',
        'Condition',
        'Exchange',
        'NotionalValue',        # Calculated
        'Aggressor'             # Derived
    ]
    
    # Ensure all these columns exist in the DataFrame, fill with a default if any are missing
    # (should not happen if all parsing steps are correct)
    final_df = pd.DataFrame(columns=columns_for_return_df) # Create an empty DF with desired columns and order
    for col in columns_for_return_df:
        if col in df.columns:
            final_df[col] = df[col]
        else:
            # Assign appropriate NaN type based on expected dtype if known, else np.nan
            if col in ['Time_dt', 'Expiration_Date']:
                 final_df[col] = pd.NaT
            elif col == 'TradeQuantity':
                 final_df[col] = pd.NA # For Int64
            else: # For float or object columns
                 final_df[col] = np.nan
    
    return final_df, parsed_lines, skipped_na_price_lines, malformed_lines


# Configuration constants
EXPECTED_COLUMNS = [
    'Time', 'StandardOptionSymbol', 'TradeQuantity', 'Trade_Price',
    'Aggressor', 'Exchange', 'IV', 'Delta', 'Underlying_Price'
]

OPTIONAL_COLUMNS = [
    'Option_Bid', 'Option_Ask', 'Volume', 'OpenInterest', 'Gamma',
    'Theta', 'Vega', 'Rho', 'Condition', 'Size', 'NotionalValue'
]

# TOS-specific output columns (modified to match your expected output)
TOS_OUTPUT_COLUMNS = [
    'Time', 'Option_Description_orig', 'Expiration_Date', 'Strike_Price',
    'Option_Type', 'TradeQuantity', 'Trade_Price', 'Option_Bid',
    'Option_Ask', 'Delta', 'IV', 'Underlying_Price', 'Condition', 'Exchange',
    'Aggressor'  # Added Aggressor column
]

def _default_log(message: str, ticker: str = None, is_error: bool = False):
    """Default logging function"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = "ERROR" if is_error else "INFO"
    print(f"[DATA_UTILS] [{timestamp}] [{prefix}] {message}")

# Enhanced TOS-specific processing functions
def derive_aggressor(row: pd.Series, tolerance: float = 1e-9) -> str:
    """Derive aggressor status based on trade price vs bid/ask."""
    trade_price = row.get('Trade_Price')
    bid_price = row.get('Option_Bid')
    ask_price = row.get('Option_Ask')

    if pd.isna(trade_price):
        return ""

    aggressor = ""

    valid_bid = pd.notna(bid_price) and bid_price > 0
    valid_ask = pd.notna(ask_price) and ask_price > 0

    if valid_bid and valid_ask:
        if np.isclose(trade_price, bid_price, atol=tolerance) and trade_price < ask_price:
            aggressor = "Seller (At Bid)"
        elif np.isclose(trade_price, ask_price, atol=tolerance) and trade_price > bid_price:
            aggressor = "Buyer (At Ask)"
        elif bid_price < trade_price < ask_price:
            aggressor = "Between Bid/Ask"
        elif np.isclose(trade_price, bid_price, atol=tolerance) and np.isclose(trade_price, ask_price, atol=tolerance): # Trade price equals bid and ask
             aggressor = "At Bid/Ask Spread" # Or could be "Between Bid/Ask" or empty based on preference
        elif trade_price <= bid_price: # Trade is at or below bid
            aggressor = "Seller (At Bid)"
        elif trade_price >= ask_price: # Trade is at or above ask
            aggressor = "Buyer (At Ask)"
        else: # Default for other cases within bid/ask not caught, or if logic needs more complex handling for wide spreads etc.
            aggressor = "" # Or "Unknown"
    elif valid_bid and np.isclose(trade_price, bid_price, atol=tolerance):
        aggressor = "Seller (At Bid)"
    elif valid_ask and np.isclose(trade_price, ask_price, atol=tolerance):
        aggressor = "Buyer (At Ask)"
    
    # If condition already indicates something specific, it might take precedence or be combined
    # For now, we rely purely on price comparison as per your example output.
    
    return aggressor

def detect_data_source_type(raw_data: str) -> str:
    """Detect if data is from TOS OTS window or other source"""
    if not raw_data or not raw_data.strip():
        return "unknown"
    
    lines = raw_data.strip().split('\n')
    if not lines:
        return "unknown"
    
    # Check first few lines for TOS OTS patterns
    sample_lines = lines[:min(5, len(lines))]
    
    tos_indicators = 0
    for line in sample_lines:
        parts = line.split('\t')
        if len(parts) >= 8:  # TOS OTS has many tab-separated fields
            # Check for time pattern (HH:MM:SS)
            if re.match(r'\d{1,2}:\d{2}:\d{2}', parts[0]):
                tos_indicators += 1
            # Check for option description pattern (e.g., "30 MAY 25 200 P")
            if len(parts) > 1 and re.match(r'\d{1,2}\s+[A-Z]{3}\s+\d{2}\s+\d+\.?\d*\s+[CP]', parts[1]):
                tos_indicators += 1
            # Check for exchange names
            if len(parts) > 4 and parts[4] in ['CBOE', 'PHLX', 'ISE', 'AMEX', 'BOX', 'MIAX', 'BEST']:
                tos_indicators += 1
    
    return "tos_ots" if tos_indicators >= 2 else "generic"

# Enhanced data processing functions in main_app.py

def process_monitoring_data(ticker, data_dict):
    """Enhanced processing of monitoring data with incremental updates"""
    
    # Update statistics display
    tab_widgets = tab_ui_widgets.get(ticker)
    if not tab_widgets:
        return
    
    # Update monitoring statistics
    if app_data["monitoring_manager"]:
        status = app_data["monitoring_manager"].get_monitoring_status(ticker)
        ui_builder.update_monitoring_statistics(ticker, tab_widgets, status)
    
    # Update option statistics if available
    if "option_statistics" in data_dict and data_dict["option_statistics"]:
        if "option_stats_display" in tab_widgets:
            stats_display = tab_widgets["option_stats_display"]
            stats_display.config(state=tk.NORMAL)
            stats_display.delete('1.0', tk.END)
            stats_display.insert(tk.END, data_dict["option_statistics"])
            stats_display.config(state=tk.DISABLED)
    
    # Process new trades if available
    if data_dict.get('new_trades_count', 0) > 0:
        cleaned_df = data_dict.get('cleaned_data')
        
        if cleaned_df is not None and not cleaned_df.empty:
            # Update the cleaned data for the ticker
            if ticker not in app_data["cleaned_dfs"]:
                app_data["cleaned_dfs"][ticker] = cleaned_df
            else:
                # Append new data
                app_data["cleaned_dfs"][ticker] = pd.concat([
                    app_data["cleaned_dfs"][ticker], 
                    cleaned_df
                ], ignore_index=True)
            
            # Update data preview
            update_data_preview(ticker, cleaned_df, is_incremental=True)
            
            # Auto-analyze if configured
            if config.USER_AUTO_PROCESS_AFTER_GRAB:
                # Run analysis in background
                handle_run_analysis(ticker, auto_triggered=True)
            
            # Feed to alpha monitor if active
            if ticker in app_data["alpha_monitors"]:
                try:
                    app_data["alpha_monitors"][ticker].add_new_trades(cleaned_df)
                except Exception as e:
                    log_status_to_ui(f"Error feeding data to alpha monitor: {e}", 
                                    ticker, is_error=True)
            
            # Update last trade timestamp
            if not cleaned_df.empty:
                last_trade_time = cleaned_df.iloc[-1]['Time']
                app_data["last_trade_timestamps"][ticker] = last_trade_time
                
            log_status_to_ui(
                f"Processed {data_dict['new_trades_count']} new trades", 
                ticker
            )

def update_data_preview(ticker, new_data_df, is_incremental=False):
    """Update data preview with new or incremental data"""
    
    tab_widgets = tab_ui_widgets.get(ticker)
    if not tab_widgets:
        return
    
    # Update cleaned data preview
    if "cleaned_data_preview_area" in tab_widgets:
        preview = tab_widgets["cleaned_data_preview_area"]
        preview.config(state=tk.NORMAL)
        
        if is_incremental:
            # For incremental updates, show recent trades
            all_data = app_data["cleaned_dfs"].get(ticker, pd.DataFrame())
            
            if not all_data.empty:
                preview.delete('1.0', tk.END)
                preview.insert(tk.END, f"--- Cleaned Data (Total: {len(all_data)} trades) ---\n\n")
                
                # Show last 50 trades
                recent_trades = all_data.tail(50)
                preview.insert(tk.END, "=== Recent Trades ===\n")
                preview.insert(tk.END, recent_trades.to_string())
                
                # Show summary statistics
                preview.insert(tk.END, "\n\n=== Summary Statistics ===\n")
                summary = generate_data_summary(all_data)
                preview.insert(tk.END, summary)
        else:
            # Full data update
            preview.delete('1.0', tk.END)
            preview.insert(tk.END, f"--- Cleaned Data ({len(new_data_df)} trades) ---\n")
            preview.insert(tk.END, new_data_df.head(50).to_string())
            if len(new_data_df) > 50:
                preview.insert(tk.END, "\n\n... (showing first 50 trades) ...\n\n")
                preview.insert(tk.END, new_data_df.tail(10).to_string())
        
        preview.config(state=tk.DISABLED)
        preview.see('1.0')  # Scroll to top

def generate_data_summary(df):
    """Generate summary statistics for the data"""
    
    if df.empty:
        return "No data available"
    
    summary_lines = []
    
    # Time range
    if 'Time' in df.columns:
        summary_lines.append(f"Time Range: {df['Time'].iloc[0]} - {df['Time'].iloc[-1]}")
    
    # Trade count by type
    if 'Option_Type' in df.columns:
        type_counts = df['Option_Type'].value_counts()
        summary_lines.append(f"Calls: {type_counts.get('Call', 0)}, Puts: {type_counts.get('Put', 0)}")
    
    # Volume statistics
    if 'TradeQuantity' in df.columns:
        total_volume = df['TradeQuantity'].sum()
        avg_size = df['TradeQuantity'].mean()
        max_size = df['TradeQuantity'].max()
        summary_lines.append(f"Total Volume: {total_volume:,}")
        summary_lines.append(f"Avg Trade Size: {avg_size:.1f}")
        summary_lines.append(f"Largest Trade: {max_size:,}")
    
    # Notional value
    if 'NotionalValue' in df.columns:
        total_notional = df['NotionalValue'].sum()
        summary_lines.append(f"Total Notional: ${total_notional:,.0f}")
    
    # Most active strikes
    if 'Strike_Price' in df.columns and 'TradeQuantity' in df.columns:
        strike_volume = df.groupby('Strike_Price')['TradeQuantity'].sum().sort_values(ascending=False)
        top_strikes = strike_volume.head(5)
        summary_lines.append("\nMost Active Strikes:")
        for strike, vol in top_strikes.items():
            summary_lines.append(f"  ${strike}: {vol:,} contracts")
    
    return '\n'.join(summary_lines)

def check_monitoring_data():
    """Enhanced monitoring data check with incremental processing"""
    if not app_data["monitoring_manager"]:
        root.after(1000, check_monitoring_data)
        return
    
    # Process data for each monitored ticker
    for ticker in list(app_data["monitoring_manager"].active_monitors.keys()):
        try:
            # Get new incremental data
            new_data = app_data["monitoring_manager"].get_new_data(ticker)
            
            if new_data and new_data.get('new_trades_count', 0) > 0:
                # Process the new data
                process_monitoring_data(ticker, new_data)
                
                # Log summary
                log_status_to_ui(
                    f"Received {new_data['new_trades_count']} new trades from monitor",
                    ticker
                )
        except Exception as e:
            log_status_to_ui(f"Error processing monitoring data: {e}", 
                           ticker, is_error=True)
    
    # Update monitoring status displays
    for ticker in app_data["monitoring_manager"].active_monitors:
        tab_widgets = tab_ui_widgets.get(ticker)
        if tab_widgets:
            status = app_data["monitoring_manager"].get_monitoring_status(ticker)
            ui_builder.update_monitoring_statistics(ticker, tab_widgets, status)
    
    root.after(1000, check_monitoring_data)

def handle_run_analysis(ticker, auto_triggered=False):
    """Enhanced analysis with support for incremental data"""
    
    df_cleaned = app_data["cleaned_dfs"].get(ticker)
    if df_cleaned is None or df_cleaned.empty:
        if not auto_triggered:
            messagebox.showinfo("Info", f"No processed data for {ticker}")
        return
    
    # Get daily data from monitoring manager if available
    if app_data["monitoring_manager"]:
        daily_df = app_data["monitoring_manager"].get_daily_data(ticker)
        if not daily_df.empty:
            df_cleaned = daily_df
            log_status_to_ui(f"Using complete daily data ({len(daily_df)} trades) for analysis", ticker)
    
    if not auto_triggered:
        log_status_to_ui(f"Starting analysis for {ticker} ({len(df_cleaned)} trades)...", ticker)
    
    # Update UI
    tab_widgets = tab_ui_widgets.get(ticker)
    if tab_widgets and "analyze_data_button" in tab_widgets:
        tab_widgets["analyze_data_button"].config(state=tk.DISABLED, text="Analyzing...")
    
    # Run analysis in thread
    analysis_thread = threading.Thread(
        target=analysis_thread_target,
        args=(df_cleaned.copy(), ticker, log_status_to_ui),
        daemon=True
    )
    analysis_thread.start()

def process_tos_options_data(raw_data: str, ticker: str, log_func: Callable = None, 
                           data_directory: str = "Daily_Data") -> Tuple[pd.DataFrame, dict]:
    """
    Enhanced processing for TOS Options Time & Sales data with incremental updates
    
    Args:
        raw_data: Raw data string from TOS OTS window
        ticker: Stock ticker symbol
        log_func: Logging function
        data_directory: Directory for daily files
    
    Returns:
        Tuple of (processed_dataframe, processing_summary)
    """
    
    log_func = log_func or _default_log
    ticker = ticker.upper()
    
    log_func(f"Processing TOS Options data for {ticker}")
    
    # Setup directories and file paths
    Path(data_directory).mkdir(exist_ok=True)
    Path(f"{data_directory}/metadata").mkdir(exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    daily_file_path = f"{data_directory}/{ticker}_{today}_cleaned.csv"
    metadata_file = f"{data_directory}/metadata/{ticker}_{today}_metadata.json"
    
    # Load existing metadata
    last_trade_info = _load_last_trade_metadata(metadata_file, log_func)
    
    # Parse raw TOS data
    parsed_df = _parse_tos_raw_data(raw_data, log_func)
    
    if parsed_df.empty:
        log_func("No valid trades parsed from raw data", ticker, is_error=True)
        return pd.DataFrame(), {"error": "No valid trades found"}
    
    # Process incrementally
    if last_trade_info['last_trade_time']:
        # Find only new trades
        incremental_df = _find_incremental_tos_trades(parsed_df, last_trade_info, log_func)
        
        if incremental_df.empty:
            log_func("No new trades found since last update", ticker)
            return pd.DataFrame(), {
                "new_trades": 0,
                "message": "No new trades to process"
            }
        
        # Append to existing file
        new_trades_count = _append_tos_trades(incremental_df, daily_file_path, log_func)
        processed_df = incremental_df
    else:
        # First time processing - save all trades
        new_trades_count = _save_all_tos_trades(parsed_df, daily_file_path, log_func)
        processed_df = parsed_df
    
    # Update metadata
    if not processed_df.empty:
        _save_last_trade_metadata(processed_df, metadata_file, ticker, new_trades_count, log_func)
    
    # Generate summary
    summary = {
        "ticker": ticker,
        "new_trades_processed": new_trades_count,
        "total_trades_in_session": len(processed_df),
        "processing_timestamp": datetime.now().isoformat(),
        "daily_file": daily_file_path,
        "data_quality_score": _calculate_tos_data_quality(processed_df)
    }
    
    log_func(f"TOS processing complete: {new_trades_count} new trades processed", ticker)
    
    return processed_df, summary

def _parse_tos_raw_data(raw_data: str, log_func: Callable) -> pd.DataFrame:
    """Parse raw TOS OTS data into structured DataFrame"""
    
    if not raw_data or not raw_data.strip():
        return pd.DataFrame()
    
    lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
    if not lines:
        return pd.DataFrame()
    
    parsed_trades = []
    
    for line_num, line in enumerate(lines, 1):
        try:
            trade_data = _parse_tos_line(line)
            if trade_data:
                parsed_trades.append(trade_data)
        except Exception as e:
            log_func(f"Error parsing line {line_num}: {e}")
            continue
    
    if not parsed_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(parsed_trades)
    
    # Filter out invalid/pre-market trades
    df = _filter_valid_tos_trades(df, log_func)
    
    # Standardize columns
    df = _standardize_tos_columns(df)
    
    # Sort by time
    if 'Time' in df.columns:
        df = df.sort_values('Time').reset_index(drop=True)
    
    return df

def _parse_tos_line(line: str) -> Optional[dict]:
    """Parse single line of TOS OTS data"""
    
    parts = line.split('\t')
    
    if len(parts) < 9:  # Minimum required fields for TOS OTS
        return None
    
    # Extract fields
    time_str = parts[0].strip()
    option_desc = parts[1].strip()
    qty_str = parts[2].strip()
    price_str = parts[3].strip()
    exchange = parts[4].strip()
    market_str = parts[5].strip() if len(parts) > 5 else ""
    delta_str = parts[6].strip() if len(parts) > 6 else ""
    iv_str = parts[7].strip() if len(parts) > 7 else ""
    underlying_str = parts[8].strip() if len(parts) > 8 else ""
    condition = parts[9].strip() if len(parts) > 9 else ""
    
    # Skip pre-market/invalid data
    if qty_str in ["0", ""] or price_str in ["N/A", "0", ""]:
        return None
    
    # Parse numeric values
    try:
        qty = int(qty_str)
        price = float(price_str)
        underlying_price = float(underlying_str) if underlying_str not in ["N/A", ""] else None
    except (ValueError, TypeError):
        return None
    
    # Parse market data (bid x ask)
    bid_price, ask_price = _parse_tos_market_data(market_str)
    
    # Parse Greeks
    delta = _parse_tos_numeric(delta_str)
    iv = _parse_tos_iv(iv_str)
    
    # Parse option description
    exp_date, strike_price, option_type = _parse_tos_option_description(option_desc)
    
    return {
        'Time': time_str,
        'Option_Description_orig': option_desc,
        'Expiration_Date': exp_date,
        'Strike_Price': strike_price,
        'Option_Type': option_type,
        'TradeQuantity': qty,
        'Trade_Price': price,
        'Option_Bid': bid_price,
        'Option_Ask': ask_price,
        'Delta': delta,
        'IV': iv,
        'Underlying_Price': underlying_price,
        'Condition': condition,
        'Exchange': exchange
    }

def _parse_tos_market_data(market_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse TOS market data (e.g., '1.68x1.71') into bid and ask"""
    
    if not market_str or market_str in ["N/A", ""]:
        return None, None
    
    try:
        if 'x' in market_str:
            bid_str, ask_str = market_str.split('x', 1)
            bid = float(bid_str.strip()) if bid_str.strip() not in ["N/A", ""] else None
            ask = float(ask_str.strip()) if ask_str.strip() not in ["N/A", ""] else None
            return bid, ask
    except (ValueError, AttributeError):
        pass
    
    return None, None

def _parse_tos_numeric(value_str: str) -> Optional[float]:
    """Parse numeric string from TOS data"""
    
    if not value_str or value_str in ["N/A", ""]:
        return None
    
    try:
        return float(value_str)
    except (ValueError, TypeError):
        return None

def _parse_tos_iv(iv_str: str) -> Optional[float]:
    """Parse IV string from TOS (e.g., '35.94%')"""
    
    if not iv_str or iv_str in ["N/A", ""]:
        return None
    
    try:
        clean_iv = iv_str.replace('%', '').strip()
        iv_value = float(clean_iv)
        
        # Convert percentage to decimal if needed
        if iv_value > 5:  # Assume >5 means percentage format
            iv_value = iv_value / 100
        
        return iv_value
    except (ValueError, TypeError):
        return None

def _parse_tos_option_description(option_desc: str) -> Tuple[str, float, str]:
    """Parse TOS option description (e.g., '30 MAY 25 200 P')"""
    
    if not option_desc:
        return None, None, None
    
    try:
        # Pattern: "DAY MONTH YEAR STRIKE TYPE"
        parts = option_desc.split()
        
        if len(parts) < 5:
            return None, None, None
        
        day = parts[0]
        month = parts[1]  
        year = parts[2]
        strike = float(parts[3])
        option_type = "Call" if parts[4].upper() == 'C' else "Put"
        
        exp_date = f"{day} {month} {year}"
        
        return exp_date, strike, option_type
        
    except (ValueError, IndexError, AttributeError):
        return None, None, None

def _filter_valid_tos_trades(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """Filter out invalid TOS trades"""
    
    if df.empty:
        return df
    
    initial_count = len(df)
    
    # Remove zero quantity trades
    df = df[df['TradeQuantity'] > 0]
    
    # Remove zero/invalid prices
    df = df[df['Trade_Price'] > 0]
    
    # Remove invalid underlying prices
    df = df[df['Underlying_Price'].notna()]
    df = df[df['Underlying_Price'] > 0]
    
    # Remove invalid option descriptions
    df = df[df['Option_Description_orig'].notna()]
    df = df[df['Option_Description_orig'] != '']
    
    # Remove invalid strikes/expirations
    df = df[df['Strike_Price'].notna()]
    df = df[df['Strike_Price'] > 0]
    
    filtered_count = len(df)
    
    if initial_count != filtered_count:
        log_func(f"Filtered out {initial_count - filtered_count} invalid TOS trades")
    
    return df

def _standardize_tos_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize TOS DataFrame columns"""
    
    if df.empty:
        return df
    
    # Ensure all required columns exist
    for col in TOS_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    
    # Reorder columns
    df = df[TOS_OUTPUT_COLUMNS]
    
    # Add derived columns for compatibility with existing system
    df['StandardOptionSymbol'] = df['Option_Description_orig']  # For compatibility
    
    # Calculate NotionalValue
    df['NotionalValue'] = df['TradeQuantity'] * df['Trade_Price'] * 100
    
    return df

def _find_incremental_tos_trades(new_df: pd.DataFrame, last_trade_info: dict, 
                                log_func: Callable) -> pd.DataFrame:
    """Find TOS trades that occurred after the last processed trade"""
    
    last_trade_time = last_trade_info['last_trade_time']
    last_trade_hash = last_trade_info.get('last_trade_hash')
    
    log_func(f"Looking for trades after {last_trade_time}")
    
    # Convert times for comparison
    new_df['time_for_comparison'] = pd.to_datetime(
        new_df['Time'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    
    last_time = datetime.strptime(last_trade_time, '%H:%M:%S').time()
    
    # Find trades after last time
    after_mask = new_df['time_for_comparison'] > last_time
    
    # Handle same-time trades using hash comparison
    same_time_mask = new_df['time_for_comparison'] == last_time
    same_time_new = pd.DataFrame()
    
    if same_time_mask.any() and last_trade_hash:
        same_time_df = new_df[same_time_mask].copy()
        same_time_df['trade_hash'] = same_time_df.apply(_create_tos_trade_hash, axis=1)
        same_time_new = same_time_df[same_time_df['trade_hash'] != last_trade_hash]
        same_time_new = same_time_new.drop(['trade_hash'], axis=1)
    
    # Combine incremental trades
    after_time_df = new_df[after_mask]
    incremental_df = pd.concat([same_time_new, after_time_df], ignore_index=True)
    
    # Clean up temporary columns
    incremental_df = incremental_df.drop(['time_for_comparison'], axis=1, errors='ignore')
    
    # Sort by time
    incremental_df = incremental_df.sort_values('Time').reset_index(drop=True)
    
    log_func(f"Found {len(incremental_df)} new TOS trades")
    
    return incremental_df

def _create_tos_trade_hash(row: pd.Series) -> str:
    """Create unique hash for TOS trade"""
    hash_components = [
        str(row['Time']),
        str(row['Option_Description_orig']),
        str(row['TradeQuantity']),
        str(row['Trade_Price'])
    ]
    hash_string = "_".join(hash_components)
    return str(hash(hash_string))

def _load_last_trade_metadata(metadata_file: str, log_func: Callable) -> dict:
    """Load last trade metadata"""
    
    default_metadata = {
        'last_trade_time': None,
        'last_trade_hash': None,
        'total_trades_processed': 0,
        'last_update': None
    }
    
    try:
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        log_func(f"Error loading metadata: {e}")
    
    return default_metadata

def _save_last_trade_metadata(df: pd.DataFrame, metadata_file: str, ticker: str, 
                             new_trades_count: int, log_func: Callable):
    """Save last trade metadata"""
    
    try:
        if df.empty:
            return
        
        last_trade = df.iloc[-1]
        last_trade_hash = _create_tos_trade_hash(last_trade)
        
        # Load existing metadata to get total count
        existing_metadata = _load_last_trade_metadata(metadata_file, log_func)
        total_trades = existing_metadata.get('total_trades_processed', 0) + new_trades_count
        
        metadata = {
            'last_trade_time': last_trade['Time'],
            'last_trade_hash': last_trade_hash,
            'total_trades_processed': total_trades,
            'last_update': datetime.now().isoformat(),
            'ticker': ticker
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        log_func(f"Error saving metadata: {e}")

def _save_all_tos_trades(df: pd.DataFrame, daily_file_path: str, log_func: Callable) -> int:
    """Save all TOS trades to new daily file"""
    
    try:
        df.to_csv(daily_file_path, index=False)
        log_func(f"Saved {len(df)} TOS trades to {daily_file_path}")
        return len(df)
    except Exception as e:
        log_func(f"Error saving TOS trades: {e}")
        return 0

def _append_tos_trades(incremental_df: pd.DataFrame, daily_file_path: str, 
                      log_func: Callable) -> int:
    """Append new TOS trades to existing daily file"""
    
    try:
        incremental_df.to_csv(
            daily_file_path,
            mode='a',
            header=not os.path.exists(daily_file_path),
            index=False
        )
        log_func(f"Appended {len(incremental_df)} new TOS trades")
        return len(incremental_df)
    except Exception as e:
        log_func(f"Error appending TOS trades: {e}")
        return 0

def _calculate_tos_data_quality(df: pd.DataFrame) -> float:
    """Calculate data quality score for TOS data"""
    
    if df.empty:
        return 0
    
    total_score = 100
    
    # Check completeness of critical fields
    critical_fields = ['Time', 'TradeQuantity', 'Trade_Price', 'Underlying_Price']
    for field in critical_fields:
        if field in df.columns:
            null_pct = df[field].isnull().sum() / len(df)
            total_score -= (null_pct * 20)  # -20 points per critical field with nulls
    
    # Check option parsing success
    if 'Strike_Price' in df.columns:
        parsed_strikes = df['Strike_Price'].notna().sum() / len(df)
        total_score -= ((1 - parsed_strikes) * 15)  # -15 points for parsing failures
    
    # Check time ordering
    if 'Time' in df.columns:
        time_series = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
        if not time_series.is_monotonic_increasing:
            total_score -= 10  # -10 points for time ordering issues
    
    return max(0, total_score)

# Enhanced versions of existing functions to work with both TOS and generic data

def parse_data_from_string(raw_data: str, log_func: Callable = None) -> Tuple[pd.DataFrame, str]:
    """
    Enhanced parsing that detects TOS OTS data vs generic data
    Returns: (DataFrame, parsing_method_used)
    """
    
    if not raw_data or not raw_data.strip():
        return pd.DataFrame(), "empty_data"
    
    log_func = log_func or _default_log
    
    # Detect data source type
    data_type = detect_data_source_type(raw_data)
    
    if data_type == "tos_ots":
        log_func("Detected TOS Options Time & Sales data format")
        df = _parse_tos_raw_data(raw_data, log_func)
        return df, "tos_ots_format"
    else:
        log_func("Using generic parsing methods")
        # Use existing generic parsing methods
        return _parse_generic_data(raw_data, log_func)

def _parse_generic_data(raw_data: str, log_func: Callable) -> Tuple[pd.DataFrame, str]:
    """Generic parsing methods (existing logic)"""
    
    # Try different parsing methods in order of reliability
    parsing_methods = [
        ("tab_delimited", _parse_tab_delimited),
        ("csv_format", _parse_csv_format),
        ("space_delimited", _parse_space_delimited),
        ("tos_export", _parse_tos_export_format),
        ("custom_format", _parse_custom_format)
    ]
    
    for method_name, parse_func in parsing_methods:
        try:
            df = parse_func(raw_data)
            if not df.empty and len(df.columns) >= 5:
                log_func(f"Successfully parsed data using {method_name} method")
                return df, method_name
        except Exception as e:
            log_func(f"Failed parsing with {method_name}: {e}")
            continue
    
    # Basic fallback parsing
    log_func("All standard parsing methods failed, attempting basic parsing")
    df = _parse_basic_format(raw_data)
    if not df.empty:
        return df, "basic_fallback"
    
    return pd.DataFrame(), "failed_all_methods"

def process_raw_options_data(df: pd.DataFrame, ticker: str, log_func: Callable = None) -> pd.DataFrame:
    """
    Enhanced processing that handles both TOS and generic data formats
    """
    log_func = log_func or _default_log
    
    if df.empty:
        log_func("Empty DataFrame provided for processing")
        return pd.DataFrame()
    
    log_func(f"Processing {len(df)} rows of raw data for {ticker}")
    
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Check if this looks like already-processed TOS data
    if all(col in df_clean.columns for col in ['Option_Description_orig', 'Expiration_Date', 'Strike_Price']):
        log_func("Data appears to be pre-processed TOS format")
        # Just standardize and validate
        df_clean = _standardize_tos_columns(df_clean)
        df_clean = _validate_and_filter_data(df_clean, log_func)
    else:
        # Use existing generic processing pipeline
        log_func("Using generic processing pipeline")
        df_clean = _process_generic_options_data(df_clean, ticker, log_func)
    
    # Sort by time if available
    time_cols = ['Time_dt', 'Time']
    for time_col in time_cols:
        if time_col in df_clean.columns:
            try:
                df_clean = df_clean.sort_values(time_col).reset_index(drop=True)
                break
            except:
                continue
    
    log_func(f"Processing complete: {len(df_clean)} valid rows remaining")
    
    return df_clean

def _process_generic_options_data(df: pd.DataFrame, ticker: str, log_func: Callable) -> pd.DataFrame:
    """Process generic (non-TOS) options data using existing pipeline"""
    
    # Step 1: Standardize column names
    df = _standardize_column_names(df, log_func)
    
    # Step 2: Parse and clean time data
    df = _parse_time_data(df, log_func)
    
    # Step 3: Parse option symbols
    df = _parse_option_symbols(df, ticker, log_func)
    
    # Step 4: Clean numeric data
    df = _clean_numeric_data(df, log_func)
    
    # Step 5: Calculate derived fields
    df = _calculate_derived_fields(df, log_func)
    
    # Step 6: Validate and filter data
    df = _validate_and_filter_data(df, log_func)
    
    return df

# Main integration function for your existing system

def process_options_data_enhanced(raw_data: str, ticker: str, log_func: Callable = None,
                                data_directory: str = "Daily_Data") -> Tuple[pd.DataFrame, dict]:
    """
    Main enhanced function that replaces the existing process_raw_options_data workflow
    Handles both TOS OTS data and generic formats with incremental processing
    """
    
    log_func = log_func or _default_log
    
    # Detect data type and route accordingly
    data_type = detect_data_source_type(raw_data)
    
    if data_type == "tos_ots":
        # Use specialized TOS processing with incremental updates
        return process_tos_options_data(raw_data, ticker, log_func, data_directory)
    else:
        # Use existing generic processing
        df, parse_method = parse_data_from_string(raw_data, log_func)
        if df.empty:
            return pd.DataFrame(), {"error": "Failed to parse data", "parse_method": parse_method}
        
        cleaned_df = process_raw_options_data(df, ticker, log_func)
        
        summary = {
            "ticker": ticker,
            "total_trades": len(cleaned_df),
            "parse_method": parse_method,
            "data_type": "generic",
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return cleaned_df, summary

# Keep all existing functions for backward compatibility
# [All your existing functions remain unchanged below this point]

def _standardize_column_names(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """Standardize column names to expected format"""
    
    # Column mapping for common variations
    column_mapping = {
        # Time columns
        'time': 'Time',
        'timestamp': 'Time',
        'trade_time': 'Time',
        'datetime': 'Time',
        
        # Symbol columns
        'symbol': 'StandardOptionSymbol',
        'option_symbol': 'StandardOptionSymbol',
        'contract': 'StandardOptionSymbol',
        'option': 'StandardOptionSymbol',
        
        # Quantity columns
        'quantity': 'TradeQuantity',
        'qty': 'TradeQuantity',
        'size': 'TradeQuantity',
        'trade_quantity': 'TradeQuantity',
        'volume': 'TradeQuantity',
        
        # Price columns
        'price': 'Trade_Price',
        'trade_price': 'Trade_Price',
        'last': 'Trade_Price',
        'last_price': 'Trade_Price',
        
        # Aggressor columns
        'side': 'Aggressor',
        'buyer_seller': 'Aggressor',
        'direction': 'Aggressor',
        
        # Exchange columns
        'exch': 'Exchange',
        'exchange': 'Exchange',
        'venue': 'Exchange',
        
        # Greeks and other option data
        'implied_volatility': 'IV',
        'iv': 'IV',
        'impl_vol': 'IV',
        'delta': 'Delta',
        'gamma': 'Gamma',
        'theta': 'Theta',
        'vega': 'Vega',
        'rho': 'Rho',
        
        # Bid/Ask
        'bid': 'Option_Bid',
        'ask': 'Option_Ask',
        'bid_price': 'Option_Bid',
        'ask_price': 'Option_Ask',
        
        # Underlying
        'underlying': 'Underlying_Price',
        'underlying_price': 'Underlying_Price',
        'stock_price': 'Underlying_Price',
        
        # Other
        'condition': 'Condition',
        'open_interest': 'OpenInterest',
        'oi': 'OpenInterest'
    }
    
    # Create new column names
    new_columns = []
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in column_mapping:
            new_columns.append(column_mapping[col_lower])
        else:
            new_columns.append(col)
    
    df.columns = new_columns
    
    log_func(f"Standardized column names: {list(df.columns)}")
    
    return df

def _parse_time_data(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """Parse time data into datetime objects"""
    
    if 'Time' not in df.columns:
        log_func("No time column found, using current time")
        df['Time_dt'] = datetime.now()
        return df
    
    # Try different time parsing methods
    time_formats = [
        '%H:%M:%S',
        '%H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S',
        '%Y%m%d %H:%M:%S',
        '%H:%M',
        '%I:%M:%S %p',
        '%I:%M %p'
    ]
    
    df['Time_dt'] = None
    parsed_count = 0
    
    for idx, time_str in enumerate(df['Time']):
        if pd.isna(time_str) or not str(time_str).strip():
            continue
            
        time_str = str(time_str).strip()
        
        # Try each format
        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, fmt)
                
                # If only time (no date), add today's date
                if fmt in ['%H:%M:%S', '%H:%M:%S.%f', '%H:%M', '%I:%M:%S %p', '%I:%M %p']:
                    today = datetime.now().date()
                    parsed_time = datetime.combine(today, parsed_time.time())
                
                df.at[idx, 'Time_dt'] = parsed_time
                parsed_count += 1
                break
                
            except ValueError:
                continue
    
    # If no standard format worked, try pandas
    if parsed_count == 0:
        try:
            df['Time_dt'] = pd.to_datetime(df['Time'], errors='coerce')
            parsed_count = df['Time_dt'].notna().sum()
        except:
            pass
    
    log_func(f"Successfully parsed {parsed_count}/{len(df)} time entries")
    
    return df

def _parse_option_symbols(df: pd.DataFrame, ticker: str, log_func: Callable) -> pd.DataFrame:
    """Parse option symbols and extract strike, expiration, type"""
    
    if 'StandardOptionSymbol' not in df.columns:
        log_func("No option symbol column found")
        return df
    
    # Initialize new columns
    df['Strike_Price_calc'] = None
    df['Expiration_Date_calc'] = None
    df['Option_Type_calc'] = None
    df['DTE_calc'] = None
    
    parsed_count = 0
    
    for idx, symbol in enumerate(df['StandardOptionSymbol']):
        if pd.isna(symbol) or not str(symbol).strip():
            continue
            
        symbol_str = str(symbol).strip()
        
        # Try to parse standard option symbol format
        parsed_data = _parse_single_option_symbol(symbol_str, ticker)
        
        if parsed_data:
            df.at[idx, 'Strike_Price_calc'] = parsed_data['strike']
            df.at[idx, 'Expiration_Date_calc'] = parsed_data['expiration']
            df.at[idx, 'Option_Type_calc'] = parsed_data['option_type']
            df.at[idx, 'DTE_calc'] = parsed_data['dte']
            parsed_count += 1
    
    log_func(f"Successfully parsed {parsed_count}/{len(df)} option symbols")
    
    return df

def _parse_single_option_symbol(symbol: str, ticker: str) -> Optional[dict]:
    """Parse a single option symbol"""
    
    # Common option symbol formats:
    # 1. SPY240315C00500000 (new OCC format)
    # 2. SPY   240315C00500000 (with spaces)
    # 3. .SPY240315C00500000 (old format)
    # 4. SPY 03/15/24 C500 (human readable)
    
    symbol = symbol.strip().upper()
    
    # Remove leading dots
    symbol = symbol.lstrip('.')
    
    # Try OCC 21-character format
    occ_match = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})', symbol.replace(' ', ''))
    if occ_match:
        root, date_str, call_put, strike_str = occ_match.groups()
        
        try:
            # Parse date (YYMMDD)
            exp_date = datetime.strptime(date_str, '%y%m%d').date()
            
            # Parse strike (multiply by 1000 and divide by 100000 for price)
            strike_price = int(strike_str) / 1000
            
            # Calculate DTE
            dte = (exp_date - datetime.now().date()).days
            
            return {
                'strike': strike_price,
                'expiration': exp_date,
                'option_type': 'Call' if call_put == 'C' else 'Put',
                'dte': dte
            }
        except:
            pass
    
    # Try alternative formats
    patterns = [
        r'([A-Z]+)\s*(\d{2})(\d{2})(\d{2})\s*([CP])\s*(\d+)',  # SPY240315C500
        r'([A-Z]+)\s*(\d{1,2})/(\d{1,2})/(\d{2,4})\s*([CP])\s*(\d+)',  # SPY 3/15/24 C500
        r'([A-Z]+)\s*(\d{6})\s*([CP])\s*(\d+)',  # SPY 240315 C 500
    ]
    
    for pattern in patterns:
        match = re.match(pattern, symbol)
        if match:
            try:
                groups = match.groups()
                
                if len(groups) == 6 and '/' in symbol:
                    # Date format MM/DD/YY
                    root, month, day, year, call_put, strike = groups
                    year = int(year)
                    if year < 50:
                        year += 2000
                    elif year < 100:
                        year += 1900
                    exp_date = datetime(year, int(month), int(day)).date()
                else:
                    # Other formats
                    root, date_part1, date_part2, date_part3, call_put, strike = groups
                    
                    if len(date_part1) == 2:  # YYMMDD format
                        date_str = date_part1 + date_part2 + date_part3
                        exp_date = datetime.strptime(date_str, '%y%m%d').date()
                    else:
                        continue
                
                strike_price = float(strike)
                dte = (exp_date - datetime.now().date()).days
                
                return {
                    'strike': strike_price,
                    'expiration': exp_date,
                    'option_type': 'Call' if call_put == 'C' else 'Put',
                    'dte': dte
                }
            except:
                continue
    
    return None

def _clean_numeric_data(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """Clean and convert numeric columns"""
    
    numeric_columns = [
        'TradeQuantity', 'Trade_Price', 'IV', 'Delta', 'Gamma', 'Theta',
        'Vega', 'Rho', 'Option_Bid', 'Option_Ask', 'Underlying_Price',
        'Volume', 'OpenInterest', 'Strike_Price_calc'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = _convert_to_numeric(df[col], col, log_func)
    
    return df

def _convert_to_numeric(series: pd.Series, col_name: str, log_func: Callable) -> pd.Series:
    """Convert series to numeric, handling various formats"""
    
    def clean_numeric_string(val):
        if pd.isna(val):
            return np.nan
        
        val_str = str(val).strip()
        
        # Remove common non-numeric characters
        val_str = re.sub(r'[$,%]', '', val_str)
        
        # Handle percentage
        if val_str.endswith('%'):
            try:
                return float(val_str[:-1]) / 100
            except:
                return np.nan
        
        # Try direct conversion
        try:
            return float(val_str)
        except:
            return np.nan
    
    cleaned_series = series.apply(clean_numeric_string)
    
    # Log conversion stats
    valid_count = cleaned_series.notna().sum()
    total_count = len(cleaned_series)
    
    if total_count > 0:
        log_func(f"Converted {col_name}: {valid_count}/{total_count} valid values")
    
    return cleaned_series

def _calculate_derived_fields(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """Calculate derived fields"""
    
    # Calculate NotionalValue if not present
    if 'NotionalValue' not in df.columns:
        if 'TradeQuantity' in df.columns and 'Trade_Price' in df.columns:
            df['NotionalValue'] = df['TradeQuantity'] * df['Trade_Price'] * 100
            log_func("Calculated NotionalValue from quantity and price")
    
    # Calculate moneyness if we have strike and underlying price
    if all(col in df.columns for col in ['Strike_Price_calc', 'Underlying_Price']):
        df['Moneyness'] = df['Strike_Price_calc'] / df['Underlying_Price']
        df['ITM_OTM'] = df.apply(_determine_itm_otm, axis=1)
        log_func("Calculated moneyness and ITM/OTM status")
    
    # Calculate time value if we have intrinsic value components
    if all(col in df.columns for col in ['Trade_Price', 'Strike_Price_calc', 'Underlying_Price', 'Option_Type_calc']):
        df['Intrinsic_Value'] = df.apply(_calculate_intrinsic_value, axis=1)
        df['Time_Value'] = df['Trade_Price'] - df['Intrinsic_Value']
        log_func("Calculated intrinsic and time value")
    
    # Calculate volume metrics
    if 'TradeQuantity' in df.columns:
        df['Cumulative_Volume'] = df['TradeQuantity'].cumsum()
        df['Volume_Rank'] = df['TradeQuantity'].rank(pct=True) * 100
        log_func("Calculated volume metrics")
    
    return df

def _determine_itm_otm(row) -> str:
    """Determine if option is ITM, OTM, or ATM"""
    try:
        strike = row['Strike_Price_calc']
        underlying = row['Underlying_Price']
        option_type = row['Option_Type_calc']
        
        if pd.isna(strike) or pd.isna(underlying) or pd.isna(option_type):
            return 'Unknown'
        
        diff_pct = abs(strike - underlying) / underlying
        
        if diff_pct < 0.02:  # Within 2%
            return 'ATM'
        elif option_type == 'Call':
            return 'ITM' if underlying > strike else 'OTM'
        else:  # Put
            return 'ITM' if underlying < strike else 'OTM'
    except:
        return 'Unknown'

def _calculate_intrinsic_value(row) -> float:
    """Calculate intrinsic value of option"""
    try:
        strike = row['Strike_Price_calc']
        underlying = row['Underlying_Price']
        option_type = row['Option_Type_calc']
        
        if pd.isna(strike) or pd.isna(underlying) or pd.isna(option_type):
            return 0
        
        if option_type == 'Call':
            return max(0, underlying - strike)
        else:  # Put
            return max(0, strike - underlying)
    except:
        return 0

def _validate_and_filter_data(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """Validate data and filter out invalid rows"""
    
    initial_count = len(df)
    
    # Remove rows with missing critical data
    critical_columns = ['StandardOptionSymbol', 'TradeQuantity', 'Trade_Price']
    available_critical = [col for col in critical_columns if col in df.columns]
    
    if available_critical:
        df = df.dropna(subset=available_critical)
        log_func(f"Removed rows missing critical data: {initial_count - len(df)} rows")
    
    # Filter out invalid trades
    if 'TradeQuantity' in df.columns:
        df = df[df['TradeQuantity'] > 0]
        log_func(f"Filtered positive quantity: {len(df)} rows remaining")
    
    if 'Trade_Price' in df.columns:
        df = df[df['Trade_Price'] > 0]
        log_func(f"Filtered positive price: {len(df)} rows remaining")
    
    # Remove obvious outliers
    if 'NotionalValue' in df.columns:
        # Remove trades > $50M (likely data errors)
        df = df[df['NotionalValue'] <= 50000000]
        log_func(f"Filtered extreme notional values: {len(df)} rows remaining")
    
    if 'IV' in df.columns:
        # Remove IV > 500% or < 0%
        df = df[(df['IV'] >= 0) & (df['IV'] <= 5)]
        log_func(f"Filtered reasonable IV range: {len(df)} rows remaining")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    final_count = len(df)
    log_func(f"Data validation complete: {final_count}/{initial_count} rows remaining")
    
    return df

# Existing parsing methods (keeping all your original functions)

def _parse_tab_delimited(raw_data: str) -> pd.DataFrame:
    """Parse tab-delimited data"""
    lines = raw_data.strip().split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    if len(cleaned_lines) < 2:
        return pd.DataFrame()
    
    data_str = '\n'.join(cleaned_lines)
    df = pd.read_csv(StringIO(data_str), sep='\t', dtype=str)
    return df

def _parse_csv_format(raw_data: str) -> pd.DataFrame:
    """Parse CSV formatted data"""
    lines = raw_data.strip().split('\n')
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    if len(cleaned_lines) < 2:
        return pd.DataFrame()
    
    data_str = '\n'.join(cleaned_lines)
    df = pd.read_csv(StringIO(data_str), dtype=str)
    return df

def _parse_space_delimited(raw_data: str) -> pd.DataFrame:
    """Parse space-delimited data"""
    lines = raw_data.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        if line.strip():
            # Replace multiple spaces with single tabs
            cleaned_line = re.sub(r'\s+', '\t', line.strip())
            cleaned_lines.append(cleaned_line)
    
    if len(cleaned_lines) < 2:
        return pd.DataFrame()
    
    data_str = '\n'.join(cleaned_lines)
    df = pd.read_csv(StringIO(data_str), sep='\t', dtype=str)
    return df

def _parse_tos_export_format(raw_data: str) -> pd.DataFrame:
    """Parse ThinkOrSwim export format"""
    lines = raw_data.strip().split('\n')
    
    # Find header line
    header_idx = -1
    for i, line in enumerate(lines):
        if any(col.lower() in line.lower() for col in ['time', 'symbol', 'quantity', 'price']):
            header_idx = i
            break
    
    if header_idx == -1:
        return pd.DataFrame()
    
    # Extract header and data
    header_line = lines[header_idx]
    data_lines = lines[header_idx + 1:]
    
    # Clean header
    header = [col.strip() for col in re.split(r'[\t,]', header_line) if col.strip()]
    
    # Parse data lines
    data_rows = []
    for line in data_lines:
        if line.strip():
            row = [val.strip() for val in re.split(r'[\t,]', line)]
            if len(row) >= len(header) // 2:  # At least half the columns
                data_rows.append(row)
    
    if not data_rows:
        return pd.DataFrame()
    
    # Create DataFrame
    max_cols = max(len(row) for row in data_rows)
    header = header[:max_cols]
    
    # Pad rows to same length
    padded_rows = []
    for row in data_rows:
        padded_row = row + [''] * (max_cols - len(row))
        padded_rows.append(padded_row[:max_cols])
    
    df = pd.DataFrame(padded_rows, columns=header[:len(padded_rows[0])] if padded_rows else [])
    return df

def _parse_custom_format(raw_data: str) -> pd.DataFrame:
    """Parse custom/unknown format using pattern detection"""
    lines = raw_data.strip().split('\n')
    
    if len(lines) < 2:
        return pd.DataFrame()
    
    # Detect separator
    separators = ['\t', ',', '|', ';', ' ']
    best_separator = '\t'
    max_consistency = 0
    
    sample_lines = lines[:min(10, len(lines))]
    
    for sep in separators:
        column_counts = [len(line.split(sep)) for line in sample_lines]
        if len(set(column_counts)) == 1 and column_counts[0] > 1:
            consistency = column_counts[0]
            if consistency > max_consistency:
                max_consistency = consistency
                best_separator = sep
    
    if max_consistency < 2:
        return pd.DataFrame()
    
    # Parse with detected separator
    data_str = '\n'.join(lines)
    
    try:
        df = pd.read_csv(StringIO(data_str), sep=best_separator, dtype=str)
        return df
    except:
        return pd.DataFrame()

def _parse_basic_format(raw_data: str) -> pd.DataFrame:
    """Basic fallback parsing for any readable format"""
    lines = raw_data.strip().split('\n')
    
    if len(lines) < 2:
        return pd.DataFrame()
    
    data_rows = []
    for line in lines:
        if line.strip():
            # Split by any whitespace or common delimiters
            row = re.split(r'[\s,\t|;]+', line.strip())
            # Filter out empty strings
            row = [item for item in row if item]
            if len(row) >= 3:  # Minimum viable row
                data_rows.append(row)
    
    if not data_rows:
        return pd.DataFrame()
    
    # Create basic column names
    max_cols = max(len(row) for row in data_rows)
    columns = [f'Col_{i}' for i in range(max_cols)]
    
    # Pad all rows to same length
    padded_rows = []
    for row in data_rows:
        padded_row = row + [''] * (max_cols - len(row))
        padded_rows.append(padded_row[:max_cols])
    
    df = pd.DataFrame(padded_rows, columns=columns)
    return df

# Keep all your existing utility functions unchanged...
#[Rest of your existing functions like validate_data_quality, export_processed_data, etc.]

# New utility functions for TOS integration

def get_tos_daily_data(ticker: str, data_directory: str = "Daily_Data") -> pd.DataFrame:
    """Load complete daily TOS data for analysis"""
    
    try:
        today = datetime.now().strftime("%Y%m%d")
        daily_file_path = f"{data_directory}/{ticker.upper()}_{today}_cleaned.csv"
        
        if os.path.exists(daily_file_path):
            return pd.read_csv(daily_file_path)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading TOS daily data: {e}")
        return pd.DataFrame()

def get_tos_data_summary(ticker: str, data_directory: str = "Daily_Data") -> dict:
    """Get summary of TOS data for ticker"""
    
    try:
        today = datetime.now().strftime("%Y%m%d")
        daily_file_path = f"{data_directory}/{ticker.upper()}_{today}_cleaned.csv"
        metadata_file = f"{data_directory}/metadata/{ticker.upper()}_{today}_metadata.json"
        
        summary = {
            'ticker': ticker.upper(),
            'date': today,
            'file_exists': os.path.exists(daily_file_path),
            'total_trades': 0,
            'last_trade_time': None,
            'file_size_mb': 0,
            'last_update': None
        }
        
        if os.path.exists(daily_file_path):
            df = pd.read_csv(daily_file_path)
            summary['total_trades'] = len(df)
            
            if not df.empty:
                summary['last_trade_time'] = df.iloc[-1]['Time']
            
            summary['file_size_mb'] = os.path.getsize(daily_file_path) / (1024 * 1024)
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                summary.update(metadata)
        
        return summary
        
    except Exception as e:
        print(f"Error getting TOS data summary: {e}")
        return {'error': str(e)}

# Export functions remain the same as your existing code...
def validate_data_quality(df: pd.DataFrame) -> dict:
    """Validate data quality and return quality metrics"""
    
    quality_metrics = {
        'total_rows': len(df),
        'completeness': {},
        'consistency': {},
        'validity': {},
        'outliers': {},
        'overall_score': 0
    }
    
    if df.empty:
        return quality_metrics
    
    # Completeness check
    for col in df.columns:
        non_null_pct = (1 - df[col].isnull().sum() / len(df)) * 100
        quality_metrics['completeness'][col] = non_null_pct
    
    # Consistency checks
    if 'Time_dt' in df.columns:
        time_consistency = _check_time_consistency(df['Time_dt'])
        quality_metrics['consistency']['time_order'] = time_consistency
    
    if 'TradeQuantity' in df.columns:
        qty_consistency = (df['TradeQuantity'] > 0).sum() / len(df) * 100
        quality_metrics['consistency']['positive_quantity'] = qty_consistency
    
    # Validity checks
    if 'IV' in df.columns:
        valid_iv = ((df['IV'] >= 0) & (df['IV'] <= 5)).sum() / len(df) * 100
        quality_metrics['validity']['iv_range'] = valid_iv
    
    if 'Delta' in df.columns:
        valid_delta = ((df['Delta'] >= -1) & (df['Delta'] <= 1)).sum() / len(df) * 100
        quality_metrics['validity']['delta_range'] = valid_delta
    
    # Outlier detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(df[col].dropna()) > 0:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            outlier_pct = outlier_count / len(df) * 100
            quality_metrics['outliers'][col] = outlier_pct
    
    # Calculate overall score
    completeness_avg = np.mean(list(quality_metrics['completeness'].values()))
    consistency_avg = np.mean(list(quality_metrics['consistency'].values())) if quality_metrics['consistency'] else 100
    validity_avg = np.mean(list(quality_metrics['validity'].values())) if quality_metrics['validity'] else 100
    
    quality_metrics['overall_score'] = (completeness_avg + consistency_avg + validity_avg) / 3
    
    return quality_metrics

def _check_time_consistency(time_series: pd.Series) -> float:
    """Check if timestamps are in order"""
    if len(time_series) <= 1:
        return 100
    
    # Remove null values
    valid_times = time_series.dropna()
    if len(valid_times) <= 1:
        return 100
    
    # Check if sorted
    is_sorted = valid_times.is_monotonic_increasing
    
    if is_sorted:
        return 100
    else:
        # Calculate percentage of correctly ordered pairs
        correct_pairs = 0
        total_pairs = len(valid_times) - 1
        
        for i in range(total_pairs):
            if valid_times.iloc[i] <= valid_times.iloc[i + 1]:
                correct_pairs += 1
        
        return (correct_pairs / total_pairs) * 100 if total_pairs > 0 else 100
    
