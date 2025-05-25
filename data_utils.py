# data_utils.py
# Helper functions for data conversion and the core logic for processing raw options data.

import pandas as pd
import re
import numpy as np
from datetime import datetime
# config constants will be imported by main_app.py and passed if needed,
# or this module can import config directly if it's in the same path.

# --- Helper functions for safe data conversion --- (Existing functions are good)
def safe_float_conversion(value_str, is_percentage=False):
    if pd.isna(value_str) or isinstance(value_str, str) and value_str.strip().upper() == "N/A":
        return np.nan
    try:
        cleaned_value_str = str(value_str).strip()
        if is_percentage:
            if cleaned_value_str.endswith('%'):
                return float(cleaned_value_str.rstrip('%')) / 100.0
            else:
                val = float(cleaned_value_str)
                return val / 100.0 if abs(val) > 1.5 else val 
        return float(cleaned_value_str)
    except (ValueError, TypeError):
        return np.nan

def safe_int_conversion(value_str):
    if pd.isna(value_str) or isinstance(value_str, str) and value_str.strip().upper() == "N/A":
        return np.nan
    try:
        return int(str(value_str).replace(',', '').strip())
    except (ValueError, TypeError):
        return np.nan

def parse_tos_option_symbol_for_data_utils(symbol_str):
    """
    Parses a Thinkorswim RTD option symbol string into its components.
    Returns: (ticker, exp_date, strike, opt_type) or None
    """
    if not symbol_str or not isinstance(symbol_str, str) or not symbol_str.startswith('.'):
        return None 
    match = re.match(r"^\.(?P<ticker>[A-Z0-9./]+)(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})(?P<type>[CP])(?P<strike>\d+\.?\d*)$", symbol_str.upper())
    if match:
        data = match.groupdict()
        try:
            exp_date = pd.to_datetime(f"{int(data['year'])+2000}-{data['month']}-{data['day']}")
            strike = float(data["strike"]) 
            opt_type = data["type"]
            ticker = data["ticker"]
            return ticker, exp_date, strike, opt_type
        except (ValueError, TypeError):
            return None
    return None

def parse_option_description(desc_str, active_ticker_default='VST'): # Added active_ticker_default
    """
    Parses an option description string to extract underlying, expiration date, strike price, and option type.
    Returns: (underlying, expiration_date, strike_price, option_type)
    """
    if pd.isna(desc_str): return active_ticker_default, pd.NaT, np.nan, None # Return default ticker
    desc_str_original = str(desc_str).strip()
    desc_str_upper = desc_str_original.upper() 

    option_type = None
    strike_price = np.nan
    expiration_date = pd.NaT
    underlying = active_ticker_default # Default

    # Try human-readable format first: "DD MON YY STRIKE TYPE" or "MON DD YY STRIKE TYPE"
    match_human_readable = re.search(r'([\d\.]+)\s*([CP])$', desc_str_upper) 
    if match_human_readable:
        strike_price_str = match_human_readable.group(1)
        parsed_option_type = match_human_readable.group(2)
        parsed_strike_price = safe_float_conversion(strike_price_str)
        
        date_str_part = desc_str_upper[:match_human_readable.start()].strip()
        
        # Attempt to extract an explicit underlying from the beginning of date_str_part
        # e.g., "SPY 20 JUN 25" or "IWM20JUN25"
        underlying_match_human = re.match(r"^([A-Z0-9\.]{1,6})\s*", date_str_part) # Up to 6 chars for underlying
        if underlying_match_human:
            potential_underlying = underlying_match_human.group(1)
            # Check if it's a plausible ticker (contains letters or known structure)
            if any(c.isalpha() for c in potential_underlying) or '.' in potential_underlying:
                underlying = potential_underlying
                date_str_part = date_str_part[len(potential_underlying):].strip() # Remove underlying part

        date_formats_to_try = ['%d %b %y', '%d%b%y', '%b %d %y', '%b%d%y']
        parsed_expiration_date = pd.NaT
        for fmt in date_formats_to_try:
            try:
                parsed_expiration_date = pd.to_datetime(date_str_part, format=fmt)
                break 
            except ValueError:
                continue
        
        if pd.isna(parsed_expiration_date):
            try: 
                parsed_expiration_date = pd.to_datetime(date_str_part)
            except (ValueError, TypeError, pd.errors.ParserError):
                pass
        
        if pd.notna(parsed_expiration_date) and pd.notna(parsed_strike_price) and parsed_option_type:
            return underlying, parsed_expiration_date, parsed_strike_price, parsed_option_type

    # Fallback: Try parsing as a ToS-style symbol
    parsed_tos_style = parse_tos_option_symbol_for_data_utils(desc_str_original)
    if parsed_tos_style:
        ticker_tos, exp_date_tos, strike_tos, type_tos = parsed_tos_style
        if pd.notna(exp_date_tos) and pd.notna(strike_tos) and type_tos:
            return ticker_tos, exp_date_tos, strike_tos, type_tos # ticker_tos is the new underlying
            
    return underlying, pd.NaT, np.nan, None


def parse_market_at_trade(market_str):
    # Existing function is good
    if pd.isna(market_str) or not isinstance(market_str, str) or 'x' not in market_str.lower() :
        if isinstance(market_str, str) and market_str.strip().upper() == "N/AXN/A":
            return np.nan, np.nan
        return np.nan, np.nan 

    parts = market_str.lower().split('x') 
    if len(parts) != 2:
        return np.nan, np.nan
    try:
        bid_str = parts[0].strip()
        ask_str = parts[1].strip()
        bid = safe_float_conversion(bid_str) if bid_str and bid_str != '.' and bid_str != 'n/a' else np.nan
        ask = safe_float_conversion(ask_str) if ask_str and ask_str != '.' and ask_str != 'n/a' else np.nan
        return bid, ask
    except (ValueError, IndexError, TypeError):
        return np.nan, np.nan

# --- Core Data Processing Logic ---
def process_raw_data_string(raw_data_string, active_ticker, config_params, log_status_func=print): # Added active_ticker
    lines = raw_data_string.strip().split('\n')
    list_of_trade_records = []
    skipped_na_price_lines = 0
    parsed_lines = 0
    malformed_lines = 0
    
    # Access config parameters (passed as dict or directly from imported config)
    price_field_index = config_params.get('PRICE_FIELD_INDEX', 3)
    expected_field_count_full = config_params.get('EXPECTED_FIELD_COUNT_FULL', 10)
    expected_field_count_no_condition = config_params.get('EXPECTED_FIELD_COUNT_NO_CONDITION', 9)


    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue

        fields = line.split('\t')

        if len(fields) > price_field_index and fields[price_field_index].strip().upper() == "N/A":
            skipped_na_price_lines += 1
            continue

        record = None
        if len(fields) == expected_field_count_no_condition:
            record = {
                'Time_str': fields[0].strip(), 'Option_Description_str': fields[1].strip(),
                'TradeQuantity_str': fields[2].strip(), 'Trade_Price_str': fields[3].strip(),
                'Exchange_str': fields[4].strip(), 'Market_at_trade_str': fields[5].strip(),
                'Delta_str': fields[6].strip(), 'IV_str': fields[7].strip(),
                'Underlying_Price_str': fields[8].strip(), 'Condition_str': None 
            }
        elif len(fields) >= expected_field_count_full: 
            record = {
                'Time_str': fields[0].strip(), 'Option_Description_str': fields[1].strip(),
                'TradeQuantity_str': fields[2].strip(), 'Trade_Price_str': fields[3].strip(),
                'Exchange_str': fields[4].strip(), 'Market_at_trade_str': fields[5].strip(),
                'Delta_str': fields[6].strip(), 'IV_str': fields[7].strip(),
                'Underlying_Price_str': fields[8].strip(),
                'Condition_str': fields[9].strip() if len(fields) > 9 and fields[9].strip() else None
            }
        
        if record:
            list_of_trade_records.append(record)
            parsed_lines += 1
        else:
            malformed_lines +=1
            
    if not list_of_trade_records:
        log_status_func("No trade records created from raw string.")
        return pd.DataFrame(), parsed_lines, skipped_na_price_lines, malformed_lines # Return empty DF
    
    df = pd.DataFrame(list_of_trade_records)
    
    # Time parsing
    df['Time'] = pd.to_datetime(df['Time_str'], format='%m/%d/%y %H:%M:%S.%f', errors='coerce')
    if df['Time'].isnull().all() and not df.empty: 
        df['Time'] = pd.to_datetime(df['Time_str'], format='%m/%d/%y %H:%M:%S', errors='coerce')

    # Option Description Parsing (modified to include active_ticker)
    parsed_desc_series = df['Option_Description_str'].apply(lambda x: parse_option_description(x, active_ticker_default=active_ticker))
    parsed_desc_df = pd.DataFrame(parsed_desc_series.tolist(), index=df.index, columns=['ParsedUnderlying', 'Expiration_Date', 'Strike_Price', 'Option_Type'])
    df = pd.concat([df, parsed_desc_df], axis=1)

    # Market at Trade Parsing
    parsed_market_cols = df['Market_at_trade_str'].apply(lambda x: pd.Series(parse_market_at_trade(x), index=['Option_Bid', 'Option_Ask']))
    df = pd.concat([df, parsed_market_cols], axis=1)
    
    # Convert other string columns to numeric types (essential before calculations)
    numeric_conversion_map = {
        'TradeQuantity': safe_int_conversion,
        'Trade_Price': safe_float_conversion,
        'Option_Bid': safe_float_conversion,
        'Option_Ask': safe_float_conversion,
        'Delta': safe_float_conversion, # Delta usually not a percentage input here
        'IV': lambda x: safe_float_conversion(x, is_percentage=True),
        'Underlying_Price': safe_float_conversion
    }
    string_col_map = { # Map original string columns to their numeric counterparts
        'TradeQuantity_str': 'TradeQuantity', 'Trade_Price_str': 'Trade_Price',
        'Delta_str': 'Delta', 'IV_str': 'IV', 'Underlying_Price_str': 'Underlying_Price'
    }
    for str_col, num_col_name in string_col_map.items():
        if str_col in df.columns:
            df[num_col_name] = df[str_col].apply(numeric_conversion_map[num_col_name])
    # For Option_Bid, Option_Ask, they are directly created with numeric types from parse_market_at_trade
    # For Strike_Price, it's created numerically by parse_option_description

    df.rename(columns={
        'Option_Description_str': 'Option_Description_orig', 
        'Condition_str': 'Condition',
        'Exchange_str': 'Exchange'
    }, inplace=True)
    
    # --- Start of New Feature Additions ---

    # Ensure critical numeric columns exist for subsequent calculations, fill with NaN if not
    cols_for_calc = ['TradeQuantity', 'Trade_Price', 'Strike_Price', 'Underlying_Price', 'Delta', 'IV', 'Option_Bid', 'Option_Ask']
    for col in cols_for_calc:
        if col not in df.columns:
            df[col] = np.nan # Create column if it does not exist
        # Ensure they are numeric after all parsing and creation
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # 1. StandardOptionSymbol (using already parsed components)
    def create_standard_symbol(row):
        if pd.isna(row['Expiration_Date']) or pd.isna(row['Strike_Price']) or pd.isna(row['Option_Type']) or pd.isna(row['ParsedUnderlying']):
            return "INVALID_OPTION_DATA"
        try:
            # Clean strike (e.g., 160.0 -> 160, 160.5 -> 160.5)
            strike_val = float(row['Strike_Price'])
            strike_fmt = f"{strike_val:.2f}".replace(".00", "") if strike_val == int(strike_val) else f"{strike_val:.2f}"
        except ValueError:
            strike_fmt = str(row['Strike_Price']) # Fallback

        return f"{str(row['ParsedUnderlying']).upper()}_{pd.Timestamp(row['Expiration_Date']).strftime('%y%m%d')}_{str(row['Option_Type']).upper()}_{strike_fmt}"
    
    df['StandardOptionSymbol'] = df.apply(create_standard_symbol, axis=1)

    # 2. Notional Value
    df['NotionalValue'] = df['TradeQuantity'] * df['Trade_Price'] * 100 

    # 3. Aggressor Calculation (using existing logic from provided file, ensured numeric inputs)
    aggressor_conditions = [
        (df['Trade_Price'].notna() & df['Option_Ask'].notna() & (df['Trade_Price'] >= df['Option_Ask']) & (df['Trade_Price'] > 0) & (df['Option_Ask'] > 0)),
        (df['Trade_Price'].notna() & df['Option_Bid'].notna() & (df['Trade_Price'] <= df['Option_Bid']) & (df['Trade_Price'] > 0) & (df['Option_Bid'] > 0)),
        (df['Trade_Price'].notna() & df['Option_Bid'].notna() & df['Option_Ask'].notna() &
         (df['Trade_Price'] > df['Option_Bid']) & (df['Trade_Price'] < df['Option_Ask']))
    ]
    aggressor_choices = ['Buyer (At Ask)', 'Seller (At Bid)', 'Between Bid/Ask']
    df['Aggressor'] = np.select(aggressor_conditions, aggressor_choices, default='Unknown/No Clear Market')

    # 4. TradeDirection (Numeric Aggressor)
    aggressor_map = {'Buyer (At Ask)': 1, 'Seller (At Bid)': -1}
    df['TradeDirection'] = df['Aggressor'].map(aggressor_map).fillna(0).astype(int)

    # 5. Per-Trade Aggressive Volumes & OFI Quantity
    # Ensure TradeQuantity is numeric and non-negative before these calculations
    df['TradeQuantity'] = df['TradeQuantity'].fillna(0) # Ensure no NaNs for calculation

    df['AggressiveBuyVolume'] = df.apply(lambda row: row['TradeQuantity'] if row['TradeDirection'] == 1 else 0, axis=1)
    df['AggressiveSellVolume'] = df.apply(lambda row: row['TradeQuantity'] if row['TradeDirection'] == -1 else 0, axis=1)
    df['Trade_OFI_Qty'] = df['AggressiveBuyVolume'] - df['AggressiveSellVolume']
    
    # --- End of New Feature Additions ---

    # Define the order and selection of final columns, including new ones
    final_columns_ordered = [
        'Time', 'Option_Description_orig', 'ParsedUnderlying', 'StandardOptionSymbol', 
        'Expiration_Date', 'Strike_Price', 'Option_Type',
        'TradeQuantity', 'Trade_Price', 'NotionalValue', 
        'Option_Bid', 'Option_Ask', 'Aggressor', 'TradeDirection',
        'AggressiveBuyVolume', 'AggressiveSellVolume', 'Trade_OFI_Qty',
        'Delta', 'IV', 'Underlying_Price', 
        'Condition', 'Exchange'
    ]
    
    # Ensure all final columns exist, adding them with NaNs/defaults if they were somehow missed
    for col in final_columns_ordered:
        if col not in df.columns:
            if col in ['AggressiveBuyVolume', 'AggressiveSellVolume', 'Trade_OFI_Qty', 'NotionalValue']: # Numeric defaults
                 df[col] = 0.0
            else:
                 df[col] = np.nan
    
    cleaned_df = df[final_columns_ordered].copy() # Select and order columns

    # Drop rows with critical NaNs that would hinder analysis if any were introduced
    # e.g. if Time or StandardOptionSymbol or core option characteristics are still NaN
    cleaned_df.dropna(subset=['Time', 'StandardOptionSymbol', 'Expiration_Date', 'Strike_Price', 'Option_Type', 'TradeQuantity', 'Trade_Price'], 
                      how='any', inplace=True)


    if cleaned_df.empty:
        log_status_func(f"No valid trade data remaining after processing for {active_ticker}.")
    else:
        log_status_func(f"Data processing complete for {active_ticker}. Added new features. Shape: {cleaned_df.shape}")

    # Store active_ticker and processing time as DataFrame attributes
    cleaned_df.attrs['active_ticker'] = active_ticker
    cleaned_df.attrs['processing_timestamp'] = datetime.now()
        
    return cleaned_df, parsed_lines, skipped_na_price_lines, malformed_lines
