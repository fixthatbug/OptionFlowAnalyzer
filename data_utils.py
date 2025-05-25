# data_utils.py
# Helper functions for data conversion and the core logic for processing raw options data.

import pandas as pd
import re
import numpy as np
from datetime import datetime
# config constants will be imported by main_app.py and passed if needed,
# or this module can import config directly if it's in the same path.

# --- Helper functions for safe data conversion ---
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
                # Heuristic for IVs like 120.5 (meaning 120.5%) vs 0.98 (meaning 0.98 or 98%)
                # If val > 1.5 (e.g. 55.5), assume it's a percentage value not yet decimalized.
                return val / 100.0 if abs(val) > 1.5 else val 
        return float(cleaned_value_str)
    except (ValueError, TypeError):
        return np.nan

def safe_int_conversion(value_str):
    if pd.isna(value_str) or isinstance(value_str, str) and value_str.strip().upper() == "N/A":
        return np.nan
    try:
        # Remove commas for numbers like "1,234"
        return int(str(value_str).replace(',', '').strip())
    except (ValueError, TypeError):
        return np.nan

def parse_tos_option_symbol_for_data_utils(symbol_str):
    """
    Parses a Thinkorswim RTD option symbol string into its components.
    Helper for parse_option_description.
    Example: .SPXW240719C4900000 -> ticker=SPXW, year=24, month=07, day=19, type=C, strike=4900000 (needs division if cents implied)
    This version assumes strike is directly parseable as float.
    """
    if not symbol_str or not isinstance(symbol_str, str) or not symbol_str.startswith('.'):
        return None 
    # Regex updated to handle potential variations in ticker length and strike format.
    # Assumes strike at the end is the full strike price value.
    match = re.match(r"^\.(?P<ticker>[A-Z0-9./]+)(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})(?P<type>[CP])(?P<strike>\d+\.?\d*)$", symbol_str.upper())
    if match:
        data = match.groupdict()
        try:
            # Ensure year is correctly interpreted (e.g., 20xx)
            exp_date = pd.to_datetime(f"{int(data['year'])+2000}-{data['month']}-{data['day']}")
            strike = float(data["strike"]) # ToS RTD strikes usually don't need division by 1000 unless it's for mini contracts or specific formatting.
            opt_type = data["type"]
            return exp_date, strike, opt_type
        except (ValueError, TypeError):
            return None
    return None

def parse_option_description(desc_str):
    """
    Parses an option description string to extract expiration date, strike price, and option type.
    Attempts common "human-readable" formats first, then ToS-style symbols as a fallback.
    """
    if pd.isna(desc_str): return pd.NaT, np.nan, None
    desc_str_original = str(desc_str).strip()
    desc_str_upper = desc_str_original.upper() 

    option_type = None
    strike_price = np.nan
    expiration_date = pd.NaT

    # Try human-readable format first: "DD MON YY STRIKE TYPE" or "MON DD YY STRIKE TYPE"
    # Example: 20 JUN 25 28 P
    match_human_readable = re.search(r'([\d\.]+)\s*([CP])$', desc_str_upper) # Strike and Type at the end
    if match_human_readable:
        strike_price_str = match_human_readable.group(1)
        parsed_option_type = match_human_readable.group(2)
        parsed_strike_price = safe_float_conversion(strike_price_str)
        
        date_str_part = desc_str_upper[:match_human_readable.start()].strip()
        
        # Try common date formats found in options descriptions
        date_formats_to_try = [
            '%d %b %y', # 20 JUN 25
            '%d%b%y',   # 20JUN25
            '%b %d %y', # JUN 20 25
            '%b%d%y',   # JUN2025
        ]
        parsed_expiration_date = pd.NaT
        for fmt in date_formats_to_try:
            try:
                parsed_expiration_date = pd.to_datetime(date_str_part, format=fmt)
                break # Successfully parsed
            except ValueError:
                continue
        
        if pd.isna(parsed_expiration_date): # Fallback if specific formats fail
            try: 
                parsed_expiration_date = pd.to_datetime(date_str_part)
            except (ValueError, TypeError, pd.errors.ParserError):
                pass # expiration_date remains pd.NaT
        
        # If all parts are valid from human-readable, return them
        if pd.notna(parsed_expiration_date) and pd.notna(parsed_strike_price) and parsed_option_type:
            return parsed_expiration_date, parsed_strike_price, parsed_option_type

    # Fallback: Try parsing as a ToS-style symbol if human-readable fails or isn't matched
    parsed_tos_style = parse_tos_option_symbol_for_data_utils(desc_str_original) # Use the original case for ToS parser if it matters
    if parsed_tos_style:
        exp_date_tos, strike_tos, type_tos = parsed_tos_style
        if pd.notna(exp_date_tos) and pd.notna(strike_tos) and type_tos:
            return exp_date_tos, strike_tos, type_tos
            
    # If neither parsing method works
    return pd.NaT, np.nan, None


def parse_market_at_trade(market_str):
    if pd.isna(market_str) or not isinstance(market_str, str) or 'x' not in market_str.lower() :
        if isinstance(market_str, str) and market_str.strip().upper() == "N/AXN/A":
            return np.nan, np.nan
        return np.nan, np.nan # Default for unparseable

    parts = market_str.lower().split('x') # Split by 'x' case-insensitively
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
def process_raw_data_string(raw_data_string, price_field_index, expected_field_count_full, expected_field_count_no_condition):
    lines = raw_data_string.strip().split('\n')
    list_of_trade_records = []
    skipped_na_price_lines = 0
    parsed_lines = 0
    malformed_lines = 0

    for line_number, line in enumerate(lines, 1):
        line = line.strip()
        if not line: continue

        fields = line.split('\t')

        if len(fields) > price_field_index and fields[price_field_index].strip().upper() == "N/A":
            skipped_na_price_lines += 1
            continue

        record = None
        # Try to match based on expected field counts
        if len(fields) == expected_field_count_no_condition:
            record = {
                'Time_str': fields[0].strip(), 'Option_Description_str': fields[1].strip(),
                'TradeQuantity_str': fields[2].strip(), 'Trade_Price_str': fields[3].strip(),
                'Exchange_str': fields[4].strip(), 'Market_at_trade_str': fields[5].strip(),
                'Delta_str': fields[6].strip(), 'IV_str': fields[7].strip(),
                'Underlying_Price_str': fields[8].strip(), 'Condition_str': None # No condition field
            }
        elif len(fields) >= expected_field_count_full: # Allows for extra fields at the end if any
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
        return None, parsed_lines, skipped_na_price_lines, malformed_lines
    
    df = pd.DataFrame(list_of_trade_records)
    
    if not df.empty:
        # Time parsing
        df['Time'] = pd.to_datetime(df['Time_str'], format='%m/%d/%y %H:%M:%S.%f', errors='coerce')
        # Fallback if millisecond parsing fails for all
        if df['Time'].isnull().all() and not df.empty: 
            df['Time'] = pd.to_datetime(df['Time_str'], format='%m/%d/%y %H:%M:%S', errors='coerce')

        # Option Description Parsing
        # Apply the parse_option_description function and expand the tuple result into new columns
        parsed_desc_series = df['Option_Description_str'].apply(parse_option_description)
        parsed_desc_df = pd.DataFrame(parsed_desc_series.tolist(), index=df.index, columns=['Expiration_Date', 'Strike_Price', 'Option_Type'])
        df = pd.concat([df, parsed_desc_df], axis=1)

        # Market at Trade Parsing
        parsed_market_cols = df['Market_at_trade_str'].apply(lambda x: pd.Series(parse_market_at_trade(x), index=['Option_Bid', 'Option_Ask']))
        df = pd.concat([df, parsed_market_cols], axis=1)
        
        # Convert other string columns to appropriate types
        df['IV'] = df['IV_str'].apply(lambda x: safe_float_conversion(x, is_percentage=True))
        df['TradeQuantity'] = df['TradeQuantity_str'].apply(safe_int_conversion)
        df['Trade_Price'] = df['Trade_Price_str'].apply(safe_float_conversion)
        df['Delta'] = df['Delta_str'].apply(safe_float_conversion) # Delta is usually -1 to 1
        df['Underlying_Price'] = df['Underlying_Price_str'].apply(safe_float_conversion)
        
        # Rename columns for clarity and consistency
        df.rename(columns={
            'Option_Description_str': 'Option_Description_orig', # Keep original description
            'Condition_str': 'Condition',
            'Exchange_str': 'Exchange'
        }, inplace=True)
        
        # Define the order and selection of final columns
        final_columns = ['Time', 'Option_Description_orig', 'Expiration_Date', 'Strike_Price', 'Option_Type',
                         'TradeQuantity', 'Trade_Price', 'Option_Bid', 'Option_Ask', 'Delta', 'IV',
                         'Underlying_Price', 'Condition', 'Exchange']
        
        # Ensure all final columns exist, adding them with NaNs if they were somehow missed
        for col in final_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        cleaned_df = df[final_columns].copy()
        
        # Aggressor Calculation: Ensure necessary columns are numeric before calculation
        for col_to_num in ['Option_Bid', 'Option_Ask', 'Trade_Price']:
            if col_to_num in cleaned_df.columns:
                 cleaned_df[col_to_num] = pd.to_numeric(cleaned_df[col_to_num], errors='coerce')

        aggressor_conditions = [
            (cleaned_df['Trade_Price'].notna() & cleaned_df['Option_Ask'].notna() & (cleaned_df['Trade_Price'] >= cleaned_df['Option_Ask']) & (cleaned_df['Trade_Price'] > 0) & (cleaned_df['Option_Ask'] > 0)),
            (cleaned_df['Trade_Price'].notna() & cleaned_df['Option_Bid'].notna() & (cleaned_df['Trade_Price'] <= cleaned_df['Option_Bid']) & (cleaned_df['Trade_Price'] > 0) & (cleaned_df['Option_Bid'] > 0)),
            (cleaned_df['Trade_Price'].notna() & cleaned_df['Option_Bid'].notna() & cleaned_df['Option_Ask'].notna() &
             (cleaned_df['Trade_Price'] > cleaned_df['Option_Bid']) & (cleaned_df['Trade_Price'] < cleaned_df['Option_Ask']))
        ]
        aggressor_choices = ['Buyer (At Ask)', 'Seller (At Bid)', 'Between Bid/Ask']
        cleaned_df['Aggressor'] = np.select(aggressor_conditions, aggressor_choices, default='Unknown/No Clear Market')
        
        return cleaned_df, parsed_lines, skipped_na_price_lines, malformed_lines
        
    return None, parsed_lines, skipped_na_price_lines, malformed_lines
