# analysis_modules/analysis_helpers.py
# Utility functions and shared constants for the analysis engine.

import pandas as pd
import numpy as np
from datetime import datetime 
import config # Import the config module

# --- Module-Level Definitions for Sorting Options of Interest ---
# REASON_PRIORITY is defined directly as it does not depend on config at definition time.
REASON_PRIORITY = {
    "Large Straddle Identified": 0, 
    "Large Strangle Identified": 0, 
    "Buy Sweep Order": 1, 
    "Sell Sweep Order": 1,
    "Active in HF NetDollarOFI Buying Burst": 2, 
    "Active in HF NetAggressorQty Buying Burst": 2,
    "Active in HF NetDollarOFI Selling Burst": 2, 
    "Active in HF NetAggressorQty Selling Burst": 2,
    "Individual Block Trade": 3, 
    "Top Net Aggressive Call Buy Qty": 4, 
    "Top Net Aggressive Put Sell Qty": 4, 
    "Top Net Aggressive Put Buy Qty": 4, 
    "Top Net Aggressive Call Sell Qty": 4, 
    "High Total Volume": 5
}

def get_sort_key(item):
    """
    Helper function to determine the sort key for an item in options_of_interest_details.
    Sorts by:
    1. Predefined reason priority (lower is better).
    2. Time (earlier is better for timed events like sweeps, HF bursts).
    3. Metric (absolute value, descending - larger is better).
    """
    reason_full = item.get('reason', 'Unknown Reason') 
    metric_val_raw = item.get('metric', 0) 

    if "Sweep Order" in reason_full:
        reason_base = " ".join(reason_full.split(" ")[:3]) 
    elif "Active in HF" in reason_full: 
        reason_base = " ".join(reason_full.split(" ")[:4]) 
    else: 
        reason_base = reason_full
    
    priority = REASON_PRIORITY.get(reason_base, 99) 
    
    time_val = item.get('time', pd.Timestamp.max) 
    if not isinstance(time_val, pd.Timestamp): 
        time_val = pd.Timestamp.max

    metric_val_abs = 0
    if isinstance(metric_val_raw, (int, float)) and pd.notna(metric_val_raw):
        metric_val_abs = abs(metric_val_raw)
        
    return (priority, time_val, -metric_val_abs)

def categorize_moneyness(delta, option_type=None):
    """
    Categorizes an option's moneyness based on its delta.
    Accesses delta thresholds directly from the imported config module.
    """
    if pd.isna(delta): return "Unknown"
    try:
        abs_delta = abs(float(delta))
        # Access config attributes directly inside the function
        if abs_delta <= config.OTM_DELTA_THRESHOLD: return "OTM"
        elif config.OTM_DELTA_THRESHOLD < abs_delta < config.ITM_DELTA_THRESHOLD: return "ATM"
        elif abs_delta >= config.ITM_DELTA_THRESHOLD: return "ITM"
    except ValueError:
        return "Unknown" 
    return "Unknown"

def categorize_dte(expiration_date, trade_date):
    """
    Categorizes an option's DTE (Days To Expiration).
    Accesses DTE thresholds directly from the imported config module.
    """
    if pd.isna(expiration_date) or pd.isna(trade_date): return "Unknown"
    try:
        exp_date_ts = pd.Timestamp(expiration_date)
        trade_date_ts = pd.Timestamp(trade_date)
        dte = (exp_date_ts - trade_date_ts).days
        if dte < 0: return "Expired"
        # Access config attributes directly inside the function
        if dte <= config.SHORT_DTE_MAX: return "Short-Term (0-30D)"
        elif dte <= config.MID_DTE_MAX: return "Mid-Term (31-90D)"
        else: return "Long-Term (>90D)"
    except Exception: 
        return "Unknown"

