# analysis_modules/analysis_helpers.py
"""
Helper functions for analysis modules
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any

def get_sort_key(option_detail: dict) -> float:
    """Get sort key for option details"""
    return option_detail.get('score', 0)

def calculate_moneyness(strike: float, underlying: float) -> str:
    """Calculate option moneyness"""
    if abs(strike - underlying) / underlying < 0.02:
        return "ATM"
    elif strike < underlying:
        return "ITM" if strike > 0 else "OTM"
    else:
        return "OTM" if strike > underlying else "ITM"

def format_currency(amount: float) -> str:
    """Format currency amount"""
    if abs(amount) >= 1e6:
        return f"${amount/1e6:.1f}M"
    elif abs(amount) >= 1e3:
        return f"${amount/1e3:.0f}K"
    else:
        return f"${amount:.0f}"

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default