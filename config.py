# config.py
# This file will store configuration constants, default paths, and theme definitions.

import os

# --- Directory and File Configurations ---
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.getcwd(), 'Parsed_Data_CSV') 
ANALYSIS_TXT_OUTPUT_DIRECTORY = os.path.join(os.getcwd(), 'Parsed_Data_TXT_Reports') 
PERSISTENT_CLEANED_DATA_DIRECTORY = os.path.join(os.getcwd(), 'Historical_Cleaned_Data') 
LAST_TRADE_META_DIRECTORY = os.path.join(os.getcwd(), 'Processing_Metadata') 

EXCEL_WORKBOOK_NAME = 'Your_ToS_RTD_Workbook.xlsx' # IMPORTANT: User needs to update this
EXCEL_RTD_SHEET_NAME = 'Master_RTD_Sheet' 

# --- Parsing Configurations (from data_utils.py needs) ---
EXPECTED_FIELD_COUNT_FULL = 10
EXPECTED_FIELD_COUNT_NO_CONDITION = 9
PRICE_FIELD_INDEX = 3 # Index of the 'Trade_Price_str' field in raw tab-separated data

# --- Analysis Engine Constants ---

# **Moneyness & DTE Categorization**
OTM_DELTA_THRESHOLD = 0.30  # Abs delta below or equal to this is OTM
# ATM is between OTM_DELTA_THRESHOLD and ITM_DELTA_THRESHOLD
ITM_DELTA_THRESHOLD = 0.70  # Abs delta above or equal to this is ITM
# Note: ATM_DELTA_LOWER = 0.30 from your file seems redundant if OTM_DELTA_THRESHOLD defines the boundary.
# We can simplify to just OTM and ITM thresholds, with ATM being in between.
# Or, define ATM explicitly:
ATM_DELTA_LOWER_BOUND = 0.40 # Example: Abs delta >= 0.40 for ATM lower bound
ATM_DELTA_UPPER_BOUND = 0.60 # Example: Abs delta <= 0.60 for ATM upper bound
# Choose one approach for moneyness delta thresholds. For now, I'll keep your existing ones
# and you can decide if ATM_DELTA_LOWER is still needed or how it's used with the other two.

SHORT_DTE_MAX = 30          # DTE <= this is Short-Term
MID_DTE_MAX = 90            # DTE > SHORT_DTE_MAX and <= MID_DTE_MAX is Mid-Term
                            # DTE > MID_DTE_MAX is Long-Term

# **Trade Event Identification**
# Block Trades
LARGE_TRADE_THRESHOLD_QTY = 50
VERY_LARGE_TRADE_THRESHOLD_QTY = 250 
LARGE_TRADE_NOTIONAL_THRESHOLD = 200000 # New: For block trades by notional value (e.g., $200,000)

# Sweep Orders
SWEEP_MAX_TIME_DIFF_MS = 500  # Max time between legs of a sweep in milliseconds
SWEEP_MIN_TRADES = 3          # Minimum number of trades to constitute a sweep
SWEEP_MIN_EXCHANGES = 2       # Minimum number of different exchanges involved in a sweep
SWEEP_STRONG_QTY_THRESHOLD = 200 # Threshold for a sweep to be considered "strong" by quantity

# Strategy Identification
STRATEGY_TIME_WINDOW_SECONDS = 2 # Max time between legs of an identified strategy

# **Volatility Analysis**
IV_HIGH_THRESHOLD = 0.70      # IV considered "High" (e.g., 70%)
IV_LOW_THRESHOLD = 0.30       # IV considered "Low" (e.g., 30%)
IV_ROLLING_WINDOW = 20        # New: Rolling window size for IV mean/std dev (e.g., 20 trades/periods)
IV_STD_DEV_THRESHOLD = 2.5    # New: Std deviation multiple for IV spike/collapse detection

# **Flow & Aggression Thresholds**
OFI_STRONG_THRESHOLD_DOLLAR = 500000 
AGGRESSION_STRONG_THRESHOLD_CONTRACTS = 500 

# --- High-Frequency (HF) Analysis Parameters ---
HF_TIME_RESAMPLE_FREQ = '1s' 
HF_ROLLING_WINDOW_FOR_STATS = 60 
HF_Z_SCORE_THRESHOLD_OFI = 2.5   
HF_Z_SCORE_THRESHOLD_AGG = 2.5   
HF_MIN_BURST_DURATION_WINDOWS = 2 
HF_MIN_OFI_VALUE_FOR_BURST_ABS = 10000 
HF_MIN_AGG_QTY_FOR_BURST_ABS = 25      
HF_BURST_TOP_N_OPTIONS = 3 
HF_BURST_TOP_N_OPTIONS_PER_BURST = 3 
HF_BURST_BRIEFING_TOP_N_DISPLAY = 3 

# --- Briefing, Reporting & RTD ---
MAX_OPTIONS_FOR_RTD = 15 
MAX_OPTIONS_FOR_BRIEFING_MAP = 25 
MAX_OPTIONS_FOR_BRIEFING_DISPLAY = 10 
TOP_N_VOLUME_OPTIONS = 5 
TOP_N_AGGRESSIVE_OPTIONS = 3 
TXT_REPORT_TOP_N_AGGRESSIVE_OPTIONS = 10 

# --- Weighting for Options of Interest Scoring (NEW SECTION) ---
# These are conceptual weights; actual values need tuning.
# Example:
# WEIGHT_NOTIONAL_VALUE = 0.0001 # Score = NotionalValue * WEIGHT_NOTIONAL_VALUE
# WEIGHT_AGGRESSION_BUY_AT_ASK = 50
# WEIGHT_AGGRESSION_SELL_AT_BID = 50
# WEIGHT_SWEEP_ORDER = 100
# WEIGHT_HF_BURST_PARTICIPATION = 75
# WEIGHT_IV_SPIKE = 60
# (This section can be expanded as the scoring logic in analysis_engine.py is developed)


# --- UI Theme Configurations --- (Existing themes are good)
THEMES = {
    "light": {
        "bg": "SystemButtonFace", "fg": "black", "text_bg": "white", "text_fg": "black",
        "button_bg": "#E1E1E1", "button_fg": "black", "button_active_bg": "#C0C0C0",
        "entry_bg": "white", "entry_fg": "black", "label_bg": "SystemButtonFace", "label_fg": "black",
        "frame_bg": "SystemButtonFace", "status_bg": "#F0F0F0", "status_fg": "black",
        "results_bg": "#FDFDFD", "results_fg": "black",
        "ttk_style_bg": "SystemButtonFace", "ttk_style_fg": "black",
        "ttk_entry_select_bg": "lightblue", "ttk_entry_select_fg": "black",
        "ttk_button_focus_color": "blue", "notebook_tab_bg": "#D0D0D0", "notebook_tab_fg": "black",
        "notebook_tab_selected_bg": "SystemButtonFace", "notebook_tab_selected_fg": "black",
    },
    "dark": {
        "bg": "#2E2E2E", "fg": "white", "text_bg": "#3C3C3C", "text_fg": "white",
        "button_bg": "#555555", "button_fg": "white", "button_active_bg": "#656565",
        "entry_bg": "#3C3C3C", "entry_fg": "white", "label_bg": "#2E2E2E", "label_fg": "white",
        "frame_bg": "#2E2E2E", "status_bg": "#1E1E1E", "status_fg": "white",
        "results_bg": "#252525", "results_fg": "#C8C8C8",
        "ttk_style_bg": "#2E2E2E", "ttk_style_fg": "white",
        "ttk_entry_select_bg": "#0078D7", "ttk_entry_select_fg": "white",
        "ttk_button_focus_color": "lightblue", "notebook_tab_bg": "#404040", "notebook_tab_fg": "white",
        "notebook_tab_selected_bg": "#2E2E2E", "notebook_tab_selected_fg": "white",
    }
}
