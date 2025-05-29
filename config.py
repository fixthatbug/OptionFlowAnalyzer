# config.py
"""
Main configuration file for Options Flow & RTD Analyzer
"""

import os
from datetime import datetime

# --- Directory and File Configurations ---
DEFAULT_OUTPUT_DIRECTORY = os.path.join(os.getcwd(), 'Parsed_Data_CSV')
ANALYSIS_TXT_OUTPUT_DIRECTORY = os.path.join(os.getcwd(), 'Parsed_Data_TXT_Reports')
PERSISTENT_CLEANED_DATA_DIRECTORY = os.path.join(os.getcwd(), 'Historical_Cleaned_Data')
LAST_TRADE_META_DIRECTORY = os.path.join(os.getcwd(), 'Processing_Metadata')
ALPHA_SIGNALS_DIRECTORY = os.path.join(os.getcwd(), 'Alpha_Signals')
COORDINATES_CONFIG_FILE = os.path.join(os.getcwd(), 'tos_coordinates.json')

# Excel RTD Configuration
EXCEL_WORKBOOK_NAME = 'Options_RTD_Monitor.xlsx'
EXCEL_RTD_SHEET_NAME = 'RTD_Data'
EXCEL_ALPHA_SHEET_NAME = 'Alpha_Signals'

# --- ToS Window Configuration ---
USER_TOS_TARGET_WINDOW_WIDTH = 1024
USER_TOS_TARGET_WINDOW_HEIGHT = 768
USER_TOS_WINDOW_TITLE_REGEX_OVERRIDE = ""
USER_TOS_COORDINATES = {
    "ticker_symbol": {"x": None, "y": None},
    "option_statistics": {"x": None, "y": None, "width": None, "height": None},
    "time_sales": {"x": None, "y": None, "width": None, "height": None}
}

# --- Real-time Monitoring ---
USER_MONITORING_INTERVAL = 1000  # milliseconds
USER_AUTO_SAVE_COORDINATES = True
USER_AUTO_PROCESS_AFTER_GRAB = True
USER_MIN_NOTIONAL_FOR_BLOCK_ALERT = 25000
USER_REPORT_INCLUDE_DETAILED_GREEKS = True

# --- Parsing Configurations ---
EXPECTED_FIELD_COUNT_FULL = 10
EXPECTED_FIELD_COUNT_NO_CONDITION = 9
PRICE_FIELD_INDEX = 3

# --- Analysis Engine Constants ---
OTM_DELTA_THRESHOLD = 0.30
ATM_DELTA_LOWER = 0.30
ITM_DELTA_THRESHOLD = 0.70
SHORT_DTE_MAX = 30
MID_DTE_MAX = 90
LARGE_TRADE_THRESHOLD_QTY = 50
VERY_LARGE_TRADE_THRESHOLD_QTY = 250
LARGE_TRADE_NOTIONAL_THRESHOLD = 50000
OFI_STRONG_THRESHOLD_DOLLAR = 500000
OFI_MODERATE_THRESHOLD_DOLLAR = 100000
OFI_STRONG_THRESHOLD_QTY = 1000
OFI_MODERATE_THRESHOLD_QTY = 250

# Sweep Detection
SWEEP_TIME_WINDOW_MS = 2000
SWEEP_MIN_LEG_COUNT = 3
SWEEP_MIN_TOTAL_QTY = 10
SWEEP_MAX_PRICE_DEV_FROM_FIRST_LEG = 0.05

# High-Frequency Analysis
HF_TIME_RESAMPLE_FREQ = '15s'
HF_ROLLING_WINDOW_FOR_STATS = 20
HF_Z_SCORE_THRESHOLD_OFI = 2.5
HF_Z_SCORE_THRESHOLD_AGG = 2.5
HF_MIN_BURST_DURATION_WINDOWS = 2
HF_MIN_OFI_VALUE_FOR_BURST_ABS = 25000
HF_MIN_AGG_QTY_FOR_BURST_ABS = 50
HF_BURST_TOP_N_OPTIONS = 5

# Volatility Analysis
IV_ROLLING_WINDOW = 20
IV_STD_DEV_THRESHOLD = 2.0
IV_SPIKE_PCT_THRESHOLD = 0.15
MIN_IV_FOR_PCT_CHANGE_CALC = 0.05

# Display Limits
MAX_OPTIONS_FOR_RTD = 15
MAX_OPTIONS_FOR_BRIEFING_MAP = 30
TOP_N_OPTIONS_OF_INTEREST_BRIEFING = 7
MIN_OFI_FOR_INTERESTING_CONTRACT = 10000
MIN_QTY_FOR_INTERESTING_CONTRACT = 20

# Market Stance Calculation
STANCE_SCORE_BLOCK_TRADE_MULTIPLIER = 1.5
STANCE_SCORE_SWEEP_TRADE_MULTIPLIER = 2.0
STANCE_SCORE_HF_BURST_MULTIPLIER = 1.0
STANCE_CONVICTION_HIGH_THRESHOLD = 7
STANCE_CONVICTION_MODERATE_THRESHOLD = 3

# --- UI Theme Definitions ---
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
        "treeview_heading_bg": "#E0E0E0", "treeview_heading_fg": "black",
        "treeview_bg": "white", "treeview_fg": "black", "treeview_selected_bg": "lightblue",
        "alpha_bullish": "#00AA00", "alpha_bearish": "#CC0000", "alpha_neutral": "#0066CC",
        "select_bg": "lightblue"
    },
    "dark": {
        "bg": "#2E2E2E", "fg": "white", "text_bg": "#3C3C3C", "text_fg": "white",
        "button_bg": "#555555", "button_fg": "white", "button_active_bg": "#656565",
        "entry_bg": "#3C3C3C", "entry_fg": "white", "label_bg": "#2E2E2E", "label_fg": "white",
        "frame_bg": "#2E2E2E", "status_bg": "#3C3C3C", "status_fg": "white",
        "results_bg": "#333333", "results_fg": "white",
        "ttk_style_bg": "#2E2E2E", "ttk_style_fg": "white",
        "ttk_entry_select_bg": "#0078D7", "ttk_entry_select_fg": "white",
        "ttk_button_focus_color": "white", "notebook_tab_bg": "#4A4A4A", "notebook_tab_fg": "white",
        "notebook_tab_selected_bg": "#2E2E2E", "notebook_tab_selected_fg": "white",
        "treeview_heading_bg": "#3A3A3A", "treeview_heading_fg": "white",
        "treeview_bg": "#2C2C2C", "treeview_fg": "white", "treeview_selected_bg": "#005A9E",
        "alpha_bullish": "#00FF00", "alpha_bearish": "#FF4444", "alpha_neutral": "#4488FF",
        "select_bg": "#0078D7"
    }
}

# Create directories if they don't exist
for dir_path in [DEFAULT_OUTPUT_DIRECTORY, ANALYSIS_TXT_OUTPUT_DIRECTORY, 
                 PERSISTENT_CLEANED_DATA_DIRECTORY, LAST_TRADE_META_DIRECTORY,
                 ALPHA_SIGNALS_DIRECTORY]:
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    except Exception as e:
        print(f"Warning: Could not create directory {dir_path}: {e}")

ANALYSIS_LOG_LEVEL = "INFO"

# Analysis Module Settings
ENABLE_ADVANCED_PATTERNS = True
ENABLE_GREEK_ANALYSIS = True
ENABLE_MICROSTRUCTURE_ANALYSIS = True
ENABLE_STRATEGY_DETECTION = True

# Performance Settings
MAX_DATA_ROWS_FOR_ANALYSIS = 100000
ANALYSIS_TIMEOUT_SECONDS = 300
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKER_THREADS = 4

# Notification Settings
ENABLE_AUDIO_ALERTS = False
ENABLE_POPUP_ALERTS = True
ALERT_SOUND_FILE = None

# Export Settings
DEFAULT_EXPORT_FORMAT = 'csv'
INCLUDE_METADATA_IN_EXPORTS = True
COMPRESS_LARGE_EXPORTS = True

# Debug Settings
DEBUG_MODE = False
VERBOSE_LOGGING = False
SAVE_DEBUG_DATA = False

# Application Metadata
APP_VERSION = "3.0.0"
APP_NAME = "Options Flow & RTD Analyzer - Alpha Edition"
AUTHOR = "Advanced Trading Systems"