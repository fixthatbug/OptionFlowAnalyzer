# main_app.py
"""
Main application for Options Flow & RTD Analyzer with Alpha Extraction
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog, scrolledtext
import pandas as pd
import numpy as np
import os
import json
import threading
import queue
import platform
from datetime import datetime
import time
import re

# Local imports
import config
import alpha_config
import data_utils
import rtd_handler
import analysis_engine
import ui_builder
from coordinate_setup_wizard import CoordinateSetupWizard
from monitoring_manager import MonitoringManager
from data_fusion import DataFusionEngine
from alpha_extractor import AlphaExtractor, generate_signal_report
from real_time_alpha_monitor import RealTimeAlphaMonitor
from analysis_modules.alert_manager import AlertManager
import tos_data_grabber
# Add these imports at the top of main_app.py
from rtd_handler import RTDDataHandler, fetch_and_display_rtd_data_for_tab
from rtd_integration import RTDAnalysisIntegrator, setup_comprehensive_rtd_integration, generate_rtd_integration_report

# Global variables
current_theme = "dark"
root = None
style = None
widgets = {}
tab_ui_widgets = {}
main_notebook = None

# Application data storage
app_data = {
    "cleaned_dfs": {},
    "raw_data_strings": {},
    "analysis_outputs": {},
    "options_for_rtd": {},
    "rtd_data_cache": {},
    "last_trade_timestamps": {},
    "tracked_tos_windows": {},
    "monitoring_manager": None,
    "data_fusion_engine": None,
    "alpha_monitors": {},
    "active_alpha_signals": {},
    "alert_manager": None,
    "coordinates_config": None,
    "rtd_handlers": {},  # RTD handlers by ticker
    "rtd_integrators": {},  # RTD integrators by ticker
    "rtd_active_signals": {},  # Active RTD signals by ticke
}

# Queues for thread communication
analysis_results_queue = queue.Queue()
alpha_signals_queue = queue.Queue()
monitoring_data_queue = queue.Queue()

class SettingsDialog(tk.Toplevel):
    """Settings dialog for application configuration"""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Application Settings")
        self.parent = parent
        self.transient(parent)
        self.grab_set()
        self.geometry("600x700")
        
        # Create variables
        self.tos_width_var = tk.IntVar(value=config.USER_TOS_TARGET_WINDOW_WIDTH)
        self.tos_height_var = tk.IntVar(value=config.USER_TOS_TARGET_WINDOW_HEIGHT)
        self.tos_title_regex_var = tk.StringVar(value=config.USER_TOS_WINDOW_TITLE_REGEX_OVERRIDE)
        self.monitoring_interval_var = tk.IntVar(value=config.USER_MONITORING_INTERVAL)
        self.auto_process_var = tk.BooleanVar(value=config.USER_AUTO_PROCESS_AFTER_GRAB)
        self.min_notional_alert_var = tk.IntVar(value=config.USER_MIN_NOTIONAL_FOR_BLOCK_ALERT)
        self.report_greeks_var = tk.BooleanVar(value=config.USER_REPORT_INCLUDE_DETAILED_GREEKS)
        
        # Alpha settings
        self.alpha_min_confidence_var = tk.IntVar(value=alpha_config.ALPHA_MIN_CONFIDENCE)
        self.alpha_urgent_threshold_var = tk.IntVar(value=alpha_config.ALPHA_URGENT_SIGNAL_THRESHOLD)
        self.alpha_buffer_interval_var = tk.IntVar(value=alpha_config.ALPHA_BUFFER_PROCESS_INTERVAL)
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def create_widgets(self):
        """Create all settings widgets"""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(expand=True, fill="both")
        
        # Create notebook for organized settings
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True)
        
        # ToS Settings Tab
        tos_frame = ttk.Frame(notebook, padding="10")
        notebook.add(tos_frame, text="Thinkorswim")
        
        ttk.Label(tos_frame, text="Target Window Width:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(tos_frame, textvariable=self.tos_width_var, width=15).grid(row=0, column=1, pady=5)
        
        ttk.Label(tos_frame, text="Target Window Height:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(tos_frame, textvariable=self.tos_height_var, width=15).grid(row=1, column=1, pady=5)
        
        ttk.Label(tos_frame, text="Window Title Regex:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(tos_frame, textvariable=self.tos_title_regex_var, width=40).grid(row=2, column=1, pady=5)
        
        ttk.Label(tos_frame, text="Monitoring Interval (ms):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(tos_frame, textvariable=self.monitoring_interval_var, width=15).grid(row=3, column=1, pady=5)
        
        ttk.Button(tos_frame, text="Setup Coordinates", 
                  command=self.open_coordinate_setup).grid(row=4, column=0, columnspan=2, pady=20)
        
        # Analysis Settings Tab
        analysis_frame = ttk.Frame(notebook, padding="10")
        notebook.add(analysis_frame, text="Analysis")
        
        ttk.Checkbutton(analysis_frame, text="Auto-process after grab", 
                       variable=self.auto_process_var).grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        
        ttk.Label(analysis_frame, text="Min Block Alert ($):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(analysis_frame, textvariable=self.min_notional_alert_var, width=15).grid(row=1, column=1, pady=5)
        
        ttk.Checkbutton(analysis_frame, text="Include detailed Greeks in reports", 
                       variable=self.report_greeks_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        # Alpha Settings Tab
        alpha_frame = ttk.Frame(notebook, padding="10")
        notebook.add(alpha_frame, text="Alpha Extraction")
        
        ttk.Label(alpha_frame, text="Min Signal Confidence (%):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(alpha_frame, textvariable=self.alpha_min_confidence_var, width=15).grid(row=0, column=1, pady=5)
        
        ttk.Label(alpha_frame, text="Urgent Signal Threshold:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(alpha_frame, textvariable=self.alpha_urgent_threshold_var, width=15).grid(row=1, column=1, pady=5)
        
        ttk.Label(alpha_frame, text="Buffer Process Interval (s):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(alpha_frame, textvariable=self.alpha_buffer_interval_var, width=15).grid(row=2, column=1, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.pack(fill="x", side="bottom")
        
        ttk.Button(button_frame, text="Apply", command=self.apply_settings).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side="right")
        
        # Configure column weights
        tos_frame.columnconfigure(1, weight=1)
        analysis_frame.columnconfigure(1, weight=1)
        alpha_frame.columnconfigure(1, weight=1)
    
    def open_coordinate_setup(self):
        """Open coordinate setup wizard"""
        self.withdraw()
        wizard = CoordinateSetupWizard(self.parent)
        self.wait_window(wizard)
        self.deiconify()
    
    def apply_settings(self):
        """Apply all settings"""
        try:
            # Update config values
            config.USER_TOS_TARGET_WINDOW_WIDTH = self.tos_width_var.get()
            config.USER_TOS_TARGET_WINDOW_HEIGHT = self.tos_height_var.get()
            config.USER_TOS_WINDOW_TITLE_REGEX_OVERRIDE = self.tos_title_regex_var.get().strip()
            config.USER_MONITORING_INTERVAL = self.monitoring_interval_var.get()
            config.USER_AUTO_PROCESS_AFTER_GRAB = self.auto_process_var.get()
            config.USER_MIN_NOTIONAL_FOR_BLOCK_ALERT = self.min_notional_alert_var.get()
            config.USER_REPORT_INCLUDE_DETAILED_GREEKS = self.report_greeks_var.get()
            
            # Update alpha config
            alpha_config.ALPHA_MIN_CONFIDENCE = self.alpha_min_confidence_var.get()
            alpha_config.ALPHA_URGENT_SIGNAL_THRESHOLD = self.alpha_urgent_threshold_var.get()
            alpha_config.ALPHA_BUFFER_PROCESS_INTERVAL = self.alpha_buffer_interval_var.get()
            
            log_status_to_ui("Settings updated successfully", also_to_analysis_area=False)
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Settings Error", f"Failed to apply settings: {e}", parent=self)
    
    def cancel(self):
        """Cancel and close dialog"""
        self.destroy()

def log_status_to_ui(message, active_ticker_for_log=None, also_to_analysis_area=True, is_error=False):
    """Log status message to UI"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    
    # Update status label
    status_label_widget = widgets.get("status_label")
    if status_label_widget and status_label_widget.winfo_exists():
        theme_config = config.THEMES.get(current_theme, config.THEMES['light'])
        original_color = theme_config.get("status_fg", "black")
        status_label_widget.config(text=full_message, foreground="red" if is_error else original_color)
    
    # Log to ticker-specific area
    target_ticker_log = active_ticker_for_log if active_ticker_for_log else get_active_ticker()
    if also_to_analysis_area and target_ticker_log:
        try:
            current_tab_ui_local = tab_ui_widgets.get(target_ticker_log)
            if current_tab_ui_local:
                analysis_log_area = current_tab_ui_local.get("analysis_log_area")
                if analysis_log_area and analysis_log_area.winfo_exists():
                    current_state = analysis_log_area.cget("state")
                    if current_state == tk.DISABLED:
                        analysis_log_area.config(state=tk.NORMAL)
                    analysis_log_area.insert(tk.END, full_message + "\n")
                    analysis_log_area.see(tk.END)
                    if current_state == tk.DISABLED:
                        analysis_log_area.config(state=tk.DISABLED)
        except Exception as e:
            print(f"Error logging to analysis area for {target_ticker_log}: {e}")
    
    print(full_message)

def handle_toggle_theme():
    """Toggle between light and dark themes"""
    global current_theme
    current_theme = "dark" if current_theme == "light" else "light"
    if root and style:
        ui_builder.apply_theme_to_widgets(root, style, current_theme, widgets)
    log_status_to_ui(f"Theme switched to {current_theme}")

def get_active_ticker():
    """Get currently active ticker symbol"""
    if not widgets.get("main_notebook") or not widgets["main_notebook"].tabs():
        return None
    try:
        return widgets["main_notebook"].tab(widgets["main_notebook"].select(), "text").split(" - ")[0].strip()
    except tk.TclError:
        return None
    except Exception:
        return None

def get_active_tab_id_path():
    """Get active tab widget path"""
    if not widgets.get("main_notebook") or not widgets["main_notebook"].tabs():
        return None
    try:
        return widgets["main_notebook"].select()
    except tk.TclError:
        return None

def handle_open_settings_dialog():
    """Open settings dialog"""
    SettingsDialog(root)

def rename_tos_window(window_handle: int, new_title: str):
    """Rename ToS window to ticker symbol"""
    if not tos_data_grabber.PYWINAUTO_AVAILABLE:
        return False
    
    try:
        import win32gui
        # Set new window title
        win32gui.SetWindowText(window_handle, new_title)
        return True
    except Exception as e:
        raise Exception(f"Failed to rename window: {e}")

def handle_select_tos_window(ticker_symbol):
    """Step 3-6: Select and setup ToS window for ticker"""
    
    if not platform.system() == "Windows" or not tos_data_grabber.PYWINAUTO_AVAILABLE:
        messagebox.showinfo("Not Available", "Window selection is only available on Windows.")
        return
    
    # Get list of potential ToS windows
    potential_windows = tos_data_grabber.list_potential_tos_windows()
    if not potential_windows:
        messagebox.showinfo("No ToS Windows Found", 
                          "Could not find running Thinkorswim windows.")
        return
    
    # Step 3: Show window selection dialog
    dialog = WindowSelectionDialog(root, f"Select ToS Window for {ticker_symbol}", potential_windows)
    selected_window_handle = dialog.result_handle
    
    if selected_window_handle is None:
        log_status_to_ui(f"Window selection cancelled for {ticker_symbol}.", ticker_symbol)
        return
    
    # Step 4: Rename window to ticker symbol
    try:
        rename_tos_window(selected_window_handle, ticker_symbol)
        log_status_to_ui(f"Renamed ToS window to '{ticker_symbol}'", ticker_symbol)
    except Exception as e:
        log_status_to_ui(f"Could not rename window: {e}", ticker_symbol, is_error=True)
    
    # Step 5: Window Linking
    app_data["tracked_tos_windows"][ticker_symbol] = selected_window_handle
    
    # Resize window to standard size
    log_status_to_ui(f"Resizing ToS window for {ticker_symbol}...", ticker_symbol)
    tos_data_grabber.resize_tos_window(selected_window_handle, log_status_to_ui, ticker_symbol)
    
    # Check if coordinates are set up
    if not app_data["coordinates_config"]:
        response = messagebox.askyesno("Setup Required", 
                                     "Coordinate setup is required for monitoring. Would you like to set it up now?")
        if response:
            wizard = CoordinateSetupWizard(root, selected_window_handle)
            root.wait_window(wizard)
            # Load saved coordinates
            load_coordinates_config()
        else:
            log_status_to_ui("Coordinate setup skipped. Monitoring will be limited.", is_error=True)
    
    # Update tab status
    active_tab_id = get_active_tab_id_path()
    if active_tab_id:
        widgets["main_notebook"].tab(active_tab_id, text=f"{ticker_symbol} - Linked")
    
    # Update UI buttons
    tab_widgets = tab_ui_widgets.get(ticker_symbol)
    if tab_widgets and "select_window_button" in tab_widgets:
        tab_widgets["select_window_button"].config(text="Change Window")
    
    log_status_to_ui(f"Successfully linked ToS window for {ticker_symbol}", ticker_symbol)
    
    # Step 6: Start monitoring if coordinates are configured
    if app_data["coordinates_config"]:
        response = messagebox.askyesno("Start Monitoring", 
                                     f"Would you like to start monitoring for {ticker_symbol}?")
        if response:
            handle_start_monitoring(ticker_symbol)

def handle_add_tab():
    """Modified tab creation workflow: Enter ticker first, then select window"""
    
    # Step 1: Get ticker input from user
    ticker_symbol = simpledialog.askstring(
        "Add Ticker", 
        "Enter Ticker Symbol (e.g., SPY):", 
        parent=root
    )
    
    if not ticker_symbol or not ticker_symbol.strip():
        return
    
    ticker_symbol = ticker_symbol.strip().upper()
    
    # Validate ticker format (1-5 uppercase letters)
    if not re.match(r'^[A-Z]{1,5}$', ticker_symbol):
        messagebox.showerror("Invalid Ticker", "Ticker must be 1-5 uppercase letters")
        return
    
    # Check if ticker already exists
    if ticker_symbol in tab_ui_widgets:
        messagebox.showinfo("Ticker Exists", f"A tab for {ticker_symbol} already exists.")
        return
    
    # Step 2: Create tab UI
    tab_callbacks = {
        'start_monitoring': lambda: handle_start_monitoring(ticker_symbol),
        'stop_monitoring': lambda: handle_stop_monitoring(ticker_symbol),
        'process_data': lambda: handle_process_data(ticker_symbol),
        'analyze_data': lambda: handle_run_analysis(ticker_symbol),
        'save_report': lambda: handle_save_report(ticker_symbol),
        'fetch_rtd': lambda: handle_fetch_rtd(ticker_symbol),
        'start_alpha_monitor': lambda: handle_start_alpha_monitoring(ticker_symbol),
        'export_signals': lambda: handle_export_alpha_signals(ticker_symbol),
        'select_window': lambda: handle_select_tos_window(ticker_symbol)  # New callback
    }
    
    tab_frame, specific_tab_widgets = ui_builder.create_ticker_tab_widgets(
        widgets["main_notebook"], ticker_symbol, tab_callbacks, log_status_to_ui)
    
    # Add tab without window association initially
    widgets["main_notebook"].add(tab_frame, text=f"{ticker_symbol} - No Window")
    tab_ui_widgets[ticker_symbol] = specific_tab_widgets
    
    # Initialize data structures
    app_data["cleaned_dfs"][ticker_symbol] = pd.DataFrame()
    app_data["raw_data_strings"][ticker_symbol] = ""
    app_data["analysis_outputs"][ticker_symbol] = {}
    app_data["options_for_rtd"][ticker_symbol] = []
    app_data["rtd_data_cache"][ticker_symbol] = pd.DataFrame()
    app_data["last_trade_timestamps"][ticker_symbol] = None
    app_data["tracked_tos_windows"][ticker_symbol] = None  # No window yet
    
    # Select the new tab
    widgets["main_notebook"].select(tab_frame)
    
    log_status_to_ui(f"Added new tab for ticker: {ticker_symbol}", active_ticker_for_log=ticker_symbol)
    ui_builder.apply_theme_to_widgets(root, style, current_theme, widgets)
    
    # Step 3: Prompt to select window
    if platform.system() == "Windows" and tos_data_grabber.PYWINAUTO_AVAILABLE:
        response = messagebox.askyesno(
            "Select ToS Window", 
            f"Would you like to select a ToS window to monitor for {ticker_symbol}?",
            parent=root
        )
        if response:
            handle_select_tos_window(ticker_symbol)


def handle_close_tab():
    """Close active tab"""
    current_active_ticker = get_active_ticker()
    if not current_active_ticker:
        log_status_to_ui("No tab selected to close.", is_error=True)
        return
    
    selected_tab_widget_path = get_active_tab_id_path()
    if not selected_tab_widget_path:
        return
    
    if messagebox.askyesno("Confirm Close Tab", f"Close tab for {current_active_ticker}?"):
        # Stop monitoring if active
        if app_data["monitoring_manager"]:
            app_data["monitoring_manager"].stop_monitoring(current_active_ticker)
        
        # Stop alpha monitoring
        if current_active_ticker in app_data["alpha_monitors"]:
            app_data["alpha_monitors"][current_active_ticker].stop_monitoring()
            del app_data["alpha_monitors"][current_active_ticker]
        
        # Remove tab
        widgets["main_notebook"].forget(selected_tab_widget_path)
        
        # Clean up data
        if current_active_ticker in tab_ui_widgets:
            del tab_ui_widgets[current_active_ticker]
        
        for data_key in ["cleaned_dfs", "raw_data_strings", "analysis_outputs", 
                        "options_for_rtd", "rtd_data_cache", "last_trade_timestamps", 
                        "tracked_tos_windows", "active_alpha_signals"]:
            if data_key in app_data and current_active_ticker in app_data[data_key]:
                del app_data[data_key][current_active_ticker]
        
        log_status_to_ui(f"Closed tab for ticker: {current_active_ticker}")

def handle_start_monitoring(ticker):
    """Start real-time monitoring for ticker"""
    if not app_data["monitoring_manager"]:
        app_data["monitoring_manager"] = MonitoringManager(log_status_to_ui)
    
    window_handle = app_data["tracked_tos_windows"].get(ticker)
    if not window_handle:
        messagebox.showerror("Error", f"No ToS window tracked for {ticker}")
        return
    
    if not app_data["coordinates_config"]:
        messagebox.showerror("Error", "Coordinates not configured. Please set up coordinates first.")
        return
    
    success = app_data["monitoring_manager"].start_monitoring(
        ticker, window_handle, app_data["coordinates_config"])
    
    if success:
        # Update UI
        tab_widgets = tab_ui_widgets.get(ticker)
        if tab_widgets:
            if "start_monitoring_button" in tab_widgets:
                tab_widgets["start_monitoring_button"].config(state=tk.DISABLED)
            if "stop_monitoring_button" in tab_widgets:
                tab_widgets["stop_monitoring_button"].config(state=tk.NORMAL)
            if "monitoring_status_label" in tab_widgets:
                tab_widgets["monitoring_status_label"].config(
                    text="Status: Monitoring Active", foreground="green")
        
        log_status_to_ui(f"Started monitoring for {ticker}", ticker)
    else:
        log_status_to_ui(f"Failed to start monitoring for {ticker}", ticker, is_error=True)

def handle_stop_monitoring(ticker):
    """Stop real-time monitoring for ticker"""
    if app_data["monitoring_manager"]:
        app_data["monitoring_manager"].stop_monitoring(ticker)
    
    # Update UI
    tab_widgets = tab_ui_widgets.get(ticker)
    if tab_widgets:
        if "start_monitoring_button" in tab_widgets:
            tab_widgets["start_monitoring_button"].config(state=tk.NORMAL)
        if "stop_monitoring_button" in tab_widgets:
            tab_widgets["stop_monitoring_button"].config(state=tk.DISABLED)
        if "monitoring_status_label" in tab_widgets:
            tab_widgets["monitoring_status_label"].config(
                text="Status: Monitoring Stopped", foreground="red")
    
    log_status_to_ui(f"Stopped monitoring for {ticker}", ticker)

def handle_process_data(ticker):
    """Process raw data for ticker"""
    tab_widgets = tab_ui_widgets.get(ticker)
    if not tab_widgets:
        return
    
    # Get data from monitoring or manual input
    raw_data = ""
    if app_data["monitoring_manager"] and ticker in app_data["monitoring_manager"].active_monitors:
        # Get from monitoring buffer
        monitor_data = app_data["monitoring_manager"].get_buffered_data(ticker)
        if monitor_data:
            raw_data = monitor_data
    else:
        # Get from input area
        raw_data_input = tab_widgets.get("raw_data_input_area")
        if raw_data_input:
            raw_data = raw_data_input.get("1.0", tk.END).strip()
    
    if not raw_data:
        messagebox.showinfo("Info", "No data to process")
        return
    
    log_status_to_ui(f"Processing data for {ticker}...", ticker)
    
    try:
        # Parse and clean data
        raw_df, parse_method = data_utils.parse_data_from_string(raw_data, log_status_to_ui)
        if raw_df.empty:
            messagebox.showerror("Error", f"Failed to parse data for {ticker}")
            return
        
        cleaned_df = data_utils.process_raw_options_data(raw_df, ticker, log_status_to_ui)
        if cleaned_df.empty:
            messagebox.showwarning("Warning", "No valid data after cleaning")
            return
        
        # Store cleaned data
        app_data["cleaned_dfs"][ticker] = cleaned_df
        app_data["raw_data_strings"][ticker] = raw_data
        
        # Update previews
        if "raw_data_preview_area" in tab_widgets:
            preview = tab_widgets["raw_data_preview_area"]
            preview.delete('1.0', tk.END)
            preview.insert(tk.END, f"--- Raw Data ({parse_method}) ---\n")
            preview.insert(tk.END, raw_df.head(20).to_string())
        
        if "cleaned_data_preview_area" in tab_widgets:
            preview = tab_widgets["cleaned_data_preview_area"]
            preview.delete('1.0', tk.END)
            preview.insert(tk.END, f"--- Cleaned Data (First 50 & Last 5) ---\n")
            preview.insert(tk.END, cleaned_df.head(50).to_string())
            preview.insert(tk.END, "\n\n")
            preview.insert(tk.END, cleaned_df.tail(5).to_string())
        
        # Enable analysis button
        if "analyze_data_button" in tab_widgets:
            tab_widgets["analyze_data_button"].config(state=tk.NORMAL)
        
        # Update tab status
        active_tab_id = get_active_tab_id_path()
        if active_tab_id:
            widgets["main_notebook"].tab(active_tab_id, text=f"{ticker} - Processed")
        
        log_status_to_ui(f"Data processed successfully for {ticker}", ticker)
        
    except Exception as e:
        messagebox.showerror("Processing Error", f"Error processing data: {e}")
        log_status_to_ui(f"Error processing data for {ticker}: {e}", ticker, is_error=True)

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

def analysis_thread_target(df_cleaned, ticker, log_func):
    """Analysis thread target function"""
    try:
        log_func(f"Analysis thread for {ticker} started...", ticker)
        analysis_output = analysis_engine.run_holistic_analysis(df_cleaned, ticker, log_func)
        analysis_results_queue.put((ticker, analysis_output))
    except Exception as e:
        log_func(f"Exception in analysis thread for {ticker}: {e}", ticker, is_error=True)
        analysis_results_queue.put((ticker, e))

def check_analysis_queue():
    """Check for completed analysis results"""
    try:
        ticker, result = analysis_results_queue.get_nowait()
        
        if isinstance(result, Exception):
            messagebox.showerror("Analysis Error", f"Error for {ticker}:\n{result}")
            log_status_to_ui(f"Analysis failed for {ticker}: {result}", ticker, is_error=True)
            
            # Re-enable button
            tab_widgets = tab_ui_widgets.get(ticker)
            if tab_widgets and "analyze_data_button" in tab_widgets:
                tab_widgets["analyze_data_button"].config(state=tk.NORMAL, text="Run Analysis")
        else:
            # Store results
            app_data["analysis_outputs"][ticker] = result
            
            # Update UI with results
            update_analysis_results_ui(ticker, result)
            
            # Extract options for RTD
            options_for_rtd = extract_options_for_rtd(result)
            app_data["options_for_rtd"][ticker] = options_for_rtd
            
            log_status_to_ui(f"Analysis complete for {ticker}", ticker)
    
    except queue.Empty:
        pass
    
    finally:
        root.after(100, check_analysis_queue)

def update_analysis_results_ui(ticker, results):
    """Update UI with analysis results"""
    tab_widgets = tab_ui_widgets.get(ticker)
    if not tab_widgets:
        return
    
    # Update trade briefing
    if "trade_briefing_area" in tab_widgets:
        briefing_area = tab_widgets["trade_briefing_area"]
        briefing_area.config(state=tk.NORMAL)
        briefing_area.delete('1.0', tk.END)
        
        # Generate and insert briefing
        from analysis_modules.report_generator import generate_trade_briefing
        briefing_text = generate_trade_briefing(results, ticker)
        briefing_area.insert(tk.END, briefing_text)
        briefing_area.config(state=tk.DISABLED)
    
    # Update alpha report
    if "alpha_report_area" in tab_widgets:
        alpha_area = tab_widgets["alpha_report_area"]
        alpha_area.config(state=tk.NORMAL)
        alpha_area.delete('1.0', tk.END)
        
        # Insert alpha report
        alpha_report = results.get('alpha_report', 'No alpha signals detected.')
        alpha_area.insert(tk.END, alpha_report)
        alpha_area.config(state=tk.DISABLED)
    
    # Enable buttons
    if "save_report_button" in tab_widgets:
        tab_widgets["save_report_button"].config(state=tk.NORMAL)
    if "fetch_rtd_button" in tab_widgets:
        tab_widgets["fetch_rtd_button"].config(state=tk.NORMAL)
    if "start_alpha_monitor_button" in tab_widgets:
        tab_widgets["start_alpha_monitor_button"].config(state=tk.NORMAL)
    if "analyze_data_button" in tab_widgets:
        tab_widgets["analyze_data_button"].config(state=tk.NORMAL, text="Run Analysis")
    
    # Update tab status
    active_tab_id = get_active_tab_id_path()
    if active_tab_id:
        widgets["main_notebook"].tab(active_tab_id, text=f"{ticker} - Analyzed")

def extract_options_for_rtd(results):
    """Extract top options for RTD monitoring"""
    options_details = results.get('options_of_interest_details', [])
    if not options_details:
        return []
    
    # Sort and get top options
    from analysis_modules.analysis_helpers import get_sort_key
    sorted_options = sorted(options_details, key=get_sort_key)
    
    # Extract symbols
    rtd_symbols = []
    seen = set()
    for opt in sorted_options:
        symbol = opt.get('standard_option_symbol')
        if symbol and symbol not in seen:
            rtd_symbols.append(symbol)
            seen.add(symbol)
            if len(rtd_symbols) >= config.MAX_OPTIONS_FOR_RTD:
                break
    
    return rtd_symbols

def handle_save_report(ticker):
    """Save detailed report for ticker"""
    analysis_output = app_data["analysis_outputs"].get(ticker)
    if not analysis_output:
        messagebox.showerror("Error", f"No analysis results for {ticker}")
        return
    
    # Generate report
    from analysis_modules.report_generator import generate_detailed_txt_report
    source_description = f"Options Flow Analysis for {ticker}"
    report_str = generate_detailed_txt_report(analysis_output, ticker, source_description)
    
    # Save dialog
    default_filename = f"{ticker}_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    save_path = filedialog.asksaveasfilename(
        initialdir=config.ANALYSIS_TXT_OUTPUT_DIRECTORY,
        initialfile=default_filename,
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )
    
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            log_status_to_ui(f"Report saved: {save_path}", ticker)
            messagebox.showinfo("Success", f"Report saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report: {e}")
            log_status_to_ui(f"Error saving report: {e}", ticker, is_error=True)

def handle_fetch_rtd(ticker):
    """Fetch RTD data for ticker"""
    options_symbols = app_data["options_for_rtd"].get(ticker)
    if not options_symbols:
        messagebox.showinfo("Info", f"No options to fetch RTD for {ticker}")
        return
    
    log_status_to_ui(f"Fetching RTD for {len(options_symbols)} options...", ticker)
    
    tab_widgets = tab_ui_widgets.get(ticker)
    rtd_results_widget = tab_widgets.get("rtd_results_area") if tab_widgets else None
    
    def on_rtd_complete(rtd_df):
        if isinstance(rtd_df, pd.DataFrame) and not rtd_df.empty:
            app_data["rtd_data_cache"][ticker] = rtd_df
            log_status_to_ui(f"RTD data fetched successfully for {ticker}", ticker)
        else:
            log_status_to_ui(f"RTD fetch failed for {ticker}", ticker, is_error=True)
    
    rtd_handler.fetch_and_display_rtd_data_for_tab(
        options_symbols, ticker, rtd_results_widget, log_status_to_ui, on_rtd_complete)

def handle_start_alpha_monitoring(ticker):
    """Start alpha monitoring for ticker"""
    cleaned_df = app_data["cleaned_dfs"].get(ticker)
    if cleaned_df is None or cleaned_df.empty:
        messagebox.showinfo("Info", "No data available. Process data first.")
        return
    
    # Create monitor if doesn't exist
    if ticker not in app_data["alpha_monitors"]:
        monitor = RealTimeAlphaMonitor(ticker, log_status_to_ui)
        app_data["alpha_monitors"][ticker] = monitor
        monitor.start_monitoring(cleaned_df)
        
        # Update UI
        tab_widgets = tab_ui_widgets.get(ticker)
        if tab_widgets and "alpha_monitor_status" in tab_widgets:
            tab_widgets["alpha_monitor_status"].config(
                text="Alpha Monitor: Active", foreground="green")
        
        log_status_to_ui(f"Started alpha monitoring for {ticker}", ticker)
    else:
        log_status_to_ui(f"Alpha monitoring already active for {ticker}", ticker)

def handle_export_alpha_signals(ticker):
    """Export alpha signals to file"""
    signals = app_data["active_alpha_signals"].get(ticker, [])
    if not signals:
        messagebox.showinfo("Info", f"No alpha signals to export for {ticker}")
        return
    
    # Create export data
    export_data = []
    for signal in signals:
        export_data.append({
            'timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'signal_type': signal.signal_type.value,
            'direction': signal.direction,
            'symbol': signal.option_symbol,
            'confidence': signal.confidence,
            'urgency': signal.urgency_score,
            'smart_money_score': signal.smart_money_score,
            'entry_price': signal.entry_price,
            'notional_value': signal.notional_value,
            'recommendation': signal.trade_recommendation
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(export_data)
    
    # Save dialog
    default_filename = f"{ticker}_Alpha_Signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_path = filedialog.asksaveasfilename(
        initialdir=config.ALPHA_SIGNALS_DIRECTORY,
        initialfile=default_filename,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if save_path:
        try:
            df.to_csv(save_path, index=False)
            log_status_to_ui(f"Alpha signals exported: {save_path}", ticker)
            messagebox.showinfo("Success", f"Signals exported to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export signals: {e}")

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
            
def check_alpha_signals():
    """Check for new alpha signals"""
    for ticker, monitor in app_data["alpha_monitors"].items():
        if monitor.is_monitoring:
            new_signals = monitor.get_active_signals()
            
            if new_signals:
                # Store signals
                if ticker not in app_data["active_alpha_signals"]:
                    app_data["active_alpha_signals"][ticker] = []
                app_data["active_alpha_signals"][ticker].extend(new_signals)
                
                # Update UI
                update_alpha_signals_display(ticker, new_signals)
                
                # Send alerts
                if app_data["alert_manager"]:
                    for signal in new_signals:
                        app_data["alert_manager"].process_signal(signal)
    
    root.after(500, check_alpha_signals)

def update_alpha_signals_display(ticker, new_signals):
    """Update alpha signals display"""
    tab_widgets = tab_ui_widgets.get(ticker)
    if not tab_widgets or "alpha_signals_display" not in tab_widgets:
        return
    
    signals_display = tab_widgets["alpha_signals_display"]
    
    for signal in new_signals:
        # Format signal
        timestamp = signal.timestamp.strftime("%H:%M:%S")
        
        # Determine color tag
        if signal.urgency_score >= alpha_config.ALPHA_URGENT_SIGNAL_THRESHOLD:
            tag = "urgent"
        elif signal.direction == "BULLISH":
            tag = "bullish"
        elif signal.direction == "BEARISH":
            tag = "bearish"
        else:
            tag = "neutral"
        
        # Insert signal
        signal_text = (
            f"[{timestamp}] {signal.signal_type.value} - {signal.direction}\n"
            f"  {signal.option_symbol} @ ${signal.entry_price:.2f} | "
            f"Size: ${signal.notional_value:,.0f}\n"
            f"  Confidence: {signal.confidence:.0f}% | Smart Money: {signal.smart_money_score:.0f}\n"
            f"  📊 {signal.trade_recommendation}\n"
            f"  {'─' * 70}\n"
        )
        
        signals_display.config(state=tk.NORMAL)
        signals_display.insert(tk.END, signal_text, tag)
        signals_display.see(tk.END)
        signals_display.config(state=tk.DISABLED)

def load_coordinates_config():
    """Load saved coordinates configuration"""
    try:
        if os.path.exists(config.COORDINATES_CONFIG_FILE):
            with open(config.COORDINATES_CONFIG_FILE, 'r') as f:
                app_data["coordinates_config"] = json.load(f)
                log_status_to_ui("Loaded coordinates configuration", also_to_analysis_area=False)
    except Exception as e:
        log_status_to_ui(f"Error loading coordinates: {e}", is_error=True, also_to_analysis_area=False)
def handle_fetch_rtd(ticker):
    """Enhanced RTD fetch with signal-based symbol selection"""
    analysis_output = app_data["analysis_outputs"].get(ticker)
    if not analysis_output:
        messagebox.showinfo("Info", f"No analysis results for {ticker}. Run analysis first.")
        return
    
    df_cleaned = app_data["cleaned_dfs"].get(ticker)
    if df_cleaned is None or df_cleaned.empty:
        messagebox.showinfo("Info", f"No processed data for {ticker}")
        return
    
    log_status_to_ui(f"Starting RTD fetch with signal analysis for {ticker}...", ticker)
    
    try:
        # Create temporary RTD handler to identify signal-based symbols
        temp_rtd_handler = RTDDataHandler(log_status_to_ui)
        signal_symbols = temp_rtd_handler.identify_signal_based_symbols(df_cleaned, analysis_output)
        
        if not signal_symbols:
            log_status_to_ui("No symbols with signals found for RTD monitoring", ticker, is_error=True)
            messagebox.showinfo("Info", f"No symbols with signals identified for RTD monitoring")
            return
        
        log_status_to_ui(f"Identified {len(signal_symbols)} symbols with signals for RTD", ticker)
        
        # Get RTD display widget
        tab_widgets = tab_ui_widgets.get(ticker)
        if not tab_widgets:
            return
        
        rtd_results_widget = tab_widgets.get("rtd_results_area")
        
        def on_rtd_complete(rtd_df):
            if isinstance(rtd_df, pd.DataFrame) and not rtd_df.empty:
                app_data["rtd_data_cache"][ticker] = rtd_df
                log_status_to_ui(f"RTD data fetched successfully for {len(rtd_df)} symbols", ticker)
                
                # Start comprehensive RTD integration if not already running
                if ticker not in app_data["rtd_integrators"]:
                    start_rtd_integration(ticker, analysis_output, df_cleaned)
                
            else:
                log_status_to_ui(f"RTD fetch failed for {ticker}", ticker, is_error=True)
        
        # Start RTD fetch
        fetch_and_display_rtd_data_for_tab(
            signal_symbols, ticker, rtd_results_widget, log_status_to_ui, on_rtd_complete
        )
        
        # Update UI to show RTD is active
        if "rtd_status_label" in tab_widgets:
            tab_widgets["rtd_status_label"].config(
                text="RTD Status: Fetching...", foreground="orange"
            )
        
    except Exception as e:
        log_status_to_ui(f"Error in RTD fetch: {e}", ticker, is_error=True)
        messagebox.showerror("RTD Error", f"Failed to start RTD fetch: {e}")


def start_rtd_integration(ticker, analysis_results, cleaned_df):
    """Start comprehensive RTD integration for ticker"""
    
    if ticker in app_data["rtd_integrators"]:
        log_status_to_ui(f"RTD integration already active for {ticker}", ticker)
        return
    
    log_status_to_ui(f"Starting RTD integration for {ticker}...", ticker)
    
    try:
        # Set up comprehensive RTD integration
        integrator = setup_comprehensive_rtd_integration(
            ticker, analysis_results, cleaned_df, log_status_to_ui
        )
        
        if integrator:
            app_data["rtd_integrators"][ticker] = integrator
            
            # Add signal callback to update UI
            def signal_callback(enhanced_signal):
                handle_enhanced_rtd_signal(ticker, enhanced_signal)
            
            integrator.add_signal_callback(signal_callback)
            
            # Update UI
            tab_widgets = tab_ui_widgets.get(ticker)
            if tab_widgets:
                if "rtd_status_label" in tab_widgets:
                    tab_widgets["rtd_status_label"].config(
                        text="RTD Status: Integrated & Monitoring", foreground="green"
                    )
                
                if "stop_rtd_button" in tab_widgets:
                    tab_widgets["stop_rtd_button"].config(state=tk.NORMAL)
            
            log_status_to_ui(f"RTD integration started successfully for {ticker}", ticker)
        else:
            log_status_to_ui(f"Failed to start RTD integration for {ticker}", ticker, is_error=True)
    
    except Exception as e:
        log_status_to_ui(f"Error starting RTD integration: {e}", ticker, is_error=True)


def stop_rtd_integration(ticker):
    """Stop RTD integration for ticker"""
    
    integrator = app_data["rtd_integrators"].get(ticker)
    if integrator:
        integrator.cleanup()
        del app_data["rtd_integrators"][ticker]
        
        # Update UI
        tab_widgets = tab_ui_widgets.get(ticker)
        if tab_widgets:
            if "rtd_status_label" in tab_widgets:
                tab_widgets["rtd_status_label"].config(
                    text="RTD Status: Stopped", foreground="red"
                )
            
            if "stop_rtd_button" in tab_widgets:
                tab_widgets["stop_rtd_button"].config(state=tk.DISABLED)
        
        log_status_to_ui(f"RTD integration stopped for {ticker}", ticker)
    else:
        log_status_to_ui(f"No RTD integration active for {ticker}", ticker)


def handle_enhanced_rtd_signal(ticker, enhanced_signal):
    """Handle enhanced RTD signal"""
    
    try:
        # Store signal
        if ticker not in app_data["rtd_active_signals"]:
            app_data["rtd_active_signals"][ticker] = []
        
        app_data["rtd_active_signals"][ticker].append(enhanced_signal)
        
        # Update alpha signals display
        tab_widgets = tab_ui_widgets.get(ticker)
        if tab_widgets and "alpha_signals_display" in tab_widgets:
            update_alpha_signals_display(ticker, [enhanced_signal])
        
        # Log significant signals
        if enhanced_signal.urgency_score >= alpha_config.ALPHA_URGENT_SIGNAL_THRESHOLD:
            log_status_to_ui(
                f"🚨 URGENT RTD Signal: {enhanced_signal.trade_recommendation}", 
                ticker
            )
        
        # Show alert banner for critical signals
        if enhanced_signal.urgency_score >= 90:
            show_rtd_alert_banner(ticker, enhanced_signal)
    
    except Exception as e:
        log_status_to_ui(f"Error handling enhanced RTD signal: {e}", ticker, is_error=True)


def show_rtd_alert_banner(ticker, signal):
    """Show alert banner for critical RTD signals"""
    
    try:
        # Create alert message
        alert_msg = f"CRITICAL RTD SIGNAL: {signal.option_symbol} - {signal.direction} | {signal.trade_recommendation}"
        
        # Show alert popup
        def show_alert():
            messagebox.showwarning(
                f"Critical RTD Signal - {ticker}",
                f"Symbol: {signal.option_symbol}\n"
                f"Type: {signal.signal_type.value}\n"
                f"Direction: {signal.direction}\n"
                f"Confidence: {signal.confidence:.1f}%\n"
                f"Urgency: {signal.urgency_score:.1f}\n\n"
                f"Recommendation:\n{signal.trade_recommendation}",
                parent=root
            )
        
        # Show alert in separate thread to avoid blocking
        alert_thread = threading.Thread(target=show_alert, daemon=True)
        alert_thread.start()
        
    except Exception as e:
        log_status_to_ui(f"Error showing RTD alert: {e}", ticker, is_error=True)


def handle_view_rtd_integration_report(ticker):
    """View RTD integration report"""
    
    integrator = app_data["rtd_integrators"].get(ticker)
    if not integrator:
        messagebox.showinfo("Info", f"No RTD integration active for {ticker}")
        return
    
    try:
        # Generate report
        report = generate_rtd_integration_report(integrator)
        
        # Show report in new window
        show_rtd_report_window(ticker, report)
        
    except Exception as e:
        log_status_to_ui(f"Error generating RTD report: {e}", ticker, is_error=True)
        messagebox.showerror("Error", f"Failed to generate RTD report: {e}")


def show_rtd_report_window(ticker, report):
    """Show RTD report in new window"""
    
    report_window = tk.Toplevel(root)
    report_window.title(f"RTD Integration Report - {ticker}")
    report_window.geometry("800x600")
    report_window.transient(root)
    
    # Create scrolled text widget
    text_frame = ttk.Frame(report_window, padding="10")
    text_frame.pack(fill=tk.BOTH, expand=True)
    
    text_widget = scrolledtext.ScrolledText(
        text_frame, 
        wrap=tk.WORD, 
        font=('Consolas', 10),
        state=tk.NORMAL
    )
    text_widget.pack(fill=tk.BOTH, expand=True)
    
    # Insert report
    text_widget.insert(tk.END, report)
    text_widget.config(state=tk.DISABLED)
    
    # Add close button
    button_frame = ttk.Frame(report_window, padding="10")
    button_frame.pack(fill=tk.X)
    
    ttk.Button(
        button_frame, 
        text="Close", 
        command=report_window.destroy
    ).pack(side=tk.RIGHT)
    
    # Auto-refresh button
    def refresh_report():
        try:
            integrator = app_data["rtd_integrators"].get(ticker)
            if integrator:
                new_report = generate_rtd_integration_report(integrator)
                text_widget.config(state=tk.NORMAL)
                text_widget.delete(1.0, tk.END)
                text_widget.insert(tk.END, new_report)
                text_widget.config(state=tk.DISABLED)
        except Exception as e:
            log_status_to_ui(f"Error refreshing RTD report: {e}", ticker, is_error=True)
    
    ttk.Button(
        button_frame, 
        text="Refresh", 
        command=refresh_report
    ).pack(side=tk.RIGHT, padx=(0, 5))


def handle_export_rtd_signals(ticker):
    """Export RTD signals to file"""
    
    rtd_signals = app_data["rtd_active_signals"].get(ticker, [])
    if not rtd_signals:
        messagebox.showinfo("Info", f"No RTD signals to export for {ticker}")
        return
    
    try:
        # Create export data
        export_data = []
        for signal in rtd_signals:
            export_data.append({
                'timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'signal_type': signal.signal_type.value,
                'direction': signal.direction,
                'symbol': signal.option_symbol,
                'confidence': signal.confidence,
                'urgency': signal.urgency_score,
                'smart_money_score': signal.smart_money_score,
                'entry_price': signal.entry_price,
                'target_price': signal.target_price,
                'stop_price': signal.stop_price,
                'notional_value': signal.notional_value,
                'recommendation': signal.trade_recommendation,
                'source': signal.metadata.get('source', 'Unknown')
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(export_data)
        
        # Save dialog
        default_filename = f"{ticker}_RTD_Signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        save_path = filedialog.asksaveasfilename(
            initialdir=config.ALPHA_SIGNALS_DIRECTORY,
            initialfile=default_filename,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if save_path:
            df.to_csv(save_path, index=False)
            log_status_to_ui(f"RTD signals exported: {save_path}", ticker)
            messagebox.showinfo("Success", f"RTD signals exported to:\n{save_path}")
    
    except Exception as e:
        log_status_to_ui(f"Error exporting RTD signals: {e}", ticker, is_error=True)
        messagebox.showerror("Export Error", f"Failed to export RTD signals: {e}")


# Update the tab creation function to include RTD controls
def _build_rtd_enhanced_tab(parent, ticker, callbacks, tab_widgets):
    """Build enhanced RTD tab with integration controls"""
    
    # RTD Control Frame
    rtd_control_frame = ttk.LabelFrame(parent, text="RTD Integration", padding="10")
    rtd_control_frame.pack(fill=tk.X, pady=(0, 10))
    
    # RTD Status
    rtd_status = ttk.Label(rtd_control_frame, text="RTD Status: Not Connected", foreground="red")
    rtd_status.pack(anchor=tk.W)
    tab_widgets["rtd_status_label"] = rtd_status
    
    # RTD Control Buttons
    rtd_buttons_frame = ttk.Frame(rtd_control_frame)
    rtd_buttons_frame.pack(fill=tk.X, pady=(5, 0))
    
    # Fetch RTD button
    fetch_rtd_btn = ttk.Button(
        rtd_buttons_frame,
        text="Fetch RTD Data",
        command=callbacks.get('fetch_rtd'),
        state=tk.DISABLED
    )
    fetch_rtd_btn.pack(side=tk.LEFT, padx=(0, 5))
    tab_widgets["fetch_rtd_button"] = fetch_rtd_btn
    
    # Stop RTD button
    stop_rtd_btn = ttk.Button(
        rtd_buttons_frame,
        text="Stop RTD Integration",
        command=lambda: stop_rtd_integration(ticker),
        state=tk.DISABLED
    )
    stop_rtd_btn.pack(side=tk.LEFT, padx=(0, 5))
    tab_widgets["stop_rtd_button"] = stop_rtd_btn
    
    # View report button
    view_report_btn = ttk.Button(
        rtd_buttons_frame,
        text="View RTD Report",
        command=lambda: handle_view_rtd_integration_report(ticker)
    )
    view_report_btn.pack(side=tk.LEFT, padx=(0, 5))
    tab_widgets["view_rtd_report_button"] = view_report_btn
    
    # Export RTD signals button
    export_rtd_btn = ttk.Button(
        rtd_buttons_frame,
        text="Export RTD Signals",
        command=lambda: handle_export_rtd_signals(ticker)
    )
    export_rtd_btn.pack(side=tk.RIGHT)
    tab_widgets["export_rtd_signals_button"] = export_rtd_btn
    
    # RTD Results Display
    rtd_results_frame = ttk.LabelFrame(parent, text="Real-Time Data Results", padding="5")
    rtd_results_frame.pack(fill=tk.BOTH, expand=True)
    
    rtd_results = scrolledtext.ScrolledText(rtd_results_frame, state=tk.DISABLED)
    rtd_results.pack(fill=tk.BOTH, expand=True)
    tab_widgets["rtd_results_area"] = rtd_results


# Update the close tab function to clean up RTD
def handle_close_tab_enhanced():
    """Enhanced close tab with RTD cleanup"""
    current_active_ticker = get_active_ticker()
    if not current_active_ticker:
        log_status_to_ui("No tab selected to close.", is_error=True)
        return
    
    selected_tab_widget_path = get_active_tab_id_path()
    if not selected_tab_widget_path:
        return
    
    if messagebox.askyesno("Confirm Close Tab", f"Close tab for {current_active_ticker}?"):
        # Stop RTD integration
        stop_rtd_integration(current_active_ticker)
        
        # Stop monitoring if active
        if app_data["monitoring_manager"]:
            app_data["monitoring_manager"].stop_monitoring(current_active_ticker)
        
        # Stop alpha monitoring
        if current_active_ticker in app_data["alpha_monitors"]:
            app_data["alpha_monitors"][current_active_ticker].stop_monitoring()
            del app_data["alpha_monitors"][current_active_ticker]
        
        # Remove tab
        widgets["main_notebook"].forget(selected_tab_widget_path)
        
        # Clean up all data including RTD
        cleanup_data_keys = [
            "cleaned_dfs", "raw_data_strings", "analysis_outputs", 
            "options_for_rtd", "rtd_data_cache", "last_trade_timestamps", 
            "tracked_tos_windows", "active_alpha_signals", "rtd_handlers",
            "rtd_integrators", "rtd_active_signals"
        ]
        
        for data_key in cleanup_data_keys:
            if data_key in app_data and current_active_ticker in app_data[data_key]:
                del app_data[data_key][current_active_ticker]
        
        # Clean up UI widgets
        if current_active_ticker in tab_ui_widgets:
            del tab_ui_widgets[current_active_ticker]
        
        log_status_to_ui(f"Closed tab and cleaned up all resources for ticker: {current_active_ticker}")


# Add RTD status monitoring to the main loop
def check_rtd_status():
    """Check RTD integration status for all active tickers"""
    
    for ticker, integrator in app_data["rtd_integrators"].items():
        if integrator and integrator.is_integrated:
            try:
                # Get recent performance metrics
                metrics = integrator.get_performance_metrics()
                
                # Update UI if significant activity
                if metrics.get('urgent_signals_last_hour', 0) > 0:
                    tab_widgets = tab_ui_widgets.get(ticker)
                    if tab_widgets and "rtd_status_label" in tab_widgets:
                        urgent_count = metrics['urgent_signals_last_hour']
                        tab_widgets["rtd_status_label"].config(
                            text=f"RTD Status: Active ({urgent_count} urgent signals/hour)",
                            foreground="orange"
                        )
                
                # Log periodic stats (every 10 minutes)
                if datetime.now().minute % 10 == 0 and datetime.now().second < 5:
                    summary = integrator.get_integration_summary()
                    recent_signals = summary.get('recent_enhanced_signals', 0)
                    if recent_signals > 0:
                        log_status_to_ui(
                            f"RTD Integration Stats: {recent_signals} recent signals", 
                            ticker
                        )
            
            except Exception as e:
                log_status_to_ui(f"Error checking RTD status for {ticker}: {e}", ticker, is_error=True)
    
    # Schedule next check
    root.after(30000, check_rtd_status)  # Check every 30 seconds


# Update the main function to include RTD status monitoring
def main_with_rtd():
    """Enhanced main function with RTD integration"""
    
    # ... existing main() code ...
    
    # Start background tasks (add RTD status monitoring)
    root.after(100, check_analysis_queue)
    root.after(500, check_alpha_signals)
    root.after(1000, check_monitoring_data)
    root.after(30000, check_rtd_status)  # Add RTD status checking
    
    log_status_to_ui("Application initialized with RTD integration. Add ticker tab to begin.", also_to_analysis_area=False)
    
    # Start main loop
    root.mainloop()


# Add menu item for RTD management
def add_rtd_menu_items():
    """Add RTD-specific menu items"""
    
    # Tools menu additions
    toolsmenu = widgets.get("menubar").children.get("!menu2")  # Tools menu
    if toolsmenu:
        toolsmenu.add_separator()
        toolsmenu.add_command(label="RTD Integration Status", command=show_global_rtd_status)
        toolsmenu.add_command(label="Stop All RTD", command=stop_all_rtd_integrations)


def show_global_rtd_status():
    """Show global RTD status for all tickers"""
    
    status_report = "RTD INTEGRATION GLOBAL STATUS\n"
    status_report += "=" * 50 + "\n\n"
    
    active_integrators = app_data["rtd_integrators"]
    
    if not active_integrators:
        status_report += "No RTD integrations currently active.\n"
    else:
        for ticker, integrator in active_integrators.items():
            summary = integrator.get_integration_summary()
            metrics = integrator.get_performance_metrics()
            
            status_report += f"Ticker: {ticker}\n"
            status_report += f"  Status: {'ACTIVE' if summary['is_integrated'] else 'INACTIVE'}\n"
            status_report += f"  Enhanced Signals: {summary['enhanced_signals_total']}\n"
            status_report += f"  Recent Signals (15min): {summary['recent_enhanced_signals']}\n"
            status_report += f"  Signals/Hour: {metrics.get('enhanced_signals_per_hour', 0):.1f}\n"
            status_report += f"  Avg Confidence: {metrics.get('avg_confidence', 0):.1f}%\n\n"
    
    # Show in message box
    messagebox.showinfo("RTD Integration Status", status_report)


def stop_all_rtd_integrations():
    """Stop all RTD integrations"""
    
    if not app_data["rtd_integrators"]:
        messagebox.showinfo("Info", "No RTD integrations currently active.")
        return
    
    if messagebox.askyesno("Confirm", "Stop all RTD integrations?"):
        tickers_to_stop = list(app_data["rtd_integrators"].keys())
        
        for ticker in tickers_to_stop:
            stop_rtd_integration(ticker)
        
        log_status_to_ui(f"Stopped RTD integration for {len(tickers_to_stop)} tickers")
        messagebox.showinfo("Success", f"Stopped RTD integration for {len(tickers_to_stop)} tickers")


# Example usage in updated tab creation
def create_enhanced_ticker_tab_with_rtd(parent_notebook, ticker, callbacks, log_func):
    """Create enhanced ticker tab with RTD integration"""
    
    # Create main tab frame
    tab_frame = ttk.Frame(parent_notebook, padding="5")
    
    # dictionary to store all widgets for this tab
    tab_widgets = {}
    
    # Create main paned window (horizontal split)
    main_paned = ttk.PanedWindow(tab_frame, orient=tk.HORIZONTAL)
    main_paned.pack(fill=tk.BOTH, expand=True)
    
    # Left panel (controls and data input)
    left_panel = ttk.Frame(main_paned, padding="5")
    main_paned.add(left_panel, weight=1)
    
    # Right panel (analysis results with RTD)
    right_panel = ttk.Frame(main_paned, padding="5")
    main_paned.add(right_panel, weight=2)
    
    # Build panels with RTD enhancements
    _build_left_panel(left_panel, ticker, callbacks, tab_widgets, log_func)
    _build_right_panel_with_rtd(right_panel, ticker, callbacks, tab_widgets, log_func)
    
    return tab_frame, tab_widgets


def _build_right_panel_with_rtd(parent, ticker, callbacks, tab_widgets, log_func):
    """Build right panel with RTD integration"""
    
    # Create notebook for analysis results
    results_notebook = ttk.Notebook(parent)
    results_notebook.pack(fill=tk.BOTH, expand=True)
    
    # Alpha Signals Tab (enhanced with RTD)
    alpha_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(alpha_tab, text="Alpha Signals")
    _build_alpha_signals_tab(alpha_tab, callbacks, tab_widgets)
    
    # RTD Integration Tab (new)
    rtd_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(rtd_tab, text="RTD Integration")
    _build_rtd_enhanced_tab(rtd_tab, ticker, callbacks, tab_widgets)
    
    # Trade Briefing Tab
    briefing_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(briefing_tab, text="Trade Briefing")
    _build_trade_briefing_tab(briefing_tab, tab_widgets)
    
    # Data Preview Tab
    preview_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(preview_tab, text="Data Preview")
    _build_data_preview_tab(preview_tab, tab_widgets)
    
    # Analysis Log Tab
    log_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(log_tab, text="Analysis Log")
    _build_analysis_log_tab(log_tab, tab_widgets)
    
class WindowSelectionDialog(simpledialog.Dialog):
    """Dialog for selecting ToS window"""
    def __init__(self, parent, title, windows_list):
        self.windows_list = windows_list
        self.result_handle = None
        self.selected_window_title = tk.StringVar()
        super().__init__(parent, title)
    
    def body(self, master):
        ttk.Label(master, text="Select ToS window to monitor:").pack(pady=5)
        
        self.listbox = tk.listbox(master, width=80, height=10, exportselection=False)
        self.listbox.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        for title, handle in self.windows_list:
            self.listbox.insert(tk.END, f"{title} (Handle: {handle})")
        
        if self.windows_list:
            self.listbox.selection_set(0)
            self.listbox.see(0)
        
        self.listbox.bind("<<listboxSelect>>", self.on_select)
        self.listbox.bind("<Double-Button-1>", self.ok)
        
        return self.listbox
    
    def on_select(self, event):
        if event.widget.curselection():
            self.selected_window_title.set(event.widget.get(event.widget.curselection()[0]))
    
    def apply(self):
        if self.listbox.curselection():
            self.result_handle = self.windows_list[self.listbox.curselection()[0]][1]
        else:
            self.result_handle = None

def main():
    """Main application entry point"""
    global root, style, widgets, main_notebook
    
    # Create root window
    root = tk.Tk()
    root.title("Options Flow & RTD Analyzer v3.0 - Alpha Edition")
    root.geometry("1600x900")
    
    # Create style
    style = ttk.Style(root)
    
    # Create menu bar
    menubar = tk.Menu(root)
    
    # File menu
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Add Ticker Tab", command=handle_add_tab)
    filemenu.add_command(label="Close Active Tab", command=handle_close_tab)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=root.quit)
    menubar.add_cascade(label="File", menu=filemenu)
    
    # Tools menu
    toolsmenu = tk.Menu(menubar, tearoff=0)
    toolsmenu.add_command(label="Settings", command=handle_open_settings_dialog)
    toolsmenu.add_command(label="Toggle Theme", command=handle_toggle_theme)
    menubar.add_cascade(label="Tools", menu=toolsmenu)
    
    # Help menu
    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="About", command=lambda: messagebox.showinfo(
        "About", "Options Flow & RTD Analyzer v3.0\nAlpha Extraction Edition\n\nBy: Advanced Trading Systems"))
    menubar.add_cascade(label="Help", menu=helpmenu)
    
    root.config(menu=menubar)
    widgets["menubar"] = menubar
    
    # Create top control frame
    top_control_frame = ttk.Frame(root, padding=(5, 5))
    top_control_frame.pack(side=tk.TOP, fill=tk.X)
    widgets["top_control_frame"] = top_control_frame
    
    # Control buttons
    add_tab_button = ttk.Button(top_control_frame, text="Add Ticker Tab", command=handle_add_tab)
    add_tab_button.pack(side=tk.LEFT, padx=5)
    widgets["add_tab_button"] = add_tab_button
    
    close_tab_button = ttk.Button(top_control_frame, text="Close Tab", command=handle_close_tab)
    close_tab_button.pack(side=tk.LEFT, padx=5)
    widgets["close_tab_button"] = close_tab_button
    
    theme_toggle_button = ttk.Button(top_control_frame, text="Toggle Theme", command=handle_toggle_theme)
    theme_toggle_button.pack(side=tk.RIGHT, padx=5)
    widgets["theme_toggle_button"] = theme_toggle_button
    
    # Create main notebook
    main_notebook = ttk.Notebook(root)
    main_notebook.pack(expand=True, fill='both', padx=5, pady=(0, 5))
    widgets["main_notebook"] = main_notebook
    
    # Create status bar
    status_label = ttk.Label(root, text="Ready.", relief=tk.SUNKEN, anchor='w', padding=(3, 3))
    status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
    widgets["status_label"] = status_label
    
    # Apply initial theme
    ui_builder.apply_theme_to_widgets(root, style, current_theme, widgets)
    
    # Initialize managers
    app_data["data_fusion_engine"] = DataFusionEngine()
    app_data["alert_manager"] = AlertManager(log_status_to_ui)
    
    # Load saved coordinates
    load_coordinates_config()
    
    # Start background tasks
    root.after(100, check_analysis_queue)
    root.after(500, check_alpha_signals)
    root.after(1000, check_monitoring_data)
    
    log_status_to_ui("Application initialized. Add ticker tab to begin.", also_to_analysis_area=False)
    
    # Start main loop
    root.mainloop()

if __name__ == '__main__':
    main()
    print("Application closed.")