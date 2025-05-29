# ui_builder.py
"""
User interface builder for the Options Flow Analyzer
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, font
from typing import List, Tuple, Optional, Dict, Callable, Any
import config

class ModernScrolledText(scrolledtext.ScrolledText):
    """Enhanced ScrolledText with modern styling"""
    
    def __init__(self, parent, **kwargs):
        # Set default styling
        default_kwargs = {
            'wrap': tk.WORD,
            'padx': 10,
            'pady': 5,
            'font': ('Consolas', 10),
            'relief': tk.FLAT,
            'borderwidth': 1
        }
        default_kwargs.update(kwargs)
        
        super().__init__(parent, **default_kwargs)
        
        # Configure text tags for colored output
        self.tag_configure("urgent", foreground="red", font=('Consolas', 10, 'bold'))
        self.tag_configure("bullish", foreground="green")
        self.tag_configure("bearish", foreground="red")
        self.tag_configure("neutral", foreground="blue")
        self.tag_configure("header", font=('Consolas', 11, 'bold'))

def apply_theme_to_widgets(root: tk.Tk, style: ttk.Style, theme_name: str, widgets: dict):
    """Apply theme to all widgets"""
    
    theme_config = config.THEMES.get(theme_name, config.THEMES['light'])
    
    # Configure root window
    root.configure(bg=theme_config["bg"])
    
    # Configure ttk style
    style.theme_use('clam')
    
    # Configure specific widget styles
    style.configure('TFrame', background=theme_config["bg"])
    style.configure('TLabel', background=theme_config["bg"], foreground=theme_config["fg"])
    style.configure('TButton', fieldbackground=theme_config["button_bg"], foreground=theme_config["button_fg"])
    style.configure('TNotebook', background=theme_config["bg"], fieldbackground=theme_config["bg"])
    style.configure('TNotebook.Tab', padding=(20, 8), font=('Arial', 9))
    
    # Configure text widgets
    for widget_name, widget in widgets.items():
        if widget and hasattr(widget, 'winfo_exists') and widget.winfo_exists():
            try:
                if isinstance(widget, (tk.Text, scrolledtext.ScrolledText, ModernScrolledText)):
                    widget.configure(
                        bg=theme_config["text_bg"],
                        fg=theme_config["text_fg"],
                        insertbackground=theme_config["text_fg"],
                        selectbackground=theme_config["select_bg"]
                    )
                elif isinstance(widget, tk.Label):
                    widget.configure(
                        bg=theme_config["bg"],
                        fg=theme_config["fg"]
                    )
            except tk.TclError:
                # Widget might be destroyed
                pass

def create_ticker_tab_widgets(parent_notebook, ticker, callbacks, log_func):
    """Create enhanced tab with window selection capability"""
    
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
    
    # Right panel (analysis results)
    right_panel = ttk.Frame(main_paned, padding="5")
    main_paned.add(right_panel, weight=2)
    
    # Build panels
    _build_left_panel_enhanced(left_panel, ticker, callbacks, tab_widgets, log_func)
    _build_right_panel(right_panel, ticker, callbacks, tab_widgets, log_func)
    
    return tab_frame, tab_widgets

def _build_left_panel_enhanced(parent, ticker, callbacks, tab_widgets, log_func):
    """Build enhanced left panel with window selection"""
    
    # Window Control Frame (NEW)
    window_frame = ttk.LabelFrame(parent, text="ToS Window Control", padding="10")
    window_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Window status label
    window_status = ttk.Label(window_frame, text="Window: Not Linked", foreground="red")
    window_status.pack(anchor=tk.W)
    tab_widgets["window_status_label"] = window_status
    
    # Select window button
    select_window_btn = ttk.Button(
        window_frame,
        text="Select ToS Window",
        command=callbacks.get('select_window')
    )
    select_window_btn.pack(pady=(5, 0), fill=tk.X)
    tab_widgets["select_window_button"] = select_window_btn
    
    # Monitoring Control Frame
    monitoring_frame = ttk.LabelFrame(parent, text="Data Monitoring", padding="10")
    monitoring_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Monitoring status
    monitoring_status = ttk.Label(monitoring_frame, text="Status: Not Active", foreground="red")
    monitoring_status.pack(anchor=tk.W)
    tab_widgets["monitoring_status_label"] = monitoring_status
    
    # Monitoring buttons
    button_frame = ttk.Frame(monitoring_frame)
    button_frame.pack(fill=tk.X, pady=(5, 0))
    
    start_btn = ttk.Button(
        button_frame,
        text="Start Monitoring",
        command=callbacks.get('start_monitoring'),
        state=tk.DISABLED  # Disabled until window is selected
    )
    start_btn.pack(side=tk.LEFT, padx=(0, 5))
    tab_widgets["start_monitoring_button"] = start_btn
    
    stop_btn = ttk.Button(
        button_frame,
        text="Stop Monitoring",
        command=callbacks.get('stop_monitoring'),
        state=tk.DISABLED
    )
    stop_btn.pack(side=tk.LEFT)
    tab_widgets["stop_monitoring_button"] = stop_btn
    
    # Monitoring statistics
    stats_frame = ttk.Frame(monitoring_frame)
    stats_frame.pack(fill=tk.X, pady=(10, 0))
    
    trades_label = ttk.Label(stats_frame, text="Trades Today: 0")
    trades_label.pack(anchor=tk.W)
    tab_widgets["trades_today_label"] = trades_label
    
    last_trade_label = ttk.Label(stats_frame, text="Last Trade: --:--:--")
    last_trade_label.pack(anchor=tk.W)
    tab_widgets["last_trade_label"] = last_trade_label
    
    # Manual Data Input Frame
    input_frame = ttk.LabelFrame(parent, text="Manual Data Input", padding="10")
    input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    # Raw data input area
    raw_data_input = scrolledtext.ScrolledText(input_frame, height=10, wrap=tk.WORD)
    raw_data_input.pack(fill=tk.BOTH, expand=True)
    tab_widgets["raw_data_input_area"] = raw_data_input
    
    # Process button
    process_btn = ttk.Button(
        input_frame,
        text="Process Data",
        command=callbacks.get('process_data')
    )
    process_btn.pack(pady=(5, 0))
    tab_widgets["process_data_button"] = process_btn
    
    # Analysis Control Frame
    analysis_frame = ttk.LabelFrame(parent, text="Analysis", padding="10")
    analysis_frame.pack(fill=tk.X)
    
    analyze_btn = ttk.Button(
        analysis_frame,
        text="Run Analysis",
        command=callbacks.get('analyze_data'),
        state=tk.DISABLED
    )
    analyze_btn.pack(fill=tk.X)
    tab_widgets["analyze_data_button"] = analyze_btn
    
    save_report_btn = ttk.Button(
        analysis_frame,
        text="Save Report",
        command=callbacks.get('save_report'),
        state=tk.DISABLED
    )
    save_report_btn.pack(fill=tk.X, pady=(5, 0))
    tab_widgets["save_report_button"] = save_report_btn

def update_window_link_status(ticker, tab_widgets, is_linked=False, window_handle=None):
    """Update UI to reflect window link status"""
    
    if "window_status_label" in tab_widgets:
        if is_linked:
            tab_widgets["window_status_label"].config(
                text=f"Window: Linked (Handle: {window_handle})",
                foreground="green"
            )
        else:
            tab_widgets["window_status_label"].config(
                text="Window: Not Linked",
                foreground="red"
            )
    
    if "select_window_button" in tab_widgets:
        tab_widgets["select_window_button"].config(
            text="Change Window" if is_linked else "Select ToS Window"
        )
    
    if "start_monitoring_button" in tab_widgets:
        # Enable start monitoring only if window is linked
        tab_widgets["start_monitoring_button"].config(
            state=tk.NORMAL if is_linked else tk.DISABLED
        )

def update_monitoring_statistics(ticker, tab_widgets, stats):
    """Update monitoring statistics in UI"""
    
    if "trades_today_label" in tab_widgets:
        tab_widgets["trades_today_label"].config(
            text=f"Trades Today: {stats.get('total_new_trades', 0)}"
        )
    
    if "last_trade_label" in tab_widgets:
        last_trade = stats.get('last_trade_time', '--:--:--')
        tab_widgets["last_trade_label"].config(
            text=f"Last Trade: {last_trade}"
        )
    
    if "monitoring_status_label" in tab_widgets:
        if stats.get('is_running', False):
            status_text = "Status: Monitoring Active"
            status_color = "green"
            
            if stats.get('extraction_errors', 0) > 0:
                status_text += f" (Errors: {stats['extraction_errors']})"
                status_color = "orange"
        else:
            status_text = "Status: Not Active"
            status_color = "red"
        
        tab_widgets["monitoring_status_label"].config(
            text=status_text,
            foreground=status_color
        )

def _build_left_panel(parent: ttk.Frame, ticker: str, callbacks: dict, 
                     tab_widgets: dict, log_func: Callable):
    """Build the left panel with controls and data input"""
    
    # Monitoring Control Section
    monitoring_frame = ttk.LabelFrame(parent, text="Real-time Monitoring", padding="10")
    monitoring_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Monitoring status
    status_label = ttk.Label(monitoring_frame, text="Status: Not Monitoring", foreground="red")
    status_label.pack(anchor=tk.W)
    tab_widgets["monitoring_status_label"] = status_label
    
    # Monitoring buttons
    monitor_buttons_frame = ttk.Frame(monitoring_frame)
    monitor_buttons_frame.pack(fill=tk.X, pady=(5, 0))
    
    start_monitoring_btn = ttk.Button(
        monitor_buttons_frame, 
        text="Start Monitoring",
        command=callbacks.get('start_monitoring')
    )
    start_monitoring_btn.pack(side=tk.LEFT, padx=(0, 5))
    tab_widgets["start_monitoring_button"] = start_monitoring_btn
    
    stop_monitoring_btn = ttk.Button(
        monitor_buttons_frame,
        text="Stop Monitoring", 
        command=callbacks.get('stop_monitoring'),
        state=tk.DISABLED
    )
    stop_monitoring_btn.pack(side=tk.LEFT)
    tab_widgets["stop_monitoring_button"] = stop_monitoring_btn
    
    # Data Input Section
    data_input_frame = ttk.LabelFrame(parent, text="Data Input & Processing", padding="10")
    data_input_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    # Raw data input
    ttk.Label(data_input_frame, text="Raw Data (Paste from ToS):").pack(anchor=tk.W)
    
    raw_data_input = ModernScrolledText(data_input_frame, height=8)
    raw_data_input.pack(fill=tk.BOTH, expand=True, pady=(5, 10))
    tab_widgets["raw_data_input_area"] = raw_data_input
    
    # Processing buttons
    process_buttons_frame = ttk.Frame(data_input_frame)
    process_buttons_frame.pack(fill=tk.X)
    
    process_btn = ttk.Button(
        process_buttons_frame,
        text="Process Data",
        command=callbacks.get('process_data')
    )
    process_btn.pack(side=tk.LEFT, padx=(0, 5))
    tab_widgets["process_data_button"] = process_btn
    
    analyze_btn = ttk.Button(
        process_buttons_frame,
        text="Run Analysis",
        command=callbacks.get('analyze_data'),
        state=tk.DISABLED
    )
    analyze_btn.pack(side=tk.LEFT)
    tab_widgets["analyze_data_button"] = analyze_btn
    
    # Statistics Display
    stats_frame = ttk.LabelFrame(parent, text="Live Option Statistics", padding="10")
    stats_frame.pack(fill=tk.X, pady=(0, 10))
    
    stats_display = ModernScrolledText(stats_frame, height=6, state=tk.DISABLED)
    stats_display.pack(fill=tk.BOTH, expand=True)
    tab_widgets["option_stats_display"] = stats_display
    
    # Quick Actions
    actions_frame = ttk.LabelFrame(parent, text="Quick Actions", padding="10")
    actions_frame.pack(fill=tk.X)
    
    save_report_btn = ttk.Button(
        actions_frame,
        text="Save Report",
        command=callbacks.get('save_report'),
        state=tk.DISABLED
    )
    save_report_btn.pack(fill=tk.X, pady=(0, 5))
    tab_widgets["save_report_button"] = save_report_btn
    
    fetch_rtd_btn = ttk.Button(
        actions_frame,
        text="Fetch RTD Data",
        command=callbacks.get('fetch_rtd'),
        state=tk.DISABLED
    )
    fetch_rtd_btn.pack(fill=tk.X)
    tab_widgets["fetch_rtd_button"] = fetch_rtd_btn

def _build_right_panel(parent: ttk.Frame, ticker: str, callbacks: dict,
                      tab_widgets: dict, log_func: Callable):
    """Build the right panel with analysis results"""
    
    # Create notebook for analysis results
    results_notebook = ttk.Notebook(parent)
    results_notebook.pack(fill=tk.BOTH, expand=True)
    
    # Alpha Signals Tab
    alpha_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(alpha_tab, text="Alpha Signals")
    _build_alpha_signals_tab(alpha_tab, callbacks, tab_widgets)
    
    # Trade Briefing Tab
    briefing_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(briefing_tab, text="Trade Briefing")
    _build_trade_briefing_tab(briefing_tab, tab_widgets)
    
    # Data Preview Tab
    preview_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(preview_tab, text="Data Preview")
    _build_data_preview_tab(preview_tab, tab_widgets)
    
    # RTD Results Tab
    rtd_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(rtd_tab, text="RTD Results")
    _build_rtd_results_tab(rtd_tab, tab_widgets)
    
    # Analysis Log Tab
    log_tab = ttk.Frame(results_notebook, padding="5")
    results_notebook.add(log_tab, text="Analysis Log")
    _build_analysis_log_tab(log_tab, tab_widgets)

def _build_alpha_signals_tab(parent: ttk.Frame, callbacks: dict, tab_widgets: dict):
    """Build the alpha signals tab"""
    
    # Alpha monitoring controls
    control_frame = ttk.Frame(parent)
    control_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Alpha monitor status
    alpha_status = ttk.Label(control_frame, text="Alpha Monitor: Inactive", foreground="red")
    alpha_status.pack(side=tk.LEFT)
    tab_widgets["alpha_monitor_status"] = alpha_status
    
    # Alpha monitoring button
    start_alpha_btn = ttk.Button(
        control_frame,
        text="Start Alpha Monitor",
        command=callbacks.get('start_alpha_monitor'),
        state=tk.DISABLED
    )
    start_alpha_btn.pack(side=tk.RIGHT, padx=(5, 0))
    tab_widgets["start_alpha_monitor_button"] = start_alpha_btn
    
    # Export signals button
    export_signals_btn = ttk.Button(
        control_frame,
        text="Export Signals",
        command=callbacks.get('export_signals')
    )
    export_signals_btn.pack(side=tk.RIGHT)
    tab_widgets["export_signals_button"] = export_signals_btn
    
    # Create paned window for signals and report
    signals_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
    signals_paned.pack(fill=tk.BOTH, expand=True)
    
    # Live signals display
    signals_frame = ttk.LabelFrame(signals_paned, text="Live Alpha Signals", padding="5")
    signals_paned.add(signals_frame, weight=1)
    
    signals_display = ModernScrolledText(signals_frame, height=15)
    signals_display.pack(fill=tk.BOTH, expand=True)
    tab_widgets["alpha_signals_display"] = signals_display
    
    # Alpha report
    report_frame = ttk.LabelFrame(signals_paned, text="Alpha Analysis Report", padding="5")
    signals_paned.add(report_frame, weight=1)
    
    alpha_report = ModernScrolledText(report_frame, height=15, state=tk.DISABLED)
    alpha_report.pack(fill=tk.BOTH, expand=True)
    tab_widgets["alpha_report_area"] = alpha_report

def _build_trade_briefing_tab(parent: ttk.Frame, tab_widgets: dict):
    """Build the trade briefing tab"""
    
    briefing_frame = ttk.LabelFrame(parent, text="Executive Trade Briefing", padding="5")
    briefing_frame.pack(fill=tk.BOTH, expand=True)
    
    briefing_area = ModernScrolledText(briefing_frame, state=tk.DISABLED)
    briefing_area.pack(fill=tk.BOTH, expand=True)
    tab_widgets["trade_briefing_area"] = briefing_area

def _build_data_preview_tab(parent: ttk.Frame, tab_widgets: dict):
    """Build the data preview tab"""
    
    # Create paned window for raw and cleaned data
    preview_paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
    preview_paned.pack(fill=tk.BOTH, expand=True)
    
    # Raw data preview
    raw_frame = ttk.LabelFrame(preview_paned, text="Raw Data Preview", padding="5")
    preview_paned.add(raw_frame, weight=1)
    
    raw_preview = ModernScrolledText(raw_frame, state=tk.DISABLED)
    raw_preview.pack(fill=tk.BOTH, expand=True)
    tab_widgets["raw_data_preview_area"] = raw_preview
    
    # Cleaned data preview
    cleaned_frame = ttk.LabelFrame(preview_paned, text="Processed Data Preview", padding="5")
    preview_paned.add(cleaned_frame, weight=1)
    
    cleaned_preview = ModernScrolledText(cleaned_frame, state=tk.DISABLED)
    cleaned_preview.pack(fill=tk.BOTH, expand=True)
    tab_widgets["cleaned_data_preview_area"] = cleaned_preview

def _build_rtd_results_tab(parent: ttk.Frame, tab_widgets: dict):
    """Build the RTD results tab"""
    
    rtd_frame = ttk.LabelFrame(parent, text="Real-Time Data Results", padding="5")
    rtd_frame.pack(fill=tk.BOTH, expand=True)
    
    rtd_results = ModernScrolledText(rtd_frame, state=tk.DISABLED)
    rtd_results.pack(fill=tk.BOTH, expand=True)
    tab_widgets["rtd_results_area"] = rtd_results

def _build_analysis_log_tab(parent: ttk.Frame, tab_widgets: dict):
    """Build the analysis log tab"""
    
    log_frame = ttk.LabelFrame(parent, text="Analysis Log & Debug Info", padding="5")
    log_frame.pack(fill=tk.BOTH, expand=True)
    
    log_area = ModernScrolledText(log_frame)
    log_area.pack(fill=tk.BOTH, expand=True)
    tab_widgets["analysis_log_area"] = log_area

def create_status_bar(parent: tk.Tk) -> ttk.Label:
    """Create a status bar"""
    status_bar = ttk.Label(
        parent,
        text="Ready",
        relief=tk.SUNKEN,
        anchor=tk.W,
        padding=(5, 2)
    )
    return status_bar

def create_progress_dialog(parent: tk.Tk, title: str, message: str) -> tk.Toplevel:
    """Create a progress dialog window"""
    
    progress_window = tk.Toplevel(parent)
    progress_window.title(title)
    progress_window.geometry("400x150")
    progress_window.resizable(False, False)
    progress_window.transient(parent)
    progress_window.grab_set()
    
    # Center the window
    progress_window.update_idletasks()
    x = (progress_window.winfo_screenwidth() // 2) - (400 // 2)
    y = (progress_window.winfo_screenheight() // 2) - (150 // 2)
    progress_window.geometry(f"400x150+{x}+{y}")
    
    # Add content
    main_frame = ttk.Frame(progress_window, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Message label
    message_label = ttk.Label(main_frame, text=message, font=('Arial', 10))
    message_label.pack(pady=(0, 20))
    
    # Progress bar
    progress_bar = ttk.Progressbar(
        main_frame,
        mode='indeterminate',
        length=350
    )
    progress_bar.pack(pady=(0, 20))
    progress_bar.start()
    
    # Cancel button
    cancel_btn = ttk.Button(main_frame, text="Cancel")
    cancel_btn.pack()
    
    # Store references
    progress_window.progress_bar = progress_bar
    progress_window.message_label = message_label
    progress_window.cancel_btn = cancel_btn
    
    return progress_window

def update_progress_dialog(progress_window: tk.Toplevel, message: str, progress: float = None):
    """Update progress dialog"""
    if progress_window and progress_window.winfo_exists():
        progress_window.message_label.config(text=message)
        
        if progress is not None:
            # Switch to determinate mode if needed
            if progress_window.progress_bar['mode'] == 'indeterminate':
                progress_window.progress_bar.stop()
                progress_window.progress_bar.config(mode='determinate', maximum=100)
            
            progress_window.progress_bar['value'] = progress
        
        progress_window.update()

def close_progress_dialog(progress_window: tk.Toplevel):
    """Close progress dialog"""
    if progress_window and progress_window.winfo_exists():
        progress_window.progress_bar.stop()
        progress_window.destroy()

def create_tooltip(widget: tk.Widget, text: str):
    """Create a tooltip for a widget"""
    
    def on_enter(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root + 10}+{event.y_root + 10}")
        
        label = tk.Label(
            tooltip,
            text=text,
            background="lightyellow",
            relief=tk.SOLID,
            borderwidth=1,
            font=('Arial', 8)
        )
        label.pack()
        
        widget.tooltip = tooltip
    
    def on_leave(event):
        if hasattr(widget, 'tooltip'):
            widget.tooltip.destroy()
            del widget.tooltip
    
    widget.bind('<Enter>', on_enter)
    widget.bind('<Leave>', on_leave)

def configure_text_tags(text_widget: tk.Text):
    """Configure text tags for colored output"""
    
    # Define text tags with colors and formatting
    tags_config = {
        'header': {'font': ('Arial', 12, 'bold'), 'foreground': 'navy'},
        'subheader': {'font': ('Arial', 10, 'bold'), 'foreground': 'darkblue'},
        'urgent': {'foreground': 'red', 'font': ('Arial', 10, 'bold')},
        'bullish': {'foreground': 'green', 'font': ('Arial', 10, 'bold')},
        'bearish': {'foreground': 'red', 'font': ('Arial', 10, 'bold')},
        'neutral': {'foreground': 'blue'},
        'highlight': {'background': 'yellow'},
        'success': {'foreground': 'darkgreen'},
        'warning': {'foreground': 'orange'},
        'error': {'foreground': 'red', 'font': ('Arial', 10, 'bold')},
        'timestamp': {'foreground': 'gray', 'font': ('Arial', 8)},
        'money': {'foreground': 'darkgreen', 'font': ('Arial', 10, 'bold')},
        'volume': {'foreground': 'purple'},
        'greek': {'foreground': 'brown'},
        'code': {'font': ('Consolas', 9), 'background': 'lightgray'}
    }
    
    for tag_name, tag_config in tags_config.items():
        text_widget.tag_configure(tag_name, **tag_config)

def insert_formatted_text(text_widget: tk.Text, text: str, tag: str = None):
    """Insert formatted text with optional tag"""
    text_widget.insert(tk.END, text, tag)
    text_widget.see(tk.END)

def clear_text_widget(text_widget: tk.Text):
    """Clear text widget content"""
    current_state = text_widget.cget('state')
    if current_state == tk.DISABLED:
        text_widget.config(state=tk.NORMAL)
    
    text_widget.delete('1.0', tk.END)
    
    if current_state == tk.DISABLED:
        text_widget.config(state=tk.DISABLED)

def set_text_widget_content(text_widget: tk.Text, content: str, tag: str = None):
    """Set text widget content, replacing existing content"""
    current_state = text_widget.cget('state')
    if current_state == tk.DISABLED:
        text_widget.config(state=tk.NORMAL)
    
    text_widget.delete('1.0', tk.END)
    text_widget.insert('1.0', content, tag)
    text_widget.see(tk.END)
    
    if current_state == tk.DISABLED:
        text_widget.config(state=tk.DISABLED)

def create_alert_popup(parent: tk.Tk, title: str, message: str, alert_type: str = "info"):
    """Create an alert popup window"""
    
    popup = tk.Toplevel(parent)
    popup.title(title)
    popup.geometry("400x200")
    popup.resizable(False, False)
    popup.transient(parent)
    popup.grab_set()
    
    # Center the window
    popup.update_idletasks()
    x = (popup.winfo_screenwidth() // 2) - (400 // 2)
    y = (popup.winfo_screenheight() // 2) - (200 // 2)
    popup.geometry(f"400x200+{x}+{y}")
    
    # Configure based on alert type
    colors = {
        "info": {"bg": "lightblue", "fg": "navy"},
        "warning": {"bg": "lightyellow", "fg": "darkorange"},
        "error": {"bg": "lightcoral", "fg": "darkred"},
        "success": {"bg": "lightgreen", "fg": "darkgreen"}
    }
    
    color_config = colors.get(alert_type, colors["info"])
    popup.configure(bg=color_config["bg"])
    
    # Main frame
    main_frame = ttk.Frame(popup, padding="20")
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Message
    message_label = ttk.Label(
        main_frame,
        text=message,
        font=('Arial', 11),
        wraplength=350,
        justify=tk.CENTER
    )
    message_label.pack(expand=True)
    
    # OK button
    ok_btn = ttk.Button(
        main_frame,
        text="OK",
        command=popup.destroy
    )
    ok_btn.pack(pady=(20, 0))
    
    # Auto-close for non-error alerts after 5 seconds
    if alert_type != "error":
        popup.after(5000, lambda: popup.destroy() if popup.winfo_exists() else None)
    
    return popup

def validate_numeric_input(char: str) -> bool:
    """Validate numeric input for Entry widgets"""
    return char.isdigit() or char in '.-'

def create_numeric_entry(parent: tk.Widget, **kwargs) -> ttk.Entry:
    """Create a numeric-only entry widget"""
    vcmd = (parent.register(validate_numeric_input), '%S')
    
    entry = ttk.Entry(parent, validate='key', validatecommand=vcmd, **kwargs)
    return entry

def get_screen_dimensions() -> Tuple[int, int]:
    """Get screen dimensions"""
    root = tk.Tk()
    root.withdraw()  # Hide the window
    
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    
    root.destroy()
    return width, height

def center_window(window: tk.Toplevel, width: int, height: int):
    """Center a window on screen"""
    screen_width, screen_height = get_screen_dimensions()
    
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    window.geometry(f"{width}x{height}+{x}+{y}")

class AutoResizingFrame(ttk.Frame):
    """Frame that automatically resizes based on content"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind('<Configure>', self._on_configure)
    
    def _on_configure(self, event):
        # Update scroll region when content changes
        if hasattr(self.master, 'configure'):
            self.master.configure(scrollregion=self.master.bbox("all"))

class CollapsibleFrame(ttk.Frame):
    """A frame that can be collapsed/expanded"""
    
    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title = title
        self.is_expanded = True
        
        # Title frame with expand/collapse button
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=tk.X)
        
        self.toggle_btn = ttk.Button(
            self.title_frame,
            text="▼",
            width=3,
            command=self.toggle
        )
        self.toggle_btn.pack(side=tk.LEFT)
        
        self.title_label = ttk.Label(
            self.title_frame,
            text=title,
            font=('Arial', 10, 'bold')
        )
        self.title_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
    
    def toggle(self):
        """Toggle expanded/collapsed state"""
        if self.is_expanded:
            self.content_frame.pack_forget()
            self.toggle_btn.config(text="▶")
            self.is_expanded = False
        else:
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.toggle_btn.config(text="▼")
            self.is_expanded = True
    
    def expand(self):
        """Expand the frame"""
        if not self.is_expanded:
            self.toggle()
    
    def collapse(self):
        """Collapse the frame"""
        if self.is_expanded:
            self.toggle()

def create_data_table(parent: tk.Widget, columns: list, data: list = None) -> ttk.Treeview:
    """Create a data table using Treeview"""
    
    # Create frame for table and scrollbars
    table_frame = ttk.Frame(parent)
    table_frame.pack(fill=tk.BOTH, expand=True)
    
    # Create treeview
    tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
    
    # Configure columns
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor=tk.CENTER)
    
    # Add scrollbars
    v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
    h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
    
    tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
    
    # Pack scrollbars and tree
    tree.grid(row=0, column=0, sticky='nsew')
    v_scrollbar.grid(row=0, column=1, sticky='ns')
    h_scrollbar.grid(row=1, column=0, sticky='ew')
    
    # Configure grid weights
    table_frame.grid_rowconfigure(0, weight=1)
    table_frame.grid_columnconfigure(0, weight=1)
    
    # Add data if provided
    if data:
        for row in data:
            tree.insert('', 'end', values=row)
    
    return tree

def update_data_table(tree: ttk.Treeview, data: list):
    """Update data table with new data"""
    # Clear existing data
    for item in tree.get_children():
        tree.delete(item)
    
    # Insert new data
    for row in data:
        tree.insert('', 'end', values=row)

def create_metrics_dashboard(parent: tk.Widget) -> dict[str, ttk.Label]:
    """Create a metrics dashboard with key performance indicators"""
    
    dashboard_frame = ttk.LabelFrame(parent, text="Key Metrics Dashboard", padding="10")
    dashboard_frame.pack(fill=tk.X, pady=(0, 10))
    
    # Create grid of metric labels
    metrics = {}
    
    # Row 1 - Volume metrics
    ttk.Label(dashboard_frame, text="Total Volume:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w', padx=(0, 10))
    metrics['total_volume'] = ttk.Label(dashboard_frame, text="0", foreground='blue')
    metrics['total_volume'].grid(row=0, column=1, sticky='w', padx=(0, 20))
    
    ttk.Label(dashboard_frame, text="Total Notional:", font=('Arial', 9, 'bold')).grid(row=0, column=2, sticky='w', padx=(0, 10))
    metrics['total_notional'] = ttk.Label(dashboard_frame, text="$0", foreground='green')
    metrics['total_notional'].grid(row=0, column=3, sticky='w')
    
    # Row 2 - Call/Put metrics
    ttk.Label(dashboard_frame, text="Call/Put Ratio:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky='w', padx=(0, 10))
    metrics['call_put_ratio'] = ttk.Label(dashboard_frame, text="0.00", foreground='purple')
    metrics['call_put_ratio'].grid(row=1, column=1, sticky='w', padx=(0, 20))
    
    ttk.Label(dashboard_frame, text="Alpha Signals:", font=('Arial', 9, 'bold')).grid(row=1, column=2, sticky='w', padx=(0, 10))
    metrics['alpha_signals'] = ttk.Label(dashboard_frame, text="0", foreground='red')
    metrics['alpha_signals'].grid(row=1, column=3, sticky='w')
    
    # Row 3 - Time and activity
    ttk.Label(dashboard_frame, text="Last Update:", font=('Arial', 9, 'bold')).grid(row=2, column=0, sticky='w', padx=(0, 10))
    metrics['last_update'] = ttk.Label(dashboard_frame, text="Never", foreground='gray')
    metrics['last_update'].grid(row=2, column=1, sticky='w', padx=(0, 20))
    
    ttk.Label(dashboard_frame, text="Unique Options:", font=('Arial', 9, 'bold')).grid(row=2, column=2, sticky='w', padx=(0, 10))
    metrics['unique_options'] = ttk.Label(dashboard_frame, text="0", foreground='navy')
    metrics['unique_options'].grid(row=2, column=3, sticky='w')
    
    return metrics

def update_metrics_dashboard(metrics: dict[str, ttk.Label], data: dict):
    """Update metrics dashboard with new data"""
    
    # Update each metric if data is available
    if 'total_volume' in data:
        metrics['total_volume'].config(text=f"{data['total_volume']:,}")
    
    if 'total_notional' in data:
        metrics['total_notional'].config(text=f"${data['total_notional']:,.0f}")
    
    if 'call_put_ratio' in data:
        ratio = data['call_put_ratio']
        color = 'green' if ratio > 1 else 'red' if ratio < 1 else 'gray'
        metrics['call_put_ratio'].config(text=f"{ratio:.2f}", foreground=color)
    
    if 'alpha_signals' in data:
        count = data['alpha_signals']
        color = 'red' if count > 0 else 'gray'
        metrics['alpha_signals'].config(text=str(count), foreground=color)
    
    if 'last_update' in data:
        metrics['last_update'].config(text=data['last_update'], foreground='green')
    
    if 'unique_options' in data:
        metrics['unique_options'].config(text=str(data['unique_options']))

def create_alert_banner(parent: tk.Widget) -> ttk.Label:
    """Create an alert banner for important notifications"""
    
    alert_frame = ttk.Frame(parent)
    alert_frame.pack(fill=tk.X, pady=(0, 5))
    
    alert_label = ttk.Label(
        alert_frame,
        text="",
        font=('Arial', 10, 'bold'),
        anchor=tk.CENTER,
        relief=tk.RAISED,
        padding=(10, 5)
    )
    alert_label.pack(fill=tk.X)
    
    # Initially hide the banner
    alert_frame.pack_forget()
    
    # Store reference to frame for show/hide
    alert_label.alert_frame = alert_frame
    
    return alert_label

def show_alert_banner(alert_label: ttk.Label, message: str, alert_type: str = "info"):
    """Show alert banner with message"""
    
    colors = {
        "info": {"bg": "lightblue", "fg": "navy"},
        "warning": {"bg": "lightyellow", "fg": "darkorange"}, 
        "error": {"bg": "lightcoral", "fg": "darkred"},
        "success": {"bg": "lightgreen", "fg": "darkgreen"},
        "urgent": {"bg": "red", "fg": "white"}
    }
    
    color_config = colors.get(alert_type, colors["info"])
    
    alert_label.config(
        text=message,
        background=color_config["bg"],
        foreground=color_config["fg"]
    )
    
    # Show the banner
    alert_label.alert_frame.pack(fill=tk.X, pady=(0, 5))
    
    # Auto-hide after 10 seconds for non-urgent alerts
    if alert_type != "urgent":
        alert_label.after(10000, lambda: hide_alert_banner(alert_label))

def hide_alert_banner(alert_label: ttk.Label):
    """Hide alert banner"""
    if hasattr(alert_label, 'alert_frame') and alert_label.alert_frame.winfo_exists():
        alert_label.alert_frame.pack_forget()

def create_loading_overlay(parent: tk.Widget, message: str = "Loading...") -> tk.Toplevel:
    """Create a loading overlay"""
    
    overlay = tk.Toplevel(parent)
    overlay.title("")
    overlay.geometry("300x100")
    overlay.resizable(False, False)
    overlay.transient(parent)
    overlay.grab_set()
    
    # Remove window decorations
    overlay.overrideredirect(True)
    
    # Center on parent
    parent.update_idletasks()
    x = parent.winfo_rootx() + (parent.winfo_width() // 2) - 150
    y = parent.winfo_rooty() + (parent.winfo_height() // 2) - 50
    overlay.geometry(f"300x100+{x}+{y}")
    
    # Create semi-transparent background
    overlay.configure(bg='white', relief=tk.RAISED, borderwidth=2)
    
    # Loading message
    message_label = ttk.Label(overlay, text=message, font=('Arial', 11))
    message_label.pack(expand=True)
    
    # Animated progress indicator
    progress_label = ttk.Label(overlay, text="●●●", font=('Arial', 14))
    progress_label.pack()
    
    # Animation
    def animate():
        if overlay.winfo_exists():
            current = progress_label.cget('text')
            if current == "●●●":
                progress_label.config(text="○●●")
            elif current == "○●●":
                progress_label.config(text="○○●")
            elif current == "○○●":
                progress_label.config(text="○○○")
            else:
                progress_label.config(text="●●●")
            
            overlay.after(500, animate)
    
    animate()
    
    return overlay

def create_context_menu(widget: tk.Widget, menu_items: dict[str, Callable]) -> tk.Menu:
    """Create a context menu for a widget"""
    
    context_menu = tk.Menu(widget, tearoff=0)
    
    for label, command in menu_items.items():
        if label == "separator":
            context_menu.add_separator()
        else:
            context_menu.add_command(label=label, command=command)
    
    def show_context_menu(event):
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    widget.bind("<Button-3>", show_context_menu)  # Right click
    
    return context_menu

def configure_widget_fonts(parent: tk.Widget, base_size: int = 9):
    """Configure font sizes for all widgets in a parent"""
    
    def configure_fonts_recursive(widget, size):
        widget_type = widget.winfo_class()
        
        if widget_type in ['Label', 'Button', 'Entry', 'Text']:
            try:
                current_font = widget.cget('font')
                if isinstance(current_font, str):
                    # Extract font family
                    font_family = current_font.split()[0] if current_font else 'Arial'
                else:
                    font_family = 'Arial'
                
                widget.configure(font=(font_family, size))
            except:
                pass
        
        # Recursively configure children
        for child in widget.winfo_children():
            configure_fonts_recursive(child, size)
    
    configure_fonts_recursive(parent, base_size)

def create_splitter_window(parent: tk.Widget, orientation: str = 'horizontal') -> ttk.PanedWindow:
    """Create a splitter window"""
    
    orient = tk.HORIZONTAL if orientation.lower() == 'horizontal' else tk.VERTICAL
    
    paned_window = ttk.PanedWindow(parent, orient=orient)
    paned_window.pack(fill=tk.BOTH, expand=True)
    
    return paned_window

def add_keyboard_shortcuts(root: tk.Tk, shortcuts: dict[str, Callable]):
    """Add keyboard shortcuts to the root window"""
    
    for key_combo, command in shortcuts.items():
        root.bind(key_combo, lambda event, cmd=command: cmd())

def create_tabbed_interface(parent: tk.Widget, tabs: dict[str, tk.Widget]) -> ttk.Notebook:
    """Create a tabbed interface"""
    
    notebook = ttk.Notebook(parent)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    for tab_name, tab_widget in tabs.items():
        notebook.add(tab_widget, text=tab_name)
    
    return notebook

def save_window_state(window: tk.Tk, config_file: str):
    """Save window position and size"""
    try:
        geometry = window.geometry()
        with open(config_file, 'w') as f:
            f.write(geometry)
    except Exception as e:
        print(f"Error saving window state: {e}")

def restore_window_state(window: tk.Tk, config_file: str):
    """Restore window position and size"""
    try:
        with open(config_file, 'r') as f:
            geometry = f.read().strip()
            window.geometry(geometry)
    except FileNotFoundError:
        # Default size if no saved state
        window.geometry("1200x800")
    except Exception as e:
        print(f"Error restoring window state: {e}")
        window.geometry("1200x800")

def apply_modern_styling(root: tk.Tk, style: ttk.Style):
    """Apply modern styling to the application"""
    
    # Configure modern colors
    style.theme_use('clam')
    
    # Configure notebook tabs
    style.configure('TNotebook.Tab', 
                   padding=[20, 8], 
                   font=('Arial', 9))
    
    # Configure buttons
    style.configure('TButton',
                   padding=[10, 5],
                   font=('Arial', 9))
    
    # Configure frames
    style.configure('TLabelFrame',
                   relief='flat',
                   borderwidth=1)
    
    # Configure treeview
    style.configure('Treeview',
                   background='white',
                   foreground='black',
                   fieldbackground='white')
    style.configure('Treeview.Heading',
                   font=('Arial', 9, 'bold'))

def create_main_window_layout(root: tk.Tk) -> dict[str, tk.Widget]:
    """Create the main window layout and return widget references"""
    
    widgets = {}
    
    # Configure root window
    root.title("Options Flow & Alpha Analyzer")
    root.geometry("1400x900")
    
    # Create main container
    main_container = ttk.Frame(root)
    main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    widgets['main_container'] = main_container
    
    # Create toolbar
    toolbar = ttk.Frame(main_container)
    toolbar.pack(fill=tk.X, pady=(0, 5))
    widgets['toolbar'] = toolbar
    
    # Create main content area
    content_area = ttk.Frame(main_container)
    content_area.pack(fill=tk.BOTH, expand=True)
    widgets['content_area'] = content_area
    
    # Create status bar
    status_bar = create_status_bar(root)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 5))
    widgets['status_bar'] = status_bar
    
    return widgets

# Constants for consistent spacing and sizing
PADDING_SMALL = 5
PADDING_MEDIUM = 10
PADDING_LARGE = 15

BUTTON_WIDTH_SMALL = 10
BUTTON_WIDTH_MEDIUM = 15
BUTTON_WIDTH_LARGE = 20

ENTRY_WIDTH_SMALL = 10
ENTRY_WIDTH_MEDIUM = 20
ENTRY_WIDTH_LARGE = 30

# Color schemes for different themes
THEME_COLORS = {
    'light': {
        'bg': '#FFFFFF',
        'fg': '#000000',
        'accent': '#0078D4',
        'success': '#107C10',
        'warning': '#FF8C00',
        'error': '#D13438'
    },
    'dark': {
        'bg': '#2D2D30',
        'fg': '#FFFFFF',
        'accent': '#0E639C',
        'success': '#0F7B0F',
        'warning': '#CA5010',
        'error': '#A4262C'
    }
}