# ui_builder.py
# Functions for creating the Tkinter UI components and managing themes.

import tkinter as tk
from tkinter import scrolledtext, ttk, font as tkFont
import config # For THEMES

# --- Theme Application Function ---
def apply_theme_to_widgets(root, style, theme_name, widgets_map):
    if not widgets_map or not style or not root: return
    
    theme = config.THEMES.get(theme_name)
    if not theme:
        # print(f"Warning: Theme '{theme_name}' not found in config. Using default.")
        return 

    # Apply to root window
    if "root" in widgets_map and widgets_map["root"] == root:
        try: root.config(bg=theme["bg"])
        except tk.TclError: pass # In case root is already destroyed

    # Configure ttk styles
    style.configure("TFrame", background=theme["frame_bg"])
    style.configure("TLabel", background=theme["label_bg"], foreground=theme["label_fg"])
    style.configure("TButton", background=theme["button_bg"], foreground=theme["button_fg"],
                    focuscolor=theme.get("ttk_button_focus_color", theme["fg"])) 
    style.map("TButton", background=[('active', theme["button_active_bg"])])
    
    style.configure("TEntry",
                    fieldbackground=theme["entry_bg"],
                    foreground=theme["entry_fg"],
                    insertcolor=theme["fg"]) # Cursor color
    style.map("TEntry",
              selectbackground=[('focus', theme.get("ttk_entry_select_bg", "blue")), ('!focus', theme.get("ttk_entry_select_bg", "blue"))],
              selectforeground=[('focus', theme.get("ttk_entry_select_fg", "white")), ('!focus', theme.get("ttk_entry_select_fg", "white"))])

    style.configure("TLabelframe", background=theme["frame_bg"], bordercolor=theme.get("fg")) 
    style.configure("TLabelframe.Label", background=theme["label_bg"], foreground=theme["label_fg"])
    
    # Configure Scrollbars
    style.configure("Vertical.TScrollbar", background=theme["button_bg"], troughcolor=theme["frame_bg"])
    style.configure("Horizontal.TScrollbar", background=theme["button_bg"], troughcolor=theme["frame_bg"])
    
    style.configure("TMenubutton", background=theme["button_bg"], foreground=theme["button_fg"]) # For OptionMenu
    
    # Configure Notebook and Tabs
    style.configure("TNotebook", background=theme["bg"])
    style.configure("TNotebook.Tab",
                    background=theme.get("notebook_tab_bg", theme["button_bg"]),
                    foreground=theme.get("notebook_tab_fg", theme["button_fg"]),
                    padding=[5, 2]) 
    style.map("TNotebook.Tab",
              background=[("selected", theme.get("notebook_tab_selected_bg", theme["button_active_bg"]))],
              foreground=[("selected", theme.get("notebook_tab_selected_fg", theme["fg"]))],
              expand=[("selected", [1, 1, 1, 0])]) # Optional: make selected tab slightly larger

    # Apply to specific ScrolledText widgets based on their role
    text_widget_keys_suffixes = ["data_text_area", "analysis_results_area", "rtd_results_area", "trade_briefing_area"]
    for widget_key_prefix, widget_obj in widgets_map.items():
        if not isinstance(widget_obj, tk.Widget) or not widget_obj.winfo_exists():
            continue
        # Check if the widget_key_prefix ends with one of the suffixes
        # This is a bit broad; ideally, widget_key_prefix IS the key from widgets_map
        for suffix in text_widget_keys_suffixes:
            if widget_key_prefix.endswith(suffix): # More precise check
                if isinstance(widget_obj, scrolledtext.ScrolledText):
                    # Determine bg/fg based on the role of the text area
                    bg_key = "results_bg" if "results" in suffix or "briefing" in suffix else "text_bg"
                    fg_key = "results_fg" if "results" in suffix or "briefing" in suffix else "text_fg"
                    try:
                        widget_obj.config(bg=theme[bg_key], fg=theme[fg_key], insertbackground=theme["fg"])
                    except tk.TclError: pass # Widget might be destroyed
                break # Found matching suffix for this widget_obj

    # Apply to Status Label specifically
    status_label_widget = widgets_map.get("status_label")
    if status_label_widget and isinstance(status_label_widget, ttk.Label) and status_label_widget.winfo_exists():
        try:
            # Define a unique style for the status label if not already done
            style.configure("Status.TLabel", background=theme["status_bg"], foreground=theme["status_fg"], padding=(3,3))
            status_label_widget.configure(style="Status.TLabel")
        except tk.TclError: pass


    # Apply generic ttk styles to other ttk widgets if not handled specifically
    for key, widget_or_list in widgets_map.items():
        if widget_or_list is None: continue
        
        widgets_to_style_list = widget_or_list if isinstance(widget_or_list, list) else [widget_or_list]
        
        for widget in widgets_to_style_list:
            if not isinstance(widget, tk.Widget) or not widget.winfo_exists(): continue

            style_to_apply = None
            # Determine the base ttk style name
            if isinstance(widget, ttk.Frame) and not isinstance(widget, ttk.LabelFrame) and not isinstance(widget, ttk.Notebook): style_to_apply = "TFrame"
            elif isinstance(widget, ttk.LabelFrame): style_to_apply = "TLabelframe"
            elif isinstance(widget, ttk.Label) and key != "status_label": style_to_apply = "TLabel" # Avoid re-styling status_label if already done
            elif isinstance(widget, ttk.Button): style_to_apply = "TButton"
            elif isinstance(widget, ttk.Entry): style_to_apply = "TEntry"
            elif isinstance(widget, ttk.OptionMenu): style_to_apply = "TMenubutton"
            elif isinstance(widget, ttk.Notebook): style_to_apply = "TNotebook"
            elif isinstance(widget, ttk.Scrollbar):
                orient = widget.cget("orient")
                style_to_apply = "Vertical.TScrollbar" if orient == tk.VERTICAL else "Horizontal.TScrollbar"

            if style_to_apply:
                try:
                    widget.configure(style=style_to_apply)
                except tk.TclError: pass # Widget might be destroyed or style not applicable

            # Special handling for LabelFrame's labelwidget (the text part of the LabelFrame)
            if isinstance(widget, ttk.LabelFrame):
                if hasattr(widget, 'labelwidget') and widget.labelwidget and widget.labelwidget.winfo_exists():
                    try:
                        widget.labelwidget.configure(background=theme["label_bg"], foreground=theme["label_fg"])
                    except tk.TclError: pass
            
            # For non-ttk tk.Canvas (used for scrollable frame)
            if isinstance(widget, tk.Canvas) and widget.winfo_exists():
                try: widget.config(bg=theme["bg"]) # Match general background
                except tk.TclError: pass


def create_ticker_tab_content(parent_tab_frame, ticker_symbol, command_callbacks):
    """Creates the content for a new ticker tab."""
    tab_widgets = {}

    # --- Scrollable Frame Setup ---
    # Main canvas for scrolling
    tab_canvas = tk.Canvas(parent_tab_frame, highlightthickness=0) # highlightthickness=0 to remove border
    tab_widgets[f"{ticker_symbol}_tab_canvas"] = tab_canvas 
    
    # Vertical scrollbar
    tab_scrollbar = ttk.Scrollbar(parent_tab_frame, orient="vertical", command=tab_canvas.yview)
    tab_widgets[f"{ticker_symbol}_tab_scrollbar"] = tab_scrollbar
    
    # Actual frame to hold content, placed inside the canvas
    tab_scrollable_frame = ttk.Frame(tab_canvas)
    tab_widgets[f"{ticker_symbol}_tab_scrollable_frame"] = tab_scrollable_frame

    # Configure canvas scrolling
    tab_scrollable_frame.bind("<Configure>", lambda e: tab_canvas.configure(scrollregion=tab_canvas.bbox("all")))
    tab_canvas.create_window((0, 0), window=tab_scrollable_frame, anchor="nw") # Embed frame in canvas
    tab_canvas.configure(yscrollcommand=tab_scrollbar.set)

    # Pack canvas and scrollbar
    tab_canvas.pack(side="left", fill="both", expand=True)
    tab_scrollbar.pack(side="right", fill="y")

    # --- Content Sections within the Scrollable Frame ---

    # Section 1: Raw Data Input
    data_input_lf = ttk.LabelFrame(tab_scrollable_frame, text=f"1. Raw Data for {ticker_symbol}", padding=(10,5))
    data_input_lf.pack(pady=5, padx=10, fill="x", anchor="n") # anchor N to ensure it's at the top
    tab_widgets["data_input_lf"] = data_input_lf
    
    data_paste_label = ttk.Label(data_input_lf, text="Paste Raw Tab-Delimited Options Data:")
    data_paste_label.pack(anchor='w', padx=5, pady=(0,2)); tab_widgets["data_paste_label"] = data_paste_label
    data_text_area = scrolledtext.ScrolledText(data_input_lf, wrap=tk.WORD, height=10, width=80) # Default height
    data_text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=2); tab_widgets["data_text_area"] = data_text_area

    # Section 2: Detailed Flow Analysis Output
    analysis_lf = ttk.LabelFrame(tab_scrollable_frame, text="2. Detailed Flow Analysis", padding=(10,5))
    analysis_lf.pack(pady=5, padx=10, fill="both", expand=True, anchor="n"); tab_widgets["analysis_lf"] = analysis_lf
    analysis_results_area = scrolledtext.ScrolledText(analysis_lf, wrap=tk.NONE, height=20, width=80) # Increased height
    analysis_results_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5); tab_widgets["analysis_results_area"] = analysis_results_area

    # Section 3: RTD Options of Interest
    rtd_lf = ttk.LabelFrame(tab_scrollable_frame, text=f"3. RTD for {ticker_symbol} Options of Interest", padding=(10,5))
    rtd_lf.pack(pady=5, padx=10, fill="x", anchor="n"); tab_widgets["rtd_lf"] = rtd_lf
    
    rtd_entries_frame = ttk.Frame(rtd_lf) # Frame to hold multiple entry rows
    rtd_entries_frame.pack(fill=tk.X)
    tab_widgets["rtd_entries_frame"] = rtd_entries_frame

    for i in range(3): # Create 3 RTD entry fields
        opt_entry_frame = ttk.Frame(rtd_entries_frame) # One frame per entry row for better layout
        opt_entry_frame.pack(fill=tk.X, pady=1)
        tab_widgets[f"rtd_option_entry_frame_{i+1}"] = opt_entry_frame

        opt_label = ttk.Label(opt_entry_frame, text=f"RTD Opt {i+1}:", width=12)
        opt_label.pack(side=tk.LEFT, padx=(5,2)); tab_widgets[f"rtd_option_label_{i+1}"] = opt_label
        
        opt_entry = ttk.Entry(opt_entry_frame, width=30) # Entry field for option symbol
        opt_entry.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X); tab_widgets[f"rtd_option_entry_{i+1}"] = opt_entry
    
    fetch_tab_rtd_button = ttk.Button(rtd_lf, text="Fetch RTD for Above Tab Options", command=command_callbacks.get('fetch_rtd_tab_specific'))
    fetch_tab_rtd_button.pack(pady=(5,2), anchor='w', padx=5) 
    tab_widgets["fetch_tab_rtd_button"] = fetch_tab_rtd_button
    
    rtd_results_title = ttk.Label(rtd_lf, text="Fetched RTD Data:")
    rtd_results_title.pack(anchor='w', padx=5, pady=(5,0)); tab_widgets["rtd_results_title"] = rtd_results_title
    rtd_results_area = scrolledtext.ScrolledText(rtd_lf, wrap=tk.NONE, height=8, width=80) 
    rtd_results_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5); tab_widgets["rtd_results_area"] = rtd_results_area

    # Section 4: Actionable Trade Briefing
    briefing_lf = ttk.LabelFrame(tab_scrollable_frame, text="4. Actionable Trade Briefing & Strategy", padding=(10,5))
    briefing_lf.pack(pady=5, padx=10, fill="both", expand=True, anchor="n"); tab_widgets["briefing_lf"] = briefing_lf
    trade_briefing_area = scrolledtext.ScrolledText(briefing_lf, wrap=tk.WORD, height=25, width=80) # Increased height significantly
    trade_briefing_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5); tab_widgets["trade_briefing_area"] = trade_briefing_area
    
    return tab_widgets


def clear_text_area(text_area_widget):
    """Clears the content of a Tkinter Text or ScrolledText widget."""
    if text_area_widget and isinstance(text_area_widget, (tk.Text, scrolledtext.ScrolledText)):
        if text_area_widget.winfo_exists(): # Check if widget still exists
            text_area_widget.delete('1.0', tk.END)
