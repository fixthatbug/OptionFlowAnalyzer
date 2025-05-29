# coordinate_setup_wizard.py
"""
Coordinate setup wizard for configuring ToS window data extraction points
"""

import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import threading
import time
from PIL import ImageGrab, ImageTk, Image, ImageDraw
import pyautogui
import config

class CoordinateSetupWizard(tk.Toplevel):
    """Wizard for setting up extraction coordinates"""
    
    def __init__(self, parent, window_handle=None):
        super().__init__(parent)
        self.parent = parent
        self.window_handle = window_handle
        self.title("Coordinate Setup Wizard")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()
        
        # Coordinates storage
        self.coordinates = {
            "ticker_symbol": {"x": 0, "y": 0},
            "option_statistics": {"x": 0, "y": 0, "width": 0, "height": 0},
            "time_sales": {"x": 0, "y": 0, "width": 0, "height": 0}
        }
        
        # Current step
        self.current_step = 0
        self.steps = [
            ("Ticker Symbol", "ticker_symbol", "point"),
            ("Option Statistics", "option_statistics", "region"),
            ("Time & Sales", "time_sales", "region")
        ]
        
        # UI elements
        self.screenshot_label = None
        self.instruction_label = None
        self.overlay_window = None
        self.selection_start = None
        self.selection_end = None
        
        self.create_widgets()
        # Ensure instruction_label is created before calling update_step
        if self.instruction_label is not None:
            self.update_step()
    
    def create_widgets(self):
        """Create wizard UI"""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instruction frame
        instruction_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        instruction_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.instruction_label = ttk.Label(instruction_frame, text="", wraplength=750)
        self.instruction_label.pack()
        
        # Screenshot frame
        screenshot_frame = ttk.LabelFrame(main_frame, text="Screenshot Preview", padding="10")
        screenshot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for screenshot
        self.canvas = tk.Canvas(screenshot_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(screenshot_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar = ttk.Scrollbar(screenshot_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.capture_button = ttk.Button(button_frame, text="Capture Screenshot", 
                                        command=self.capture_screenshot)
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        self.select_button = ttk.Button(button_frame, text="Select Area", 
                                       command=self.start_selection, state=tk.DISABLED)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.test_button = ttk.Button(button_frame, text="Test Selection", 
                                     command=self.test_selection, state=tk.DISABLED)
        self.test_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.prev_button = ttk.Button(button_frame, text="Previous", 
                                     command=self.previous_step, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.next_button = ttk.Button(button_frame, text="Next", 
                                     command=self.next_step, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save & Close", 
                                     command=self.save_coordinates, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Step 1 of 3", foreground="blue")
        self.status_label.pack(pady=(5, 0))
    
    def update_step(self):
        """Update UI for current step"""
        step_name, coord_key, selection_type = self.steps[self.current_step]
        
        # Update instruction
        if selection_type == "point":
            instruction = (f"Step {self.current_step + 1}: Select {step_name}\n\n"
                          f"1. Click 'Capture Screenshot' to take a screenshot of your ToS window\n"
                          f"2. Click 'Select Area' and click on the {step_name} location\n"
                          f"3. Click 'Test Selection' to verify the coordinates\n"
                          f"4. Click 'Next' to proceed to the next step")
        else:
            instruction = (f"Step {self.current_step + 1}: Select {step_name} Area\n\n"
                          f"1. Click 'Capture Screenshot' to take a screenshot of your ToS window\n"
                          f"2. Click 'Select Area' and drag to select the {step_name} region\n"
                          f"3. Click 'Test Selection' to verify the selection\n"
                          f"4. Click 'Next' to proceed to the next step")
        
        if self.instruction_label is not None:
            self.instruction_label.config(text=instruction)
        if self.status_label is not None:
            self.status_label.config(text=f"Step {self.current_step + 1} of {len(self.steps)}")
        
        # Update button states
        if self.prev_button is not None:
            self.prev_button.config(state=tk.NORMAL if self.current_step > 0 else tk.DISABLED)
        if self.next_button is not None:
            self.next_button.config(state=tk.DISABLED)  # Enable after selection
        if self.save_button is not None:
            self.save_button.config(state=tk.DISABLED)  # Enable on last step
        if self.select_button is not None:
            self.select_button.config(state=tk.DISABLED)  # Enable after screenshot
    
    def capture_screenshot(self):
        """Capture screenshot of ToS window or entire screen"""
        self.capture_button.config(state=tk.DISABLED)
        self.withdraw()  # Hide wizard temporarily
        
        # Wait a moment for window to hide
        time.sleep(0.5)
        
        try:
            if self.window_handle:
                # Try to capture specific window
                import tos_data_grabber
                screenshot = tos_data_grabber.capture_window_screenshot(self.window_handle)
                if screenshot:
                    self.screenshot = screenshot
                else:
                    # Fallback to full screen
                    self.screenshot = ImageGrab.grab()
            else:
                # Capture full screen
                self.screenshot = ImageGrab.grab()
            
            # Display screenshot
            self.display_screenshot()
            self.select_button.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Screenshot Error", f"Failed to capture screenshot: {e}")
        
        finally:
            self.deiconify()  # Show wizard again
            self.capture_button.config(state=tk.NORMAL)
    
    def display_screenshot(self):
        """Display captured screenshot"""
        if not hasattr(self, 'screenshot'):
            return
        
        # Resize for display if too large
        display_width = min(self.screenshot.width, 750)
        display_height = min(self.screenshot.height, 400)
        
        aspect_ratio = self.screenshot.width / self.screenshot.height
        if display_width / display_height > aspect_ratio:
            display_width = int(display_height * aspect_ratio)
        else:
            display_height = int(display_width / aspect_ratio)
        
        # Create display image
        display_image = self.screenshot.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Store scale factors for coordinate conversion
        self.scale_x = self.screenshot.width / display_width
        self.scale_y = self.screenshot.height / display_height
    
    def start_selection(self):
        """Start coordinate selection"""
        self.select_button.config(state=tk.DISABLED)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Change cursor
        self.canvas.config(cursor="crosshair")
        
        # Update status
        step_name, _, selection_type = self.steps[self.current_step]
        if selection_type == "point":
            self.status_label.config(text=f"Click on the {step_name}")
        else:
            self.status_label.config(text=f"Drag to select the {step_name} area")
    
    def on_mouse_down(self, event):
        """Handle mouse down event"""
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Store start position
        self.selection_start = (canvas_x, canvas_y)
        
        # Clear any existing selection
        self.canvas.delete("selection")
        
        # For point selection, we're done
        _, _, selection_type = self.steps[self.current_step]
        if selection_type == "point":
            self.on_mouse_up(event)
    
    def on_mouse_drag(self, event):
        """Handle mouse drag event"""
        _, _, selection_type = self.steps[self.current_step]
        if selection_type == "point":
            return
        
        if self.selection_start:
            # Get current position
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            # Draw selection rectangle
            self.canvas.delete("selection")
            self.canvas.create_rectangle(
                self.selection_start[0], self.selection_start[1],
                canvas_x, canvas_y,
                outline="red", width=2, tags="selection"
            )
    
    def on_mouse_up(self, event):
        """Handle mouse up event"""
        if not self.selection_start:
            return
        
        # Get end position
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.selection_end = (canvas_x, canvas_y)
        
        # Convert to actual coordinates
        start_x = int(self.selection_start[0] * self.scale_x)
        start_y = int(self.selection_start[1] * self.scale_y)
        end_x = int(self.selection_end[0] * self.scale_x)
        end_y = int(self.selection_end[1] * self.scale_y)
        
        # Store coordinates
        step_name, coord_key, selection_type = self.steps[self.current_step]
        
        if selection_type == "point":
            self.coordinates[coord_key]["x"] = start_x
            self.coordinates[coord_key]["y"] = start_y
            
            # Draw point marker
            self.canvas.delete("selection")
            self.canvas.create_oval(
                canvas_x - 5, canvas_y - 5,
                canvas_x + 5, canvas_y + 5,
                fill="red", outline="yellow", width=2, tags="selection"
            )
        else:
            # Ensure correct order
            x1, x2 = min(start_x, end_x), max(start_x, end_x)
            y1, y2 = min(start_y, end_y), max(start_y, end_y)
            
            self.coordinates[coord_key]["x"] = x1
            self.coordinates[coord_key]["y"] = y1
            self.coordinates[coord_key]["width"] = x2 - x1
            self.coordinates[coord_key]["height"] = y2 - y1
        
        # Unbind events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.config(cursor="")
        
        # Update UI
        self.select_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Selection complete for {step_name}")
        
        # Enable save on last step
        if self.current_step == len(self.steps) - 1:
            self.save_button.config(state=tk.NORMAL)
    
    def test_selection(self):
        """Test current selection"""
        step_name, coord_key, selection_type = self.steps[self.current_step]
        coords = self.coordinates[coord_key]
        
        if selection_type == "point":
            if coords["x"] is not None and coords["y"] is not None:
                messagebox.showinfo("Test Result", 
                                  f"{step_name} coordinates:\nX: {coords['x']}, Y: {coords['y']}")
        else:
            if all(coords[k] is not None for k in ["x", "y", "width", "height"]):
                messagebox.showinfo("Test Result", 
                                  f"{step_name} region:\nX: {coords['x']}, Y: {coords['y']}\n"
                                  f"Width: {coords['width']}, Height: {coords['height']}")
    
    def previous_step(self):
        """Go to previous step"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step()
            self.canvas.delete("selection")
    
    def next_step(self):
        """Go to next step"""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_step()
            self.canvas.delete("selection")
            
            # Redisplay screenshot
            if hasattr(self, 'screenshot'):
                self.display_screenshot()
    
    def save_coordinates(self):
        """Save coordinates to config file"""
        try:
            # Update config
            config.USER_TOS_COORDINATES = self.coordinates
            
            # Save to file
            with open(config.COORDINATES_CONFIG_FILE, 'w') as f:
                json.dump(self.coordinates, f, indent=4)
            
            messagebox.showinfo("Success", "Coordinates saved successfully!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save coordinates: {e}")
# monitoring_manager.py
"""
Real-time monitoring manager for continuous data extraction from ToS windows
"""

import threading
import queue
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Callable, Any
import pandas as pd
import pyautogui
import pyperclip
import tos_data_grabber

class MonitoringThread(threading.Thread):
    """Thread for monitoring a single ticker"""
    
    def __init__(self, ticker: str, window_handle: int, coordinates: dict, 
                 interval_ms: int, log_func: Callable):
        super().__init__(daemon=True)
        self.ticker = ticker
        self.window_handle = window_handle
        self.coordinates = coordinates
        self.interval_ms = interval_ms
        self.log_func = log_func
        self.is_running = False
        self.data_queue = queue.Queue()
        self.last_extraction_time = None
        self.extraction_errors = 0
        self.max_errors = 5
        
    def run(self):
        """Main monitoring loop"""
        self.is_running = True
        self.log_func(f"Monitoring thread started for {self.ticker}", self.ticker)
        
        while self.is_running:
            try:
                # Extract data
                data = self._extract_data()
                
                if data:
                    # Add timestamp
                    data['extraction_time'] = datetime.now()
                    
                    # Queue data
                    self.data_queue.put(data)
                    self.last_extraction_time = datetime.now()
                    self.extraction_errors = 0
                else:
                    self.extraction_errors += 1
                    if self.extraction_errors >= self.max_errors:
                        self.log_func(f"Too many extraction errors for {self.ticker}. "
                                    f"Monitoring may be compromised.", 
                                    self.ticker, is_error=True)
                
                # Sleep for interval
                time.sleep(self.interval_ms / 1000.0)
                
            except Exception as e:
                self.log_func(f"Error in monitoring thread for {self.ticker}: {e}", 
                            self.ticker, is_error=True)
                self.extraction_errors += 1
                time.sleep(1)  # Wait a bit before retrying
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        self.log_func(f"Monitoring thread stopped for {self.ticker}", self.ticker)
    
    def _extract_data(self) -> Optional[dict]:
        """Extract data from ToS window"""
        try:
            # Ensure window exists and is visible
            if not tos_data_grabber.is_window_valid(self.window_handle):
                self.log_func(f"Window handle invalid for {self.ticker}", 
                            self.ticker, is_error=True)
                return None
            
            extracted_data = {}
            
            # Extract ticker symbol (for verification)
            if self.coordinates.get("ticker_symbol"):
                ticker_text = tos_data_grabber.extract_text_from_coordinates(
                    self.window_handle, 
                    self.coordinates["ticker_symbol"]["x"],
                    self.coordinates["ticker_symbol"]["y"]
                )
                if ticker_text and ticker_text.strip().upper() != self.ticker:
                    self.log_func(f"Ticker mismatch: expected {self.ticker}, "
                                f"got {ticker_text}", self.ticker, is_error=True)
                extracted_data["verified_ticker"] = ticker_text
            
            # Extract option statistics
            if self.coordinates.get("option_statistics"):
                stats_coord = self.coordinates["option_statistics"]
                stats_text = tos_data_grabber.extract_text_from_region(
                    self.window_handle,
                    stats_coord["x"], stats_coord["y"],
                    stats_coord["width"], stats_coord["height"]
                )
                extracted_data["option_statistics"] = stats_text
            
            # Extract time & sales data
            if self.coordinates.get("time_sales"):
                ts_coord = self.coordinates["time_sales"]
                
                # Method 1: Try to copy data directly
                time_sales_data = tos_data_grabber.copy_data_from_region(
                    self.window_handle,
                    ts_coord["x"], ts_coord["y"],
                    ts_coord["width"], ts_coord["height"]
                )
                
                if time_sales_data:
                    extracted_data["time_sales_data"] = time_sales_data
                else:
                    # Method 2: OCR as fallback
                    ocr_data = tos_data_grabber.extract_text_from_region(
                        self.window_handle,
                        ts_coord["x"], ts_coord["y"],
                        ts_coord["width"], ts_coord["height"]
                    )
                    if ocr_data:
                        extracted_data["time_sales_data"] = ocr_data
            
            return extracted_data if extracted_data else None
            
        except Exception as e:
            self.log_func(f"Data extraction error for {self.ticker}: {e}", 
                        self.ticker, is_error=True)
            return None
    
    def get_buffered_data(self) -> list:
        """Get all buffered data"""
        data_list = []
        while not self.data_queue.empty():
            try:
                data_list.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data_list

class MonitoringManager:
    """Manages all monitoring threads"""
    
    def __init__(self, log_func: Callable):
        self.log_func = log_func
        self.active_monitors: dict[str, MonitoringThread] = {}
        self.monitoring_interval = 1000  # Default 1 second
        self.data_buffer: dict[str, list] = {}
        self.last_data_fetch: dict[str, datetime] = {}
        
    def start_monitoring(self, ticker: str, window_handle: int, 
                        coordinates: dict) -> bool:
        """Start monitoring for a ticker"""
        if ticker in self.active_monitors:
            self.log_func(f"Monitoring already active for {ticker}", ticker)
            return False
        
        try:
            # Create and start monitoring thread
            monitor_thread = MonitoringThread(
                ticker, window_handle, coordinates,
                self.monitoring_interval, self.log_func
            )
            monitor_thread.start()
            
            self.active_monitors[ticker] = monitor_thread
            self.data_buffer[ticker] = []
            self.last_data_fetch[ticker] = datetime.now()
            
            self.log_func(f"Started monitoring for {ticker}", ticker)
            return True
            
        except Exception as e:
            self.log_func(f"Failed to start monitoring for {ticker}: {e}", 
                        ticker, is_error=True)
            return False
    
    def stop_monitoring(self, ticker: str):
        """Stop monitoring for a ticker"""
        if ticker in self.active_monitors:
            self.active_monitors[ticker].stop()
            self.active_monitors[ticker].join(timeout=2)
            del self.active_monitors[ticker]
            
            # Clear buffer
            if ticker in self.data_buffer:
                del self.data_buffer[ticker]
            
            self.log_func(f"Stopped monitoring for {ticker}", ticker)
    
    def get_new_data(self, ticker: str) -> Optional[dict]:
        """Get new data for ticker since last fetch"""
        if ticker not in self.active_monitors:
            return None
        
        monitor = self.active_monitors[ticker]
        new_data_list = monitor.get_buffered_data()
        
        if new_data_list:
            # Combine time & sales data
            combined_time_sales = []
            latest_stats = None
            
            for data in new_data_list:
                if "time_sales_data" in data:
                    combined_time_sales.append(data["time_sales_data"])
                if "option_statistics" in data:
                    latest_stats = data["option_statistics"]
            
            result = {}
            if combined_time_sales:
                result["time_sales_data"] = "\n".join(combined_time_sales)
            if latest_stats:
                result["option_statistics"] = latest_stats
            
            self.last_data_fetch[ticker] = datetime.now()
            return result if result else None
        
        return None
    
    def get_buffered_data(self, ticker: str) -> Optional[str]:
        """Get all buffered data as string"""
        if ticker not in self.active_monitors:
            return None
        
        monitor = self.active_monitors[ticker]
        data_list = monitor.get_buffered_data()
        
        if data_list:
            # Extract time & sales data
            time_sales_lines = []
            for data in data_list:
                if "time_sales_data" in data:
                    time_sales_lines.append(data["time_sales_data"])
            
            return "\n".join(time_sales_lines) if time_sales_lines else None
        
        return None
    
    def get_monitoring_status(self, ticker: str) -> dict:
        """Get monitoring status for ticker"""
        if ticker not in self.active_monitors:
            return {"active": False}
        
        monitor = self.active_monitors[ticker]
        return {
            "active": monitor.is_running,
            "last_extraction": monitor.last_extraction_time,
            "errors": monitor.extraction_errors,
            "queue_size": monitor.data_queue.qsize()
        }
    
    def set_monitoring_interval(self, interval_ms: int):
        """Set monitoring interval for future monitors"""
        self.monitoring_interval = max(100, interval_ms)  # Minimum 100ms