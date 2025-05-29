# tos_data_grabber.py
"""
ThinkOrSwim window detection, data extraction, and monitoring utilities
"""

import os
import time
import re
import platform
from typing import List, Tuple, Optional, Dict, Callable, Any
from datetime import datetime
import pyperclip
from PIL import Image, ImageGrab
import pytesseract

# Windows-specific imports
PYWINAUTO_AVAILABLE = False
if platform.system() == "Windows":
    try:
        import pywinauto
        from pywinauto import Application
        import win32gui
        import win32con
        import win32api
        PYWINAUTO_AVAILABLE = True
    except ImportError:
        print("Windows automation libraries not available. Some features will be limited.")

def list_potential_tos_windows() -> list[Tuple[str, int]]:
    """list potential ThinkOrSwim windows"""
    
    if not PYWINAUTO_AVAILABLE:
        return []
    
    windows = []
    
    def enum_windows_callback(hwnd, windows_list):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title and ("thinkorswim" in window_title.lower() or 
                                "tos" in window_title.lower() or
                                "think" in window_title.lower()):
                windows_list.append((window_title, hwnd))
        return True
    
    try:
        win32gui.EnumWindows(enum_windows_callback, windows)
    except Exception as e:
        print(f"Error enumerating windows: {e}")
    
    return windows
def copy_time_sales_data(window_handle: int, coordinates: dict, log_func: Callable) -> Optional[str]:
    """
    Enhanced copy method specifically for Time & Sales data
    Ensures we click on the data area and use Ctrl+A, Ctrl+C
    """
    if not PYWINAUTO_AVAILABLE:
        return None
    
    try:
        import win32gui
        import win32api
        import win32con
        
        # Get time & sales coordinates
        if not coordinates.get("time_sales"):
            log_func("No time & sales coordinates configured", is_error=True)
            return None
        
        ts_coord = coordinates["time_sales"]
        
        # Clear clipboard first
        pyperclip.copy("")
        
        # Focus the window
        win32gui.SetForegroundWindow(window_handle)
        time.sleep(0.3)
        
        # Click in the middle of the time & sales area to ensure focus
        click_x = ts_coord["x"] + ts_coord["width"] // 2
        click_y = ts_coord["y"] + ts_coord["height"] // 2
        
        # Convert to screen coordinates
        window_rect = win32gui.GetWindowRect(window_handle)
        screen_x = window_rect[0] + click_x
        screen_y = window_rect[1] + click_y
        
        # Click to focus on the time & sales area
        win32api.SetCursorPos((screen_x, screen_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.1)
        
        # Select all data with Ctrl+A
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('A'), 0, 0, 0)
        win32api.keybd_event(ord('A'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.1)
        
        # Copy with Ctrl+C
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('C'), 0, 0, 0)
        win32api.keybd_event(ord('C'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.2)
        
        # Get clipboard content
        clipboard_content = pyperclip.paste()
        
        # Validate we got data
        if clipboard_content and len(clipboard_content) > 10:
            # Quick validation - check if it looks like time & sales data
            lines = clipboard_content.strip().split('\n')
            if lines and '\t' in lines[0]:  # Tab-delimited data
                log_func(f"Successfully copied {len(lines)} lines of time & sales data")
                return clipboard_content
        
        log_func("Failed to copy valid time & sales data", is_error=True)
        return None
        
    except Exception as e:
        log_func(f"Error copying time & sales data: {e}", is_error=True)
        return None
    
def rename_window(window_handle: int, new_title: str) -> bool:
    """Rename a window to a new title"""
    if not PYWINAUTO_AVAILABLE:
        return False
    
    try:
        import win32gui
        win32gui.SetWindowText(window_handle, new_title)
        return True
    except Exception as e:
        print(f"Error renaming window: {e}")
        return False

def extract_time_sales_incremental(window_handle: int, coordinates: dict, 
                                 last_trade_info: dict, log_func: Callable) -> Optional[dict]:
    """
    Extract Time & Sales data incrementally based on last processed trade
    """
    try:
        # Copy all current data
        raw_data = copy_time_sales_data(window_handle, coordinates, log_func)
        
        if not raw_data:
            return None
        
        # Parse the raw data to find new trades
        lines = raw_data.strip().split('\n')
        
        if not lines:
            return None
        
        new_trades = []
        last_time = last_trade_info.get('last_trade_time')
        last_hash = last_trade_info.get('last_trade_hash')
        
        found_last_trade = False
        
        for line in lines:
            parts = line.split('\t')
            if len(parts) >= 9:  # Valid trade line
                trade_time = parts[0].strip()
                
                # Create hash for this trade
                trade_hash = _create_trade_hash(parts)
                
                if last_time and last_hash:
                    # Skip until we find the last processed trade
                    if not found_last_trade:
                        if trade_time == last_time and trade_hash == last_hash:
                            found_last_trade = True
                        continue
                
                # This is a new trade
                new_trades.append(line)
        
        if new_trades:
            return {
                'new_trades': '\n'.join(new_trades),
                'total_new_trades': len(new_trades),
                'extraction_time': datetime.now()
            }
        else:
            return {
                'new_trades': '',
                'total_new_trades': 0,
                'extraction_time': datetime.now()
            }
            
    except Exception as e:
        log_func(f"Error in incremental extraction: {e}", is_error=True)
        return None

def _create_trade_hash(trade_parts: list) -> str:
    """Create a unique hash for a trade based on its components"""
    # Use time, option description, quantity, and price for hash
    if len(trade_parts) >= 4:
        hash_string = f"{trade_parts[0]}_{trade_parts[1]}_{trade_parts[2]}_{trade_parts[3]}"
        return str(hash(hash_string))
    return ""

def validate_tos_window_for_ticker(window_handle: int, expected_ticker: str, 
                                 coordinates: dict, log_func: Callable) -> bool:
    """
    Validate that the ToS window is showing data for the expected ticker
    """
    try:
        # Try to extract ticker from the window
        if coordinates.get("ticker_symbol"):
            ticker_coords = coordinates["ticker_symbol"]
            detected_ticker = extract_text_from_coordinates(
                window_handle, 
                ticker_coords["x"], 
                ticker_coords["y"]
            )
            
            if detected_ticker:
                detected_clean = re.sub(r'[^A-Z]', '', detected_ticker.upper())
                if detected_clean == expected_ticker:
                    log_func(f"Window validated for {expected_ticker}")
                    return True
                else:
                    log_func(f"Window shows {detected_clean}, expected {expected_ticker}", is_error=True)
                    return False
        
        # If we can't detect ticker, check window title
        window_title = get_window_title(window_handle)
        if window_title and expected_ticker in window_title:
            log_func(f"Window title matches {expected_ticker}")
            return True
        
        log_func(f"Could not validate window for {expected_ticker}", is_error=True)
        return False
        
    except Exception as e:
        log_func(f"Error validating window: {e}", is_error=True)
        return False

def find_tos_window_by_title(ticker: str) -> Optional[int]:
    """Find a ToS window by ticker in title"""
    if not PYWINAUTO_AVAILABLE:
        return None
    
    windows = []
    
    def enum_callback(hwnd, results):
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title and ticker in window_title:
                results.append(hwnd)
        return True
    
    try:
        win32gui.EnumWindows(enum_callback, windows)
        
        if windows:
            return windows[0]  # Return first matching window
    except Exception as e:
        print(f"Error finding window by title: {e}")
    
    return None

def ensure_time_sales_visible(window_handle: int, coordinates: dict, log_func: Callable) -> bool:
    """
    Ensure the Time & Sales section is visible and ready for data extraction
    Sometimes the data area needs to be clicked or scrolled
    """
    try:
        if not coordinates.get("time_sales"):
            return False
        
        ts_coord = coordinates["time_sales"]
        
        # Focus window
        win32gui.SetForegroundWindow(window_handle)
        time.sleep(0.2)
        
        # Click at top of time & sales area
        click_x = ts_coord["x"] + 50  # Offset from left edge
        click_y = ts_coord["y"] + 20  # Near top
        
        window_rect = win32gui.GetWindowRect(window_handle)
        screen_x = window_rect[0] + click_x
        screen_y = window_rect[1] + click_y
        
        win32api.SetCursorPos((screen_x, screen_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
        # Small scroll to ensure fresh data is visible
        win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, 120, 0)  # Scroll up
        time.sleep(0.1)
        
        log_func("Time & Sales area prepared for extraction")
        return True
        
    except Exception as e:
        log_func(f"Error preparing Time & Sales area: {e}", is_error=True)
        return False
    

def resize_tos_window(window_handle: int, log_func: Callable, ticker: str = "ToS"):
    """Resize ToS window to target dimensions"""
    
    if not PYWINAUTO_AVAILABLE:
        log_func("Windows automation not available", ticker, is_error=True)
        return False
    
    try:
        # Get window rect
        rect = win32gui.GetWindowRect(window_handle)
        current_width = rect[2] - rect[0]
        current_height = rect[3] - rect[1]
        
        # Import config here to avoid circular imports
        import config
        target_width = config.USER_TOS_TARGET_WINDOW_WIDTH
        target_height = config.USER_TOS_TARGET_WINDOW_HEIGHT
        
        if current_width != target_width or current_height != target_height:
            # Move and resize window
            win32gui.SetWindowPos(
                window_handle, 
                win32con.HWND_TOP,
                100, 100, target_width, target_height,
                win32con.SWP_SHOWWINDOW
            )
            
            time.sleep(0.5)  # Allow window to resize
            log_func(f"Resized ToS window to {target_width}x{target_height}", ticker)
            return True
        else:
            log_func("ToS window already at target size", ticker)
            return True
            
    except Exception as e:
        log_func(f"Error resizing ToS window: {e}", ticker, is_error=True)
        return False

def detect_ticker_from_tos_window(window_handle: int, coordinates_config: dict, 
                                 log_func: Callable) -> Optional[str]:
    """Detect ticker symbol from ToS window"""
    
    if not coordinates_config or not coordinates_config.get("ticker_symbol"):
        log_func("No ticker coordinates configured", is_error=True)
        return None
    
    try:
        ticker_coords = coordinates_config["ticker_symbol"]
        x, y = ticker_coords["x"], ticker_coords["y"]
        
        # Extract text from ticker location
        ticker_text = extract_text_from_coordinates(window_handle, x, y)
        
        if ticker_text:
            # Clean and validate ticker
            ticker_clean = re.sub(r'[^A-Z]', '', ticker_text.upper())
            if 1 <= len(ticker_clean) <= 5:  # Valid ticker length
                return ticker_clean
        
        return None
        
    except Exception as e:
        log_func(f"Error detecting ticker: {e}", is_error=True)
        return None

def extract_text_from_coordinates(window_handle: int, x: int, y: int, 
                                width: int = 100, height: int = 30) -> Optional[str]:
    """Extract text from specific coordinates"""
    
    try:
        # Capture region around coordinates
        screenshot = capture_window_screenshot(window_handle)
        if not screenshot:
            return None
        
        # Crop to specific region
        region = screenshot.crop((x, y, x + width, y + height))
        
        # OCR extraction
        text = pytesseract.image_to_string(region, config='--psm 8').strip()
        
        return text if text else None
        
    except Exception as e:
        print(f"Error extracting text from coordinates: {e}")
        return None

def extract_text_from_region(window_handle: int, x: int, y: int, 
                           width: int, height: int) -> Optional[str]:
    """Extract text from a rectangular region"""
    
    try:
        screenshot = capture_window_screenshot(window_handle)
        if not screenshot:
            return None
        
        # Crop to region
        region = screenshot.crop((x, y, x + width, y + height))
        
        # OCR with table/data configuration
        text = pytesseract.image_to_string(region, config='--psm 6').strip()
        
        return text if text else None
        
    except Exception as e:
        print(f"Error extracting text from region: {e}")
        return None

def copy_data_from_region(window_handle: int, x: int, y: int, 
                         width: int, height: int) -> Optional[str]:
    """Try to copy data from region using Ctrl+C"""
    
    if not PYWINAUTO_AVAILABLE:
        return None

    try:
        # Import win32gui here to ensure it's available
        import win32gui
        import win32api
        import win32con

        # Focus the window
        win32gui.SetForegroundWindow(window_handle)
        time.sleep(0.2)
        
        # Click on the region to select
        click_x = x + width // 2
        click_y = y + height // 2
        
        # Convert to screen coordinates
        window_rect = win32gui.GetWindowRect(window_handle)
        screen_x = window_rect[0] + click_x
        screen_y = window_rect[1] + click_y
        
        # Click and select
        win32api.SetCursorPos((screen_x, screen_y))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        
        time.sleep(0.1)
        
        # Try Ctrl+A then Ctrl+C
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('A'), 0, 0, 0)
        win32api.keybd_event(ord('A'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        
        time.sleep(0.1)
        
        win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
        win32api.keybd_event(ord('C'), 0, 0, 0)
        win32api.keybd_event(ord('C'), 0, win32con.KEYEVENTF_KEYUP, 0)
        win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        
        time.sleep(0.2)
        
        # Get clipboard content
        clipboard_content = pyperclip.paste()
        
        return clipboard_content if clipboard_content else None
        
    except Exception as e:
        print(f"Error copying data from region: {e}")
        return None

def capture_window_screenshot(window_handle: int) -> Optional[Image.Image]:
    """Capture screenshot of specific window"""
    
    if not PYWINAUTO_AVAILABLE:
        return None
    
    try:
        # Get window rectangle
        rect = win32gui.GetWindowRect(window_handle)
        
        # Capture the window area
        screenshot = ImageGrab.grab(bbox=rect)
        
        return screenshot
        
    except Exception as e:
        print(f"Error capturing window screenshot: {e}")
        return None

def is_window_valid(window_handle: int) -> bool:
    """Check if window handle is still valid"""
    
    if not PYWINAUTO_AVAILABLE:
        return False
    
    try:
        return win32gui.IsWindow(window_handle) and win32gui.IsWindowVisible(window_handle)
    except:
        return False

def get_window_title(window_handle: int) -> Optional[str]:
    """Get window title"""
    
    if not PYWINAUTO_AVAILABLE:
        return None
    
    try:
        return win32gui.GetWindowText(window_handle)
    except:
        return None

def monitor_tos_window_data(window_handle: int, coordinates_config: dict, 
                           log_func: Callable, ticker: str) -> dict:
    """Monitor and extract data from ToS window"""
    
    extracted_data = {
        'timestamp': datetime.now(),
        'ticker_symbol': None,
        'option_statistics': None,
        'time_sales_data': None,
        'extraction_success': False
    }
    
    if not coordinates_config:
        log_func("No coordinates configuration available", ticker, is_error=True)
        return extracted_data
    
    try:
        # Extract ticker symbol
        if coordinates_config.get("ticker_symbol"):
            ticker_coords = coordinates_config["ticker_symbol"]
            ticker_text = extract_text_from_coordinates(
                window_handle, ticker_coords["x"], ticker_coords["y"]
            )
            extracted_data['ticker_symbol'] = ticker_text
        
        # Extract option statistics
        if coordinates_config.get("option_statistics"):
            stats_coords = coordinates_config["option_statistics"]
            stats_text = extract_text_from_region(
                window_handle,
                stats_coords["x"], stats_coords["y"],
                stats_coords["width"], stats_coords["height"]
            )
            extracted_data['option_statistics'] = stats_text
        
        # Extract time & sales data
        if coordinates_config.get("time_sales"):
            ts_coords = coordinates_config["time_sales"]
            
            # Try copy method first
            ts_data = copy_data_from_region(
                window_handle,
                ts_coords["x"], ts_coords["y"],
                ts_coords["width"], ts_coords["height"]
            )
            
            # Fallback to OCR if copy failed
            if not ts_data:
                ts_data = extract_text_from_region(
                    window_handle,
                    ts_coords["x"], ts_coords["y"],
                    ts_coords["width"], ts_coords["height"]
                )
            
            extracted_data['time_sales_data'] = ts_data
        
        # Check if extraction was successful
        extracted_data['extraction_success'] = any([
            extracted_data['ticker_symbol'],
            extracted_data['option_statistics'],
            extracted_data['time_sales_data']
        ])
        
        if extracted_data['extraction_success']:
            log_func("Data extraction successful", ticker)
        else:
            log_func("No data extracted", ticker, is_error=True)
        
        return extracted_data
        
    except Exception as e:
        log_func(f"Error monitoring ToS window: {e}", ticker, is_error=True)
        return extracted_data

def validate_extraction_coordinates(coordinates_config: dict) -> bool:
    """Validate that coordinates configuration is complete"""
    
    if not coordinates_config:
        return False
    
    required_sections = ["ticker_symbol", "option_statistics", "time_sales"]
    
    for section in required_sections:
        if section not in coordinates_config:
            return False
        
        coords = coordinates_config[section]
        
        if section == "ticker_symbol":
            # Point coordinates
            if not all(key in coords for key in ["x", "y"]):
                return False
        else:
            # Region coordinates
            if not all(key in coords for key in ["x", "y", "width", "height"]):
                return False
    
    return True

def setup_tos_monitoring(ticker: str, window_handle: int, coordinates_config: dict,
                        log_func: Callable) -> bool:
    """Set up ToS monitoring for a ticker"""
    
    try:
        # Validate window
        if not is_window_valid(window_handle):
            log_func("Invalid window handle", ticker, is_error=True)
            return False
        
        # Validate coordinates
        if not validate_extraction_coordinates(coordinates_config):
            log_func("Invalid coordinates configuration", ticker, is_error=True)
            return False
        
        # Resize window
        if not resize_tos_window(window_handle, log_func, ticker):
            log_func("Failed to resize window", ticker, is_error=True)
            return False
        
        # Test data extraction
        test_data = monitor_tos_window_data(window_handle, coordinates_config, log_func, ticker)
        
        if not test_data['extraction_success']:
            log_func("Test data extraction failed", ticker, is_error=True)
            return False
        
        log_func("ToS monitoring setup successful", ticker)
        return True
        
    except Exception as e:
        log_func(f"Error setting up ToS monitoring: {e}", ticker, is_error=True)
        return False

def get_tos_window_info(window_handle: int) -> dict:
    """Get detailed information about ToS window"""
    
    info = {
        'handle': window_handle,
        'title': None,
        'rect': None,
        'is_valid': False,
        'is_visible': False,
        'size': None
    }
    
    if not PYWINAUTO_AVAILABLE:
        return info
    
    try:
        info['is_valid'] = win32gui.IsWindow(window_handle)
        info['is_visible'] = win32gui.IsWindowVisible(window_handle)
        info['title'] = win32gui.GetWindowText(window_handle)
        
        if info['is_valid']:
            rect = win32gui.GetWindowRect(window_handle)
            info['rect'] = rect
            info['size'] = (rect[2] - rect[0], rect[3] - rect[1])
        
    except Exception as e:
        print(f"Error getting window info: {e}")
    
    return info

# Test and utility functions

def test_coordinates_extraction(window_handle: int, coordinates_config: dict) -> dict:
    """Test coordinate extraction without monitoring"""
    
    results = {}
    
    for section, coords in coordinates_config.items():
        try:
            if section == "ticker_symbol":
                text = extract_text_from_coordinates(
                    window_handle, coords["x"], coords["y"]
                )
                results[section] = {
                    'success': text is not None,
                    'extracted_text': text,
                    'length': len(text) if text else 0
                }
            else:
                text = extract_text_from_region(
                    window_handle,
                    coords["x"], coords["y"],
                    coords["width"], coords["height"]
                )
                results[section] = {
                    'success': text is not None,
                    'extracted_text': text[:200] if text else None,  # First 200 chars
                    'length': len(text) if text else 0
                }
        except Exception as e:
            results[section] = {
                'success': False,
                'error': str(e),
                'extracted_text': None,
                'length': 0
            }
    
    return results

def find_tos_processes() -> list[dict]:
    """Find running ToS processes"""
    
    processes = []
    
    if not PYWINAUTO_AVAILABLE:
        return processes
    
    try:
        import psutil
        
        for proc in psutil.process_iter(['pid', 'name', 'exe']):
            try:
                proc_info = proc.info
                if proc_info['name'] and 'thinkorswim' in proc_info['name'].lower():
                    processes.append({
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'exe': proc_info['exe']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        print("psutil not available for process detection")
    
    return processes

def get_system_info() -> dict:
    """Get system information for troubleshooting"""
    
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'pywinauto_available': PYWINAUTO_AVAILABLE,
        'tesseract_available': False
    }
    
    # Test tesseract
    try:
        pytesseract.get_tesseract_version()
        info['tesseract_available'] = True
    except:
        info['tesseract_available'] = False
    
    return info