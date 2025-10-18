import pyautogui
import pyperclip
from pynput.mouse import Controller, Button
import csv
import time
import subprocess
import platform
import os

# Platform-specific imports
if platform.system() == 'Darwin':  # macOS
    from Quartz.CoreGraphics import CGEventCreateMouseEvent, CGEventPost
    from Quartz.CoreGraphics import kCGHIDEventTap
    from Quartz.CoreGraphics import kCGEventMouseMoved, kCGEventLeftMouseDown, kCGEventLeftMouseUp
    from Quartz.CoreGraphics import kCGMouseButtonLeft

def windows_click(x, y):
    """Windows-specific implementation of mouse click."""
    import win32api, win32con
    # Convert to integer coordinates
    x = int(x)
    y = int(y)
    # Move mouse
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    # Click
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)

def macos_click(x, y):
    """macOS-specific implementation of mouse click using Quartz."""
    # Mouse move
    move = CGEventCreateMouseEvent(None, kCGEventMouseMoved, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, move)
    time.sleep(0.2)

    # Left click down
    click_down = CGEventCreateMouseEvent(None, kCGEventLeftMouseDown, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, click_down)
    time.sleep(0.02)

    # Left click up
    click_up = CGEventCreateMouseEvent(None, kCGEventLeftMouseUp, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, click_up)

def os_click(x, y):
    """Cross-platform click implementation that delegates to the appropriate OS-specific function."""
    if platform.system() == 'Darwin':  # macOS
        macos_click(x, y)
    else:  # Windows
        windows_click(x, y)

def windows_focus_window(window_name):
    """Windows-specific implementation of window focusing."""
    import win32gui
    def _window_enum_callback(hwnd, wildcard):
        if str(wildcard).lower() in str(win32gui.GetWindowText(hwnd)).lower():
            win32gui.SetForegroundWindow(hwnd)
            return True
    try:
        win32gui.EnumWindows(_window_enum_callback, window_name)
    except Exception as e:
        raise RuntimeError(f"Warning: Could not focus window '{window_name}'. Error: {e}")
        return False
    return True

def macos_focus_window(window_name):
    """macOS-specific implementation of window focusing using AppleScript."""
    script_check = f"""
    tell application "System Events"
        count (every process whose name is "{window_name}")
    end tell
    """
    result = subprocess.run(["osascript", "-e", script_check], capture_output=True, text=True)
    if result.stdout.strip() == "0":
        raise RuntimeError(f"Warning: The specified window '{window_name}' does not exist. Please open the game window through Steam first.")
        return False

    script_focus = f"""
    tell application "System Events"
        set frontmost of the first application process whose name is "{window_name}" to true
    end tell
    """
    subprocess.run(["osascript", "-e", script_focus])
    return True

def focus_window(window_name):
    """Cross-platform window focusing that delegates to the appropriate OS-specific function."""
    if platform.system() == 'Darwin':  # macOS
        return macos_focus_window(window_name)
    else:  # Windows
        return windows_focus_window(window_name)

def click_fractional_position(fraction_xy: tuple, debug: bool = False):
    """Click at a fractional position of the current screen."""
    fraction_x, fraction_y = fraction_xy
    screen_width, screen_height = pyautogui.size()
    x = int(screen_width * fraction_x)
    y = int(screen_height * fraction_y)
    if debug:   
        print(f"Clicking at: ({x}, {y}) - fractional: ({fraction_x:.3f}, {fraction_y:.3f})")
    os_click(x, y)

def wait(seconds: float):
    """Wait for a given number of seconds"""
    time.sleep(seconds)

def select_all():
    """Select all"""
    if platform.system() == 'Darwin':
        pyautogui.hotkey("command", "a")
    else:
        pyautogui.hotkey("ctrl", "a")

def copy_text():
    """Select all and copy text"""
    select_all()
    time.sleep(0.2)  # Small delay to ensure text is selected
    if platform.system() == 'Darwin':
        pyautogui.hotkey("command", "c")
        time.sleep(0.2)  # Small delay to ensure text is copied
    else:
        pyautogui.hotkey("ctrl", "c")
        time.sleep(0.2)  # Small delay to ensure text is copied
    return pyperclip.paste()

def save_to_csv(text, output_file: str):
    """Append text to a CSV file without quotes at the beginning and end."""
    with open(output_file, "a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        for line in text.splitlines():
            row = [item.strip() for item in line.split(",")]
            writer.writerow(row)

def delete_text(num: int):
    """Delete text"""
    for i in range(num):
        pyautogui.press("backspace")
        time.sleep(0.01)

def input_text(text: str):
    """Input text"""
    pyautogui.typewrite(text, interval=0)
    time.sleep(0.1 * len(text))

def activate_lua_script():
    """Activate the Lua script"""
    pyautogui.hotkey("ctrl", "l")

def deactivate_lua_script():
    """Deactivate the Lua script"""
    pyautogui.hotkey("ctrl", "l")

def start_simulation():
    """Start the simulation"""
    pyautogui.press("space")

def stop_simulation():
    """Stop the simulation"""
    pyautogui.press("space")

def run_simulation_sequence(machine_name: str, output_dir: str, duration: float, set_ground: bool = False):
    """
    Complete simulation sequence as specified:
    0. make sure the keyboard is english and no cap lock
    1. click Open Folder, wait 1s
    2. click Enter Name frame
    3. select all
    4. delete all (hit backspace twice would do that)
    5. Typewrite the given name
    6. click Open Machine, wait 1s
    7. activate lua
    8. hit space to start simulation, starting timing
    9. click Log Window
    10. repeat the operation every 2s: select all, copy content, save to csv
    11. wait until simulation timing reached given duration
    12. select all, copy, save one last time
    13. click Empty Space
    14. deactivate lua
    15. clean csv file
    16. click Delete
    17. click Confirm
    """
    from config import (POS_OPEN_FOLDER, POS_ENTER_NAME, POS_OPEN_MACHINE, 
                       POS_LOG_WINDOW, POS_EMPTY_SPACE, POS_START_SIMU, POS_DELETE, POS_CONFIRM, POS_SET_GROUND)
    
    # Create output file path
    csv_file = f"{output_dir}/simulation_log_{machine_name}.csv"

    # Initialize clean CSV file
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    print(f"Starting simulation sequence for machine: {machine_name}")
    print(f"Duration: {duration} seconds")
    print(f"Output CSV: {csv_file}")
    
    # Step 0: Ensure keyboard setup and focus on Besiege window
    focus_window("Besiege")
    wait(1.0)
    
    # Step 1: Click Open Folder, wait 1s
    print("Step 1: Clicking Open Folder...")
    click_fractional_position(POS_OPEN_FOLDER)
    wait(1.0)
    
    # Step 2: Click Enter Name frame
    print("Step 2: Clicking Enter Name frame...")
    click_fractional_position(POS_ENTER_NAME)
    wait(1.0)
    
    # Step 3: Select all
    print("Step 3: Selecting all text...")
    select_all()
    wait(0.2)
    
    # Step 4: Delete all (hit backspace twice)
    print("Step 4: Deleting text...")
    delete_text(2)
    wait(0.2)
    
    # Step 5: Type the machine name
    print(f"Step 5: Typing machine name: {machine_name}")
    input_text(machine_name)
    wait(1.0)
    
    # Step 6: Click Open Machine, wait 1s
    print("Step 6: Clicking Open Machine...")
    click_fractional_position(POS_OPEN_MACHINE)
    wait(3.0)

    if set_ground:
        print("Step 6: Clicking Set Ground...")
        click_fractional_position(POS_SET_GROUND)
        wait(0.5)
    
    # Step 7: Activate Lua
    print("Step 7: Activating Lua script...")
    activate_lua_script()
    wait(0.5)
    
    # Step 8: Hit space to start simulation, start timing
    print("Step 8: Starting simulation...")
    start_time = time.time()
    click_fractional_position(POS_START_SIMU)
    wait(0.5)
    
    # Step 9: Click Log Window
    print("Step 9: Clicking Log Window...")
    click_fractional_position(POS_LOG_WINDOW)
    wait(0.5)
    
    # Step 10: Repeat logging every 2s until duration is reached
    print(f"Step 10: Starting periodic logging every 1 seconds for {duration} seconds...")

    # Copy log content
    click_fractional_position(POS_LOG_WINDOW)
    log_text = copy_text()
    
    # Save to CSV with timestamp
    save_to_csv(log_text, csv_file)
    
    last_log_time = start_time
    while (time.time() - start_time) < duration:
        current_time = time.time()
        
        # Log every 2 seconds
        if (current_time - last_log_time) >= 1.0:
            
            # Copy log content
            click_fractional_position(POS_LOG_WINDOW)
            log_text = copy_text()
            
            # Save to CSV with timestamp
            save_to_csv(log_text, csv_file)
            
            last_log_time = current_time
    
    # Step 11: Stop simulation
    print("Step 11: Stopping simulation...")
    click_fractional_position(POS_START_SIMU)
    wait(0.5)

    # Step 12: Select all, copy, save one last time
    print("Step 12: Final log capture...")
    click_fractional_position(POS_LOG_WINDOW)
    wait(0.5)
    final_log_text = copy_text()
    
    save_to_csv(final_log_text, csv_file)
    
    # Step 13: Click Empty Space
    print("Step 13: Clicking Empty Space...")
    click_fractional_position(POS_EMPTY_SPACE)
    wait(0.5)
    
    # Step 14: Deactivate Lua
    print("Step 14: Deactivating Lua script...")
    deactivate_lua_script()
    wait(0.5)

    # Step 15: Clean CSV file
    print("Step 15: Cleaning CSV file...")
    clean_simulation_csv(csv_file)
    wait(0.5)
    
    # Step 16: Click Delete
    print("Step 16: Clicking Delete...")
    click_fractional_position(POS_DELETE)
    wait(0.5)
    
    # Step 17: Click Confirm
    print("Step 17: Clicking Confirm...")
    click_fractional_position(POS_CONFIRM)
    wait(0.5)
    
    print(f"Simulation sequence completed! Log saved to: {csv_file}")

    return csv_file

def clean_simulation_csv(csv_file_path: str, output_file_path: str = None):
    """
    Clean up the simulation CSV file by:
    1. Deduplicating rows based on all columns (keep only unique rows)
    2. Removing rows with only one element
    3. Reversing the time order (ascending instead of descending)
    
    Args:
        csv_file_path: Path to the input CSV file
        output_file_path: Path to the output CSV file (if None, overwrites input file)
    """
    if output_file_path is None:
        output_file_path = csv_file_path
    
    # Read and parse the CSV data
    rows = []
    seen_rows = set()
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip empty rows
            if not row:
                continue
            
            # Remove rows with only one element
            if len(row) < 2:
                continue

            # Remove rows that end with invalid characters
            if row[-1].strip().endswith(("-", ".", ",")):
                continue
            
            # Create a tuple of all columns for deduplication
            try:                
                # Create a tuple key using all columns for deduplication
                row_key = tuple(row)
                
                # Skip if we've already seen this exact row
                if row_key in seen_rows:
                    continue
                
                seen_rows.add(row_key)
                rows.append(row)
                
            except (ValueError, IndexError):
                # Skip rows where timestamp can't be converted to float
                continue
    
    # Sort by timestamp in ascending order (normal sequence)
    rows.sort(key=lambda x: float(x[0]))
    
    # Write the cleaned data
    with open(output_file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow(row)
    
    print(f"CSV cleaned: {len(rows)} unique rows written to {output_file_path}")
    return len(rows)

def main():
    """Main function to run the simulation with command line arguments"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Run Besiege simulation sequence')
    parser.add_argument('--name', required=True, help='Machine name to load')
    parser.add_argument('--output_dir', required=True, help='Output directory for logs')
    parser.add_argument('--duration', type=float, required=True, help='Simulation duration in seconds')
    parser.add_argument('--window_name', default='Besiege', help='Name of the Besiege window to focus')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Focus on Besiege window first
    print(f"Focusing on window: {args.window_name}")
    if not focus_window(args.window_name):
        raise RuntimeError(f"Failed to focus window '{args.window_name}'. Please make sure Besiege is running.")
        return
    
    wait(1.0)  # Give time for window to focus
    
    # Run the simulation sequence
    try:
        csv_file = run_simulation_sequence(args.name, args.output_dir, args.duration)
    except KeyboardInterrupt:
        raise KeyboardInterrupt("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise

if __name__ == "__main__":
    main()
