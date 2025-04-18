import time
import json
import keyboard
import threading
import pyautogui

from connection.SocketConnection import SocketConnection

# Configuration
connection = SocketConnection()
T = 0.1  # Time between simulator data packets
BASE_POWER = 50  # Initial general power for all motors
POWER_STEP = 5   # Power adjustment per keypress
MOUSE_POLL_RATE = 0.05  # How often to check mouse position
MOUSE_SENSITIVITY = 0.1  # How sensitive the mouse control is
RETURN_RATE = 1  # How quickly motors return to general power when not actively controlled

# Global variables
keep_running = True
general_power = BASE_POWER  # General power level for all motors
motor_powers = {
    "fr": BASE_POWER,  # Front right
    "fl": BASE_POWER,  # Front left
    "br": BASE_POWER,  # Back right
    "bl": BASE_POWER,  # Back left
    "rf": BASE_POWER,  # Right front
    "rb": BASE_POWER,  # Right back
    "lf": BASE_POWER,  # Left front
    "lb": BASE_POWER   # Left back
}
active_control = {  # Flags to track active control inputs
    "roll_left": False,
    "roll_right": False,
    "pitch_up": False,
    "pitch_down": False,
    "yaw_left": False,
    "yaw_right": False
}

# Screen center coordinates for relative mouse positioning
screen_width, screen_height = pyautogui.size()
center_x, center_y = screen_width // 2, screen_height // 2
prev_mouse_x, prev_mouse_y = center_x, center_y

def get_data(str_data: str):
    """Parse data from simulator"""
    data = json.loads(str_data)["dronesData"]
    return data

def concat_engines(engines, t):
    """Format all drone data for simulator"""
    result = {
        "drones": engines,
        "returnTimer": 1000*t,
    }
    return json.dumps(result)

def create_drone_data(drone_id):
    """Create drone control data based on current motor powers"""
    global motor_powers

    return {
        "id": drone_id,
        "engines": motor_powers,
        "dropExtinguisher": False
    }

def reset_active_controls():
    """Reset all active control flags"""
    global active_control
    for control in active_control:
        active_control[control] = False

def return_to_general_power():
    """Gradually return all motors to general power when not actively controlled"""
    global motor_powers, general_power

    for motor in motor_powers:
        if motor_powers[motor] > general_power:
            motor_powers[motor] -= min(RETURN_RATE, motor_powers[motor] - general_power)
        elif motor_powers[motor] < general_power:
            motor_powers[motor] += min(RETURN_RATE, general_power - motor_powers[motor])

def apply_roll(direction):
    """Apply roll control (around forward axis) - A/D keys"""
    global motor_powers, general_power

    if direction == "left":  # Roll left (A key)
        # Increase power to right motors, decrease to left motors
        motor_powers["fr"] = general_power + POWER_STEP
        motor_powers["br"] = general_power + POWER_STEP
        motor_powers["rf"] = general_power + POWER_STEP
        motor_powers["rb"] = general_power + POWER_STEP

        motor_powers["fl"] = general_power - POWER_STEP
        motor_powers["bl"] = general_power - POWER_STEP
        motor_powers["lf"] = general_power - POWER_STEP
        motor_powers["lb"] = general_power - POWER_STEP
        print("Rolling left (around forward axis)")

    elif direction == "right":  # Roll right (D key)
        # Increase power to left motors, decrease to right motors
        motor_powers["fr"] = general_power - POWER_STEP
        motor_powers["br"] = general_power - POWER_STEP
        motor_powers["rf"] = general_power - POWER_STEP
        motor_powers["rb"] = general_power - POWER_STEP

        motor_powers["fl"] = general_power + POWER_STEP
        motor_powers["bl"] = general_power + POWER_STEP
        motor_powers["lf"] = general_power + POWER_STEP
        motor_powers["lb"] = general_power + POWER_STEP
        print("Rolling right (around forward axis)")

def apply_pitch(direction):
    """Apply pitch control (mouse up/down)"""
    global motor_powers, general_power

    if direction == "up":  # Pitch up (mouse down)
        # Increase power to back motors, decrease to front motors
        motor_powers["br"] = general_power + POWER_STEP
        motor_powers["bl"] = general_power + POWER_STEP
        motor_powers["rb"] = general_power + POWER_STEP
        motor_powers["lb"] = general_power + POWER_STEP

        motor_powers["fr"] = general_power - POWER_STEP
        motor_powers["fl"] = general_power - POWER_STEP
        motor_powers["rf"] = general_power - POWER_STEP
        motor_powers["lf"] = general_power - POWER_STEP
        print("Pitching up (nose up)")

    elif direction == "down":  # Pitch down (mouse up)
        # Increase power to front motors, decrease to back motors
        motor_powers["fr"] = general_power + POWER_STEP
        motor_powers["fl"] = general_power + POWER_STEP
        motor_powers["rf"] = general_power + POWER_STEP
        motor_powers["lf"] = general_power + POWER_STEP

        motor_powers["br"] = general_power - POWER_STEP
        motor_powers["bl"] = general_power - POWER_STEP
        motor_powers["rb"] = general_power - POWER_STEP
        motor_powers["lb"] = general_power - POWER_STEP
        print("Pitching down (nose down)")

def apply_yaw(direction):
    """Apply yaw control (mouse left/right)"""
    global motor_powers, general_power

    ccw_motors  = ["bl","fl","lf","lb"]  # 1, 3, 5, 7
    cw_motors = ["br","fr","lf","rf"]  # 2, 4, 6, 8

    #ccw_motors  = ["fr","bl"]  # 1, 3, 5, 7
    #cw_motors = ["fl","br"]  # 2, 4, 6, 8
    if direction == "left":
        for m in ccw_motors:
            motor_powers[m] = general_power + POWER_STEP*5
            motor_powers[m] = 100
        for m in cw_motors:
            motor_powers[m] = general_power - POWER_STEP*5
            motor_powers[m] = 0
        print("Yawing left (CCW)")

    elif direction == "right":
        for m in cw_motors:
            motor_powers[m] = general_power + POWER_STEP*5
            motor_powers[m] = 100
        for m in ccw_motors:
            motor_powers[m] = general_power - POWER_STEP*5
            motor_powers[m] = 0
        print("Yawing right (CW)")


def control_listener():
    """Listen for mouse and keyboard input and adjust motor powers"""
    global motor_powers, general_power, keep_running, prev_mouse_x, prev_mouse_y, active_control

    print("Mouse and Keyboard Controls:")
    print("W: Increase general power for all motors")
    print("S: Decrease general power for all motors")
    print("A: Roll left (around forward axis)")
    print("D: Roll right (around forward axis)")
    print("Mouse left/right: Yaw control (turn left/right)")
    print("Mouse up/down: Pitch control (nose down/up)")
    print("R: Reset all motors to base power")
    print("C: Center mouse without affecting control")
    print("Space: Emergency stop (all motors to zero)")
    print("ESC: Exit")

    # Initial centering of mouse
    pyautogui.moveTo(center_x, center_y)
    prev_mouse_x, prev_mouse_y = center_x, center_y

    while keep_running:
        # Reset active controls at start of each cycle
        reset_active_controls()

        # Get current mouse position
        mouse_x, mouse_y = pyautogui.position()

        # Calculate mouse movement since last check
        mouse_delta_x = mouse_x - prev_mouse_x
        mouse_delta_y = mouse_y - prev_mouse_y
        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

        # Apply mouse control for yaw (left/right)
        if abs(mouse_delta_x) > 3:
            if mouse_delta_x < 0:
                apply_yaw("left")
                active_control["yaw_left"] = True
            else:
                apply_yaw("right")
                active_control["yaw_right"] = True

        # Apply mouse control for pitch (up/down)
        if abs(mouse_delta_y) > 3:
            if mouse_delta_y < 0:
                apply_pitch("down")  # Mouse moving up = pitch down (nose down)
                active_control["pitch_down"] = True
            else:
                apply_pitch("up")  # Mouse moving down = pitch up (nose up)
                active_control["pitch_up"] = True

        # Roll control (A/D) - roll around forward axis
        if keyboard.is_pressed('a'):
            apply_roll("left")
            active_control["roll_left"] = True
        if keyboard.is_pressed('d'):
            apply_roll("right")
            active_control["roll_right"] = True

        # General power control (W/S)
        general_power=50
        if keyboard.is_pressed('w'):
            general_power = 100
            if general_power > 100:
                general_power = 100
            print(f"General power increased to: {general_power}")
            time.sleep(0.05)  # Prevent too rapid changes

        if keyboard.is_pressed('s'):
            general_power = 20
            if general_power < 0:
                general_power = 0
            print(f"General power decreased to: {general_power}")
            time.sleep(0.05)  # Prevent too rapid changes

        # Return motors to general power when not actively controlled
        if not any(active_control.values()):
            return_to_general_power()

        # Center mouse without affecting control (C)
        if keyboard.is_pressed('c'):
            pyautogui.moveTo(center_x, center_y)
            prev_mouse_x, prev_mouse_y = center_x, center_y
            print("Mouse centered")
            time.sleep(0.2)  # Prevent multiple triggers

        # Reset control (R)
        if keyboard.is_pressed('r'):
            general_power = BASE_POWER
            for motor in motor_powers:
                motor_powers[motor] = general_power
            print(f"All motors reset to base power: {BASE_POWER}")
            # Center mouse when resetting
            pyautogui.moveTo(center_x, center_y)
            prev_mouse_x, prev_mouse_y = center_x, center_y
            time.sleep(0.2)  # Prevent multiple triggers

        # Emergency stop (Space)
        if keyboard.is_pressed('space'):
            general_power = 0
            for motor in motor_powers:
                motor_powers[motor] = 0
            print("EMERGENCY STOP - ALL MOTORS OFF")

        # Ensure motors don't go below 0 or above 100
        for motor in motor_powers:
            if motor_powers[motor] < -1000:
                motor_powers[motor] = 0
            elif motor_powers[motor] > 100:
                motor_powers[motor] = 100

        # Exit control (ESC)
        if keyboard.is_pressed('esc'):
            keep_running = False
            print("Exiting control...")

        time.sleep(MOUSE_POLL_RATE)

def next_step():
    """Process one step of drone movement"""
    data = get_data(connection.receive_data())
    result = []

    for drone in data:
        # Apply current motor powers
        new_data = create_drone_data(drone["id"])
        result.append(new_data)

    connection.send_data(concat_engines(result, T))
    time.sleep(T)

def run_mouse_keyboard_control():
    """Main function to run mouse and keyboard control"""
    global keep_running

    # Start control listener in a separate thread
    control_thread = threading.Thread(target=control_listener)
    control_thread.daemon = True
    control_thread.start()

    # Main control loop
    try:
        while keep_running:
            next_step()
    except KeyboardInterrupt:
        keep_running = False

    print("Control terminated.")

if __name__ == "__main__":
    run_mouse_keyboard_control()