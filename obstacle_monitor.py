
import tkinter as tk
from tkinter import messagebox
import serial
import threading
import queue
import re
import time

try:
    import winsound
except ImportError:
    winsound = None

# ---------------- SETTINGS ----------------
PORT = "COM3"
BAUD_RATE = 9600
THRESHOLD = 30.0

# Dark Gold theme
BG = "#171717"
PANEL = "#242424"
GOLD = "#FFD54A"
LIGHT_GOLD = "#FFF0B3"
WHITE = "#FFFFFF"
RED = "#FF2020"
DARK_RED = "#790000"
GREEN = "#176B3A"
LIGHT_GREEN = "#B9FFD0"
AMBER = "#A16207"

# Shared data
data_queue = queue.Queue()
stop_event = threading.Event()

arduino = None
serial_thread = None
sound_thread = None

closing = False
blink_active = False
blink_state = False
blink_job = None

current_status = "WAITING FOR SENSOR"
last_beep_status = False
last_valid_time = 0
connection_lost = False


# ---------------- MAIN WINDOW ----------------
root = tk.Tk()
root.title("Drone Obstacle Monitoring System")
root.geometry("760x540")
root.minsize(650, 480)
root.configure(bg=BG)

tk.Label(
    root,
    text="DRONE OBSTACLE MONITOR",
    font=("Arial", 23, "bold"),
    bg=BG,
    fg=GOLD
).pack(pady=(24, 5))

tk.Label(
    root,
    text="AI-BASED AUTONOMOUS NAVIGATION",
    font=("Arial", 10, "bold"),
    bg=BG,
    fg="#B8B8B8"
).pack(pady=(0, 20))

distance_panel = tk.Frame(
    root,
    bg=PANEL,
    highlightbackground="#8A6D1D",
    highlightthickness=2
)
distance_panel.pack(fill="x", padx=35)

tk.Label(
    distance_panel,
    text="LIVE SENSOR DISTANCE",
    font=("Arial", 12, "bold"),
    bg=PANEL,
    fg=LIGHT_GOLD
).pack(pady=(15, 0))

distance_label = tk.Label(
    distance_panel,
    text="-- cm",
    font=("Arial", 43, "bold"),
    bg=PANEL,
    fg=GOLD
)
distance_label.pack(pady=(5, 15))

status_label = tk.Label(
    root,
    text="CONNECTING TO SENSOR...",
    font=("Arial", 19, "bold"),
    bg="#453713",
    fg=GOLD,
    pady=17
)
status_label.pack(fill="x", padx=35, pady=18)

tk.Label(
    root,
    text="OBSTACLE THRESHOLD: 30 cm",
    font=("Arial", 12, "bold"),
    bg=BG,
    fg=LIGHT_GOLD
).pack(pady=2)

sound_label = tk.Label(
    root,
    text="WARNING SOUND: READY",
    font=("Arial", 10, "bold"),
    bg=BG,
    fg="#C9B879"
)
sound_label.pack(pady=8)

connection_label = tk.Label(
    root,
    text=f"Connecting to Arduino on {PORT}...",
    font=("Arial", 11, "bold"),
    bg=BG,
    fg=GOLD
)
connection_label.pack(pady=4)


# ---------------- SOUND ----------------
def play_siren_once():
    """Play one short two-tone warning outside the GUI thread."""
    global sound_thread

    if closing:
        return

    def sound_worker():
        try:
            if winsound is not None:
                winsound.Beep(850, 130)
                if not stop_event.is_set():
                    winsound.Beep(1250, 200)
            # On non-Windows systems, no siren is played.
        except Exception:
            pass

    sound_thread = threading.Thread(
        target=sound_worker,
        daemon=True
    )
    sound_thread.start()


# ---------------- FLASHING ALERT ----------------
def blink_alert():
    global blink_state, blink_job

    if closing or not blink_active:
        return

    blink_state = not blink_state

    if blink_state:
        status_label.config(bg=RED, fg=WHITE)
    else:
        status_label.config(bg=DARK_RED, fg="#FFE4E4")

    blink_job = root.after(350, blink_alert)


def start_blink():
    global blink_active, blink_state

    if not blink_active:
        blink_active = True
        blink_state = False
        blink_alert()


def stop_blink():
    global blink_active, blink_job

    blink_active = False

    if blink_job is not None:
        try:
            root.after_cancel(blink_job)
        except tk.TclError:
            pass

        blink_job = None


# ---------------- STATUS HANDLING ----------------
def update_status(new_status):
    global current_status, last_beep_status

    was_obstacle = current_status == "OBSTACLE DETECTED"
    is_obstacle = new_status == "OBSTACLE DETECTED"

    current_status = new_status

    if is_obstacle:
        status_label.config(text="!! OBSTACLE DETECTED !!")

        if not was_obstacle:
            start_blink()

        # One siren per detection event
        if not last_beep_status:
            sound_label.config(
                text="WARNING SOUND: PLAYED",
                fg=RED
            )
            play_siren_once()

        last_beep_status = True

    elif new_status == "PATH CLEAR":
        stop_blink()
        last_beep_status = False

        status_label.config(
            text="PATH CLEAR",
            bg=GREEN,
            fg=LIGHT_GREEN
        )

        sound_label.config(
            text="WARNING SOUND: READY",
            fg="#C9B879"
        )

    elif new_status == "NO VALID READING":
        stop_blink()
        last_beep_status = False

        status_label.config(
            text="NO VALID READING",
            bg=AMBER,
            fg=WHITE
        )

        sound_label.config(
            text="WARNING SOUND: READY",
            fg="#C9B879"
        )


# ---------------- SERIAL WORKER ----------------
def serial_reader():
    """Read Arduino continuously without blocking Tkinter."""
    global arduino

    try:
        arduino = serial.Serial(
            PORT,
            BAUD_RATE,
            timeout=0.2
        )

        # Arduino resets when serial connection opens
        time.sleep(2)
        arduino.reset_input_buffer()

        data_queue.put(("CONNECTED", PORT))

        while not stop_event.is_set():
            try:
                raw = arduino.readline()

                if not raw:
                    continue

                line = raw.decode(
                    "utf-8",
                    errors="ignore"
                ).strip()

                match = re.search(
                    r"Average Distance:\s*([0-9]+(?:\.[0-9]+)?)",
                    line
                )

                if match:
                    distance = float(match.group(1))

                    if 0 < distance <= 400:
                        data_queue.put(("DISTANCE", distance))

                elif line == "No valid reading":
                    data_queue.put(("INVALID", None))

            except (serial.SerialException, OSError) as error:
                data_queue.put(("ERROR", str(error)))
                break

    except (serial.SerialException, OSError) as error:
        data_queue.put(("ERROR", str(error)))

    finally:
        if arduino is not None:
            try:
                if arduino.is_open:
                    arduino.close()
            except Exception:
                pass


# ---------------- GUI DATA PROCESSING ----------------
def process_sensor_data():
    global last_valid_time, connection_lost

    if closing:
        return

    try:
        while True:
            event, value = data_queue.get_nowait()

            if event == "CONNECTED":
                connection_label.config(
                    text=f"ARDUINO CONNECTED: {value}",
                    fg=GOLD
                )
                connection_lost = False

            elif event == "DISTANCE":
                last_valid_time = time.monotonic()
                connection_lost = False

                distance_label.config(
                    text=f"{value:.1f} cm"
                )

                if value <= THRESHOLD:
                    update_status("OBSTACLE DETECTED")
                else:
                    update_status("PATH CLEAR")

            elif event == "INVALID":
                # Do not interrupt display for a single bad sample.
                pass

            elif event == "ERROR":
                connection_label.config(
                    text="ARDUINO CONNECTION LOST",
                    fg=RED
                )
                connection_lost = True
                stop_blink()
                status_label.config(
                    text="CONNECTION LOST",
                    bg=DARK_RED,
                    fg=WHITE
                )

    except queue.Empty:
        pass

    # If readings stop, clearly show a sensor timeout
    if (
        last_valid_time > 0
        and time.monotonic() - last_valid_time > 3
        and not connection_lost
    ):
        connection_lost = True
        stop_blink()
        status_label.config(
            text="SENSOR READING TIMEOUT",
            bg=AMBER,
            fg=WHITE
        )
        connection_label.config(
            text="CHECK SENSOR / ARDUINO",
            fg=RED
        )

    root.after(50, process_sensor_data)


# ---------------- START / CLOSE ----------------
def close_app():
    global closing

    closing = True
    stop_event.set()
    stop_blink()

    if arduino is not None:
        try:
            if arduino.is_open:
                arduino.close()
        except Exception:
            pass

    root.destroy()


root.protocol("WM_DELETE_WINDOW", close_app)

def start_app():
    global serial_thread

    serial_thread = threading.Thread(
        target=serial_reader,
        daemon=True
    )
    serial_thread.start()

    root.after(50, process_sensor_data)


root.after(200, start_app)
root.mainloop()
