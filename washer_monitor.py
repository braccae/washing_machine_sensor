#!/usr/bin/env python3
"""
Washing Machine Status Monitor
Detects the status of a washing machine by analyzing indicator lights in a video feed
and publishes the status to an MQTT topic.
"""

import os
import time
import json
import sys
import argparse
from json.encoder import JSONEncoder
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

# Redirect stderr to suppress OpenCV H264 decoding errors
class NullWriter:
    def write(self, text):
        pass
    def flush(self):
        pass

# Load environment variables
load_dotenv()

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "home/washingmachine/status")
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# Home Assistant MQTT Discovery Configuration
DEVICE_ID = "washing_machine"
DEVICE_NAME = "Washing Machine"
HA_DISCOVERY_PREFIX = "homeassistant"

# Custom JSON encoder to handle boolean values
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bool):
            return str(obj).lower()  # Convert True/False to "true"/"false"
        return JSONEncoder.default(self, obj)

# RTSP Stream URL
RTSP_URL = os.getenv("RTSP_URL", "rtsp://192.168.2.96:8554/video3_unicast")

# Thresholds for light detection
FLASH_THRESHOLD = 10.0   # Threshold for detecting flashing (standard deviation)

# Define regions of interest for each indicator light (x, y, width, height)
# These will need to be adjusted based on the actual video feed and calibration
INDICATOR_REGIONS = {
    "sensing": [501, 290, 20, 10],
    "wash": [547, 289, 20, 10],
    "rinse": [594, 290, 20, 10],
    "spin": [644, 289, 20, 10],
    "done": [690, 291, 20, 10],
    "lid_locked": [738, 294, 20, 10],
}

# Initialize MQTT client
def setup_mqtt_client():
    # MQTT Callbacks
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"Successfully connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
            # Subscribe to command topic if needed
            # client.subscribe(f"{MQTT_TOPIC}/command")
        else:
            print(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def on_disconnect(client, userdata, rc):
        if rc != 0:
            print(f"Unexpected disconnection from MQTT broker, return code: {rc}")
    
    def on_publish(client, userdata, mid):
        print(f"Message published: mid={mid}")
    
    # Create client and set callbacks
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_publish = on_publish
    
    # Set username and password if provided
    if MQTT_USERNAME and MQTT_PASSWORD:
        print(f"Using MQTT authentication with username: {MQTT_USERNAME}")
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    # Connect to the broker
    print(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    return client

class ROITracker:
    """Tracks a region of interest over time to detect flashing"""
    def __init__(self, name, buffer_size=15):
        self.name = name
        self.buffer_size = buffer_size
        self.frame_buffer = []  # Store recent frames for this ROI
        self.brightness_history = []  # Store brightness values
        self.mean_brightness = 0
        self.variance = 0
        self.std_dev = 0
        self.is_flashing = False
        self.flash_count = 0  # Count consecutive detections of flashing
    
    def update(self, roi_frame):
        """Add a new frame to the buffer and update statistics"""
        try:
            # Convert to grayscale
            if len(roi_frame.shape) > 2:
                gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi_frame
                
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Add to history
            self.brightness_history.append(brightness)
            if len(self.brightness_history) > self.buffer_size:
                self.brightness_history.pop(0)
                
            # Add frame to buffer
            self.frame_buffer.append(gray.copy())
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
                
            # Calculate statistics once we have enough frames
            if len(self.brightness_history) >= 3:
                self.mean_brightness = np.mean(self.brightness_history)
                self.std_dev = np.std(self.brightness_history)
                
                # Check if there's significant variation (flashing)
                if self.std_dev > FLASH_THRESHOLD:
                    self.flash_count += 1
                    # Need 2 consecutive detections to confirm flashing
                    self.is_flashing = self.flash_count >= 2
                else:
                    # Reset flash count if under threshold
                    self.flash_count = max(0, self.flash_count - 1)  # Gradually reduce
                    if self.flash_count <= 0:
                        self.is_flashing = False
                
            return self.is_flashing, self.mean_brightness, self.std_dev
            
        except Exception as e:
            print(f"Error updating ROI tracker: {e}")
            return False, 0, 0

# Dictionary to track each ROI
roi_trackers = {}

def is_light_on(frame, roi, name=None):
    """Check if a light is flashing by analyzing temporal changes in the ROI"""
    x, y, w, h = roi
    
    # Make sure we're within frame bounds
    height, width = frame.shape[:2]
    if x >= width or y >= height or x + w <= 0 or y + h <= 0:
        print(f"Warning: ROI {x},{y},{w},{h} is outside frame dimensions {width}x{height}")
        return False, 0, 0
    
    # Ensure we don't go out of bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    
    if w <= 0 or h <= 0:
        return False, 0, 0
    
    try:
        # Extract region
        region = frame[y:y+h, x:x+w]
        
        # Create tracker if it doesn't exist
        if name not in roi_trackers:
            roi_trackers[name] = ROITracker(name)
        
        # Update tracker with new frame
        is_flashing, brightness, std_dev = roi_trackers[name].update(region)
        
        return is_flashing, brightness, std_dev
    except Exception as e:
        print(f"Error in is_light_on: {e}")
        return False, 0, 0

def get_washer_status(frame):
    """Determine the washing machine status based on the indicator lights"""
    status = {}
    brightness_values = {}
    std_dev_values = {}
    
    # Check each indicator light and track motion/flashing
    for name, roi in INDICATOR_REGIONS.items():
        is_flashing, brightness, std_dev = is_light_on(frame, roi, name=name)
        status[name] = is_flashing  # Light is considered ON if it's flashing
        brightness_values[name] = brightness
        std_dev_values[name] = std_dev
    
    # Add timestamp
    status["timestamp"] = time.time()
    
    # Store values for debugging
    status["brightness_values"] = brightness_values
    status["std_dev_values"] = std_dev_values
    
    # Determine overall status
    if status["done"] and not any([status["sensing"], status["wash"], status["rinse"], status["spin"]]):
        status["overall_status"] = "complete"
    elif not any([status["sensing"], status["wash"], status["rinse"], status["spin"], status["done"]]):
        status["overall_status"] = "off"
    else:
        # Find the current active stage
        active_stages = []
        if status["sensing"]:
            active_stages.append("sensing")
        if status["wash"]:
            active_stages.append("wash")
        if status["rinse"]:
            active_stages.append("rinse")
        if status["spin"]:
            active_stages.append("spin")
            
        if active_stages:
            status["overall_status"] = active_stages[-1]  # Get the last active stage
        else:
            status["overall_status"] = "unknown"
    
    # Add lid lock status separately as it can be combined with any stage
    status["lid_status"] = "locked" if status["lid_locked"] else "unlocked"
    
    return status

def create_debug_frame(frame, status):
    """Create a debug frame with ROIs and motion/flashing detection displayed"""
    # Draw ROIs on the frame
    debug_frame = frame.copy()
    
    # Get values if available
    brightness_values = status.get("brightness_values", {})
    std_dev_values = status.get("std_dev_values", {})
    
    # Draw a legend for the flash threshold
    cv2.putText(debug_frame, f"Flash Threshold: {FLASH_THRESHOLD}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    
    for name, roi in INDICATOR_REGIONS.items():
        x, y, w, h = roi
        brightness = brightness_values.get(name, 0)
        std_dev = std_dev_values.get(name, 0)
        
        # Color based on whether light is detected as flashing
        if status.get(name, False):
            # Green if flashing detected
            color = (0, 255, 0)
        else:
            # Color gradient from red to yellow based on how close to threshold
            ratio = min(1.0, std_dev / FLASH_THRESHOLD)
            blue = 0
            green = int(255 * ratio)
            red = 255
            color = (blue, green, red)
        
        # Draw rectangle around ROI
        cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
        
        # Display name, brightness and std dev
        cv2.putText(debug_frame, f"{name}: {brightness:.1f} (σ={std_dev:.2f})", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Add overall status to the frame
    cv2.putText(debug_frame, f"Status: {status.get('overall_status', 'unknown')}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Add lid status
    cv2.putText(debug_frame, f"Lid: {status.get('lid_status', 'unknown')}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    return debug_frame

def save_debug_frame(frame, status, frame_count):
    """Save frames periodically for debugging purposes"""
    if frame_count % 100 == 0:  # Save every 100th frame
        timestamp = int(time.time())
        debug_dir = "debug_frames"
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_frame = create_debug_frame(frame, status)
        cv2.imwrite(f"{debug_dir}/frame_{timestamp}.jpg", debug_frame)

def get_stable_status(cap, num_samples=5, sample_interval=0.1, detection_threshold=0.3):
    """
    Capture multiple frames over a period and determine a stable status
    by temporal averaging to handle flashing lights.
    
    Args:
        cap: OpenCV video capture object
        num_samples: Number of frames to sample
        sample_interval: Time between samples in seconds
        detection_threshold: Percentage threshold to consider a light as ON
        
    Returns:
        Stable status dictionary
    """
    # Initialize counters for each indicator
    light_counters = {
        "sensing": 0,
        "wash": 0,
        "rinse": 0,
        "spin": 0,
        "done": 0,
        "lid_locked": 0
    }
    
    # Sample frames
    successful_samples = 0
    for _ in range(num_samples):
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Get status for this frame
        frame_status = get_washer_status(frame)
        successful_samples += 1
        
        # Increment counters for each ON light
        for key in light_counters.keys():
            if frame_status[key]:
                light_counters[key] += 1
                
        # Small delay between samples
        time.sleep(sample_interval)
    
    # If we couldn't get any successful samples, return None
    if successful_samples == 0:
        return None
        
    # Determine stable status based on threshold
    stable_status = {}
    for key in light_counters.keys():
        # Consider a light ON if it was detected in at least threshold% of samples
        stable_status[key] = (light_counters[key] / successful_samples) >= detection_threshold
    
    # Add timestamp
    stable_status["timestamp"] = time.time()
    
    # Determine overall status
    if stable_status["done"] and not any([stable_status["sensing"], stable_status["wash"], 
                                      stable_status["rinse"], stable_status["spin"]]):
        stable_status["overall_status"] = "complete"
    elif not any([stable_status["sensing"], stable_status["wash"], 
                 stable_status["rinse"], stable_status["spin"], stable_status["done"]]):
        stable_status["overall_status"] = "off"
    else:
        # Find the current active stage
        active_stages = []
        if stable_status["sensing"]:
            active_stages.append("sensing")
        if stable_status["wash"]:
            active_stages.append("wash")
        if stable_status["rinse"]:
            active_stages.append("rinse")
        if stable_status["spin"]:
            active_stages.append("spin")
            
        if active_stages:
            stable_status["overall_status"] = active_stages[-1]  # Get the last active stage
        else:
            stable_status["overall_status"] = "unknown"
    
    # Add lid lock status separately as it can be combined with any stage
    stable_status["lid_status"] = "locked" if stable_status["lid_locked"] else "unlocked"
    
    return stable_status

def publish_ha_discovery(client):
    """Publish Home Assistant MQTT discovery configurations"""
    
    # Base information for discovery
    base_config = {
        "name": DEVICE_NAME,
        "unique_id": f"{DEVICE_ID}_status",
        "state_topic": MQTT_TOPIC,
        "json_attributes_topic": MQTT_TOPIC,
        "device": {
            "identifiers": [DEVICE_ID],
            "name": DEVICE_NAME,
            "model": "Washing Machine Monitor",
            "manufacturer": "Custom",
        }
    }
    
    # Configuration for main sensor
    sensor_config = base_config.copy()
    sensor_config.update({
        "icon": "mdi:washing-machine",
        "value_template": "{{ value_json.state }}",
    })
    
    # Discovery topic for main sensor
    discovery_topic = f"{HA_DISCOVERY_PREFIX}/sensor/{DEVICE_ID}/state/config"
    
    # Publish the discovery message for the main sensor
    client.publish(discovery_topic, json.dumps(sensor_config), retain=True)
    print(f"Published HA discovery config to {discovery_topic}")
    
    # Create binary sensors for each indicator light
    for light in ["sensing", "wash", "rinse", "spin", "done", "lid_locked"]:
        binary_config = {
            "name": f"{DEVICE_NAME} {light.replace('_', ' ').title()}",
            "unique_id": f"{DEVICE_ID}_{light}",
            "state_topic": MQTT_TOPIC,
            "value_template": f"{{{{ value_json.{light} }}}}",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device": base_config["device"],
        }
        
        # Customize icon based on light type
        if light == "lid_locked":
            binary_config["icon"] = "mdi:lock"
        elif light == "done":
            binary_config["icon"] = "mdi:check-circle"
        else:
            binary_config["icon"] = "mdi:washing-machine"
            
        # Discovery topic
        discovery_topic = f"{HA_DISCOVERY_PREFIX}/binary_sensor/{DEVICE_ID}/{light}/config"
        
        # Publish configuration
        client.publish(discovery_topic, json.dumps(binary_config), retain=True)
        print(f"Published HA discovery config for {light} to {discovery_topic}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Washing Machine Status Monitor')
    parser.add_argument('--debug-view', action='store_true', help='Enable visual debugging window')
    parser.add_argument('--threshold', type=float, default=10.0, help='Flash detection threshold (standard deviation, default: 10.0)')
    args = parser.parse_args()
    
    # Update flash threshold if specified
    global FLASH_THRESHOLD
    FLASH_THRESHOLD = args.threshold
    
    # Redirect stderr to suppress OpenCV decoding errors
    old_stderr = sys.stderr
    sys.stderr = NullWriter()
    
    # Setup MQTT client
    client = setup_mqtt_client()
    
    # Publish Home Assistant discovery configurations
    publish_ha_discovery(client)
    
    # Initialize video capture
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        sys.stderr = old_stderr  # Restore stderr for error output
        print(f"Error: Could not open video stream {RTSP_URL}")
        return
    
    print(f"Connected to video stream: {RTSP_URL}")
    print(f"Publishing status updates to MQTT topic: {MQTT_TOPIC}")
    print(f"Monitoring ROIs: {INDICATOR_REGIONS}")
    print(f"Flash detection threshold: {FLASH_THRESHOLD}")
    
    # Set up visual debugging if enabled
    if args.debug_view:
        print("Visual debugging enabled. Press 'q' to quit.")
        cv2.namedWindow('Washing Machine Monitor', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Washing Machine Monitor', 1280, 720)
    
    last_status = None
    frame_count = 0
    stable_counter = 0
    required_stable_readings = 2  # Number of consecutive stable readings required
    pending_status = None
    
    try:
        while True:
            # Check if video stream is still open
            if not cap.isOpened():
                print("Error: Video stream closed unexpectedly")
                # Try to reconnect
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(RTSP_URL)
                continue
            
            # Get stable status by sampling multiple frames
            # Using a lower detection threshold (30%) to better catch flashing lights
            status = get_stable_status(cap, num_samples=10, sample_interval=0.1, detection_threshold=0.3)
            
            if status is None:
                print("Error: Failed to get stable status. Reconnecting...")
                cap.release()
                time.sleep(5)
                cap = cv2.VideoCapture(RTSP_URL)
                continue
            
            # Use a single frame for debug visualization
            ret, debug_frame = cap.read()
            if ret:
                # Save debug frame periodically
                save_debug_frame(debug_frame, status, frame_count)
                
                # Show visual debugging if enabled
                if args.debug_view:
                    # Create debug frame with annotations
                    vis_frame = create_debug_frame(debug_frame, status)
                    
                    # Display the frame
                    cv2.imshow('Washing Machine Monitor', vis_frame)
                    
                    # Check for key press to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("User requested exit from debug view")
                        break
            
            # Check if status has changed from last published status
            status_changed = (last_status is None or status["overall_status"] != last_status["overall_status"])
            
            # State debouncing: Only publish status change after it's been stable for multiple readings
            if status_changed:
                if pending_status is None or status["overall_status"] != pending_status["overall_status"]:
                    # Reset counter for new pending status
                    pending_status = status.copy()
                    stable_counter = 1
                    print(f"Potential new status detected: {status['overall_status']}. Confirming...")
                    # Print out which lights are detected as on
                    on_lights = [name for name, is_on in status.items() if is_on and name not in ["timestamp", "overall_status", "lid_status"]]
                    print(f"Detected lights: {on_lights}")
                else:
                    # Increment counter for existing pending status
                    stable_counter += 1
                    print(f"Confirming status: {status['overall_status']} ({stable_counter}/{required_stable_readings})")
                    
                    # Check if we have enough stable readings
                    if stable_counter >= required_stable_readings:
                        print(f"Status change detected and confirmed: {status['overall_status']}")
                        
                        # Create a simplified payload for Home Assistant integration
                        state_payload = {
                            "state": status['overall_status'],
                            "lid": status['lid_status'],
                            "sensing": "ON" if status['sensing'] else "OFF",
                            "wash": "ON" if status['wash'] else "OFF",
                            "rinse": "ON" if status['rinse'] else "OFF",
                            "spin": "ON" if status['spin'] else "OFF",
                            "done": "ON" if status['done'] else "OFF",
                            "lid_locked": "ON" if status['lid_locked'] else "OFF",
                        }
                        
                        # Convert to JSON string
                        state_json = json.dumps(state_payload)
                        
                        # Publish to the main state topic for Home Assistant
                        client.publish(MQTT_TOPIC, state_json)
                        print(f"Published state: {state_payload['state']}")
                        
                        # Create detailed payload with all data for debugging
                        json_parts = []
                        for key, value in status.items():
                            if isinstance(value, bool):
                                # Convert boolean to JSON true/false (no quotes)
                                json_value = "true" if value else "false"
                            elif isinstance(value, (int, float)):
                                # Numbers don't need quotes
                                json_value = str(value)
                            elif isinstance(value, str):
                                # Strings need quotes and escaping
                                json_value = f'"{value}"'
                            else:
                                # Convert other types to string representation
                                json_value = f'"{str(value)}"'
                            
                            json_parts.append(f'"{key}":{json_value}')
                        
                        # Publish detailed data to a separate topic
                        detailed_payload = '{' + ','.join(json_parts) + '}'
                        client.publish(f"{MQTT_TOPIC}/detailed", detailed_payload)
                        last_status = status.copy()
                        pending_status = None
                        stable_counter = 0
            else:
                # Reset if status is stable
                pending_status = None
                stable_counter = 0
            
            frame_count += 1
            time.sleep(1)  # Wait between polling cycles
            
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        sys.stderr = old_stderr  # Restore stderr
        cap.release()
        client.loop_stop()
        client.disconnect()
        
        # Close any open OpenCV windows
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
