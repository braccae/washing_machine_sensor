#!/usr/bin/env python3
"""
Calibration tool for Washing Machine Status Monitor
This script helps to identify the exact positions of indicator lights
"""

import os
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# RTSP Stream URL
RTSP_URL = os.getenv("RTSP_URL", "rtsp://192.168.2.96:8554/video3_unicast")

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates when left mouse button is clicked
        print(f"Selected point: x={x}, y={y}")
        # Add a marker to the image at the selected point
        cv2.circle(param['frame'], (x, y), 5, (0, 255, 0), -1)
        # Add the selected ROI to the list
        roi = [x-10, y-5, 20, 10]  # Default ROI size
        param['rois'].append(roi)
        param['roi_names'].append(f"roi_{len(param['rois'])}")
        print(f"Added ROI: {roi}")

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print(f"Error: Could not open video stream {RTSP_URL}")
        return
    
    print(f"Connected to video stream: {RTSP_URL}")
    print("Calibration tool instructions:")
    print("1. Click on each indicator light in the order: sensing, wash, rinse, spin, done, lid_locked")
    print("2. Press 'q' to quit and save the calibration")
    print("3. Press 's' to save a screenshot for reference")
    
    # Create a window and set mouse callback
    window_name = "Calibration Tool"
    cv2.namedWindow(window_name)
    
    # Parameters to pass to callback
    params = {'frame': None, 'rois': [], 'roi_names': []}
    
    cv2.setMouseCallback(window_name, mouse_callback, params)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Store the current frame in params
        params['frame'] = frame.copy()
        
        # Display ROIs on the frame
        for i, roi in enumerate(params['rois']):
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, params['roi_names'][i], (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display frame
        cv2.imshow(window_name, frame)
        
        # Handle keypress
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current frame as a reference
            cv2.imwrite("calibration_reference.jpg", frame)
            print("Saved calibration reference image")
        
    # Generate code snippet for the ROIs
    print("\n--- Copy this to your washer_monitor.py file ---")
    
    # Predefined indicator names
    indicator_names = ["sensing", "wash", "rinse", "spin", "done", "lid_locked"]
    
    code_snippet = "INDICATOR_REGIONS = {\n"
    
    for i, roi in enumerate(params['rois']):
        if i < len(indicator_names):
            name = indicator_names[i]
        else:
            name = f"unknown_{i}"
            
        code_snippet += f"    \"{name}\": {roi},\n"
    
    code_snippet += "}"
    
    print(code_snippet)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
