import cv2
import numpy as np

# Define the color ranges in HSV
color_ranges = {
    'Red': ((0, 100, 100), (10, 255, 255)),
    'Red': ((170, 100, 100), (180, 255, 255)),  # Second range for red
    'Blue': ((100, 100, 100), (140, 255, 255)),
    'Yellow': ((20, 100, 100), (30, 255, 255)),
}

def detect_colors(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_colors = {}

    for color, (lower, upper) in color_ranges.items():
        # Create a mask for the color
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        detected_colors[color] = mask
        
        # Find contours for the detected color
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area for detection
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect colors in the frame
    frame_with_detections = detect_colors(frame)

    # Show the frame with detected colors
    cv2.imshow('Color Detection', frame_with_detections)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()