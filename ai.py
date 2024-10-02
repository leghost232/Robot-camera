import cv2
import numpy as np

# Define HSV color ranges for yellow, green, blue, and red
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([30, 255, 255])

# Add sensitivity for green close to yellow
green_lower = np.array([30, 100, 100])  # Light green range
green_upper = np.array([70, 255, 255])

# Blue range, including darker shades for shadows
blue_lower_dark = np.array([100, 50, 50])   # Include darker blue (shadows)
blue_upper_dark = np.array([140, 255, 150])

blue_lower_bright = np.array([100, 150, 150])  # Bright blue
blue_upper_bright = np.array([140, 255, 255])

# Adjusted red ranges to avoid skin tone confusion
red_lower1 = np.array([0, 150, 150])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 150, 150])
red_upper2 = np.array([180, 255, 255])

# Kernel for noise removal (morphological operations)
kernel = np.ones((5, 5), np.uint8)

# Minimum area for detected objects
MIN_AREA = 500

def detect_colors(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Detect yellow
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    
    # Detect green (close to yellow)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    
    # Detect blue (darker and brighter)
    blue_mask_dark = cv2.inRange(hsv_frame, blue_lower_dark, blue_upper_dark)
    blue_mask_bright = cv2.inRange(hsv_frame, blue_lower_bright, blue_upper_bright)
    blue_mask = cv2.bitwise_or(blue_mask_dark, blue_mask_bright)
    
    # Detect red (two ranges in HSV)
    red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Clean up noise using morphological operations
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Combine the masks
    combined_mask = cv2.bitwise_or(yellow_mask, cv2.bitwise_or(green_mask, blue_mask))
    combined_mask = cv2.bitwise_or(combined_mask, red_mask)
    
    # Highlight the detected colors on the original frame
    result_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)
    
    return result_frame, yellow_mask, green_mask, blue_mask, red_mask

def draw_bounding_boxes(frame, mask, color_name):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the red rectangle
            
            # Put the color name inside the bounding box
            cv2.putText(frame, color_name, (x + 5, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    # Access the camera feed
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect colors in the frame
        result_frame, yellow_mask, green_mask, blue_mask, red_mask = detect_colors(frame)
        
        # Draw bounding boxes for each detected color with the color name inside
        draw_bounding_boxes(frame, yellow_mask, "Yellow")
        draw_bounding_boxes(frame, green_mask, "Green")
        draw_bounding_boxes(frame, blue_mask, "Blue")
        draw_bounding_boxes(frame, red_mask, "Red")
        
        # Show the original frame with bounding boxes and detected colors
        cv2.imshow("Detected Colors", result_frame)
        cv2.imshow("Original with Bounding Boxes", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
