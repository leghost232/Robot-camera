import cv2
import numpy as np

# Global variable to toggle red box display
show_red_boxes = True
MIN_AREA = 500

# HSV color range definitions with fuzziness
def adjust_color_range(lower, upper, fuzziness=10):
    return (
        np.array([max(0, lower[0] - fuzziness), lower[1], lower[2]]),
        np.array([min(180, upper[0] + fuzziness), upper[1], upper[2]])
    )

# Define HSV color ranges
yellow_lower, yellow_upper = adjust_color_range(np.array([20, 100, 100]), np.array([30, 255, 255]))
green_lower, green_upper = adjust_color_range(np.array([30, 100, 100]), np.array([70, 255, 255]))
blue_lower = np.array([100, 150, 50])
blue_upper = np.array([140, 255, 255])
red_lower1, red_upper1 = adjust_color_range(np.array([0, 150, 150]), np.array([10, 255, 255]))
red_lower2, red_upper2 = adjust_color_range(np.array([160, 150, 150]), np.array([180, 255, 255]))

# Morphological kernels
kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)

# Adjust brightness and gamma
def adjust_brightness_and_gamma(frame, gamma=1.0):
    img_float = frame.astype(np.float32) / 255.0
    img_gamma = np.clip(np.power(img_float, gamma), 0, 1)
    img_corrected = (img_gamma * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)

    if average_brightness < 100:
        img_corrected = cv2.convertScaleAbs(img_corrected, alpha=1.2, beta=0)
    elif average_brightness > 150:
        img_corrected = cv2.convertScaleAbs(img_corrected, alpha=0.8, beta=0)

    return img_corrected

# Detect colors based on HSV range
def detect_colors(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for each color
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Apply morphological operations to clean the masks
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_open)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)

    # Remove dark areas (black) from blue mask
    dark_mask = cv2.inRange(hsv_frame, np.array([0, 0, 0]), np.array([180, 255, 50]))
    blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(dark_mask))

    return yellow_mask, green_mask, blue_mask, red_mask

# Draw bounding boxes around detected colors
def draw_bounding_boxes(frame, masks):
    colors = ['Yellow', 'Green', 'Blue', 'Red']
    color_masks = masks

    for i, mask in enumerate(color_masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        largest_area = 0
        largest_contour = None

        # Only look for the largest contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_AREA:
                if area > largest_area:
                    largest_area = area
                    largest_contour = contour

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, colors[i], (x + 5, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw red boxes for all contours if toggled on
        if show_red_boxes:
            for contour in contours:
                if contour is not largest_contour and cv2.contourArea(contour) > MIN_AREA:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, colors[i], (x + 5, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Draw toggle button on the frame
def draw_toggle_button(frame):
    button_x, button_y = 10, 10
    button_width, button_height = 200, 50

    cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), (200, 200, 200), -1)
    text = "Toggle Red Boxes" if show_red_boxes else "Red Boxes Off"
    cv2.putText(frame, text, (button_x + 5, button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# Mouse callback to toggle red boxes
def mouse_callback(event, x, y, flags, param):
    global show_red_boxes
    button_x, button_y = 10, 10
    button_width, button_height = 200, 50

    if event == cv2.EVENT_LBUTTONDOWN:
        if button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height:
            show_red_boxes = not show_red_boxes

# Main function to run the video stream
def main():
    global show_red_boxes
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cv2.namedWindow("Original with Bounding Boxes")
    cv2.setMouseCallback("Original with Bounding Boxes", mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = adjust_brightness_and_gamma(frame, gamma=1.2)
        
        masks = detect_colors(frame)
        draw_bounding_boxes(frame, masks)
        draw_toggle_button(frame)
        
        # Display the frame
        cv2.imshow("Original with Bounding Boxes", frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
