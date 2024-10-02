import cv2
import numpy as np

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

def adjust_color_range(lower, upper, fuzziness=10):
    return (
        np.array([max(0, lower[0] - fuzziness), lower[1], lower[2]]),
        np.array([min(180, upper[0] + fuzziness), upper[1], upper[2]])
    )

# Define HSV color ranges with adjusted fuzziness
yellow_lower, yellow_upper = adjust_color_range(np.array([20, 100, 100]), np.array([30, 255, 255]))
green_lower, green_upper = adjust_color_range(np.array([30, 100, 100]), np.array([70, 255, 255]))
blue_lower = np.array([100, 150, 50])
blue_upper = np.array([140, 255, 255])
red_lower1, red_upper1 = adjust_color_range(np.array([0, 100, 100]), np.array([10, 255, 255]), fuzziness=10)  # Narrowed
red_lower2, red_upper2 = adjust_color_range(np.array([160, 100, 100]), np.array([180, 255, 255]), fuzziness=10)  # Narrowed

kernel_open = np.ones((5, 5), np.uint8)
kernel_close = np.ones((7, 7), np.uint8)
MIN_AREA = 800  # Increased minimum area for red boxes

def detect_colors(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect colors
    yellow_mask = cv2.inRange(hsv_frame, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
    red_mask1 = cv2.inRange(hsv_frame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_frame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Apply morphological operations
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel_open)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel_close)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)  # Changed to close

    # Remove dark areas (black) from the blue mask
    dark_mask = cv2.inRange(hsv_frame, np.array([0, 0, 0]), np.array([180, 255, 50]))  # Low brightness
    blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(dark_mask))

    return yellow_mask, green_mask, blue_mask, red_mask

def draw_bounding_boxes(frame, masks):
    colors = ['Yellow', 'Green', 'Blue', 'Red']
    color_masks = masks

    largest_area = 0
    largest_color_index = -1

    for i, mask in enumerate(color_masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_AREA:
                if area > largest_area:
                    largest_area = area
                    largest_color_index = i
                if colors[i] != 'Green':
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box for other colors
                    cv2.putText(frame, colors[i], (x + 5, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Draw a green box for the most prominent color
    if largest_color_index != -1:
        largest_mask = color_masks[largest_color_index]
        contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area == largest_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for largest color
                cv2.putText(frame, colors[largest_color_index], (x + 5, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = adjust_brightness_and_gamma(frame, gamma=1.2)
        
        masks = detect_colors(frame)
        
        draw_bounding_boxes(frame, masks)
        
        cv2.imshow("Original with Bounding Boxes", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
