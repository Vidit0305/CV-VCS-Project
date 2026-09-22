import cv2
import numpy as np

# 1. Open laptop webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

while True:
    # 2. Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Define center Region of Interest (ROI)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    box_size = 50
    roi = frame[center_y - box_size:center_y + box_size, center_x - box_size:center_x + box_size]

    # 4. Convert ROI to HSV and calculate average values
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = np.mean(hsv_roi, axis=(0, 1))

    # 5. Determine dominant color based on HSV ranges
    if value < 50:
        color = "BLACK"
    elif saturation < 50 and value > 180:
        color = "WHITE"
    elif saturation < 50:
        color = "UNKNOWN"
    elif hue < 10 or hue > 170:
        color = "RED"
    elif 15 <= hue < 35:
        color = "YELLOW"
    elif 35 <= hue < 85:
        color = "GREEN"
    elif 85 <= hue < 135:
        color = "BLUE"
    else:
        color = "UNKNOWN"

    # 6. Draw center rectangle and display detected color
    cv2.rectangle(frame, (center_x - box_size, center_y - box_size),
                  (center_x + box_size, center_y + box_size), (0, 255, 0), 2)
    cv2.putText(frame, f"Detected Color: {color}", (center_x - 140, center_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 7. Show live feed; press 'q' or 'Q' to quit
    cv2.imshow("Color Detector", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

# 8. Release webcam and close window
cap.release()
cv2.destroyAllWindows()
