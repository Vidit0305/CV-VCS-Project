import os
os.environ["QT_LOGGING_RULES"] = "*=false"

import cv2
import numpy as np

# 1. Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open webcam.")
    exit()

# 2. Window setup: resizable (half-screen, full-screen) with clean UI (no Qt toolbars)
window_name = "Color Detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
is_fullscreen = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Mirror camera so movement feels natural
    frame = cv2.flip(frame, 1)

    # 4. Auto-fit camera feed to window dimensions (eliminates white sidebars)
    rect = cv2.getWindowImageRect(window_name)
    win_w, win_h = rect[2], rect[3]
    h, w, _ = frame.shape

    if win_w > 100 and win_h > 100:
        win_ar = win_w / win_h
        cam_ar = w / h
        if win_ar > cam_ar:
            crop_h = int(w / win_ar)
            y1 = (h - crop_h) // 2
            display = cv2.resize(frame[y1:y1 + crop_h, :], (win_w, win_h))
        else:
            crop_w = int(h * win_ar)
            x1 = (w - crop_w) // 2
            display = cv2.resize(frame[:, x1:x1 + crop_w], (win_w, win_h))
    else:
        display = frame

    # 5. Extract center Region of Interest (ROI)
    dh, dw, _ = display.shape
    center_x, center_y = dw // 2, dh // 2
    box_size = min(dw, dh) // 8
    roi = display[center_y - box_size:center_y + box_size, center_x - box_size:center_x + box_size]

    # 6. Convert to HSV and determine dominant color
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = np.mean(hsv, axis=(0, 1))

    if val < 50:
        color = "BLACK"
    elif sat < 50 and val > 180:
        color = "WHITE"
    elif sat < 50:
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

    # 7. Draw clean center target box
    cv2.rectangle(display, (center_x - box_size, center_y - box_size),
                  (center_x + box_size, center_y + box_size), (255, 255, 255), 2)

    # 8. Clean UI: Display ONLY the color on a sleek high-contrast badge
    text_size, _ = cv2.getTextSize(color, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    text_w, text_h = text_size
    text_x = center_x - text_w // 2
    text_y = center_y + box_size + 45

    pad_x, pad_y = 16, 8
    cv2.rectangle(display, (text_x - pad_x, text_y - text_h - pad_y),
                  (text_x + text_w + pad_x, text_y + pad_y), (25, 25, 25), -1)
    cv2.putText(display, color, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('f'), ord('F')):
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
