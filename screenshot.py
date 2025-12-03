import cv2
import numpy as np
import sys
import os


import numpy as np

def detect_board(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found:", img_path)
        return

    h, w = img.shape[:2]

    # Convert to HSV → wood colors have distinct hue
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # MASK FOR WOOD (light + dark squares)
    # These ranges catch both light and dark wooden tiles
    lower = np.array([5, 10, 50])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Morph clean
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)

    # Find contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        print("Board not detected.")
        return

    # Largest wooden area = chessboard
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w_b, h_b = cv2.boundingRect(cnt)

    board = img[y:y+h_b, x:x+w_b]

    # Show final cropped board
    cv2.imshow("Cropped Board", board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return board



# ...existing code...
detect_board("C:\\Users\\ayush\\OneDrive\\Pictures\\Screenshots\\Screenshot (240).png")