# src/postprocess.py

import cv2
import numpy as np


def generate_mask(image_shape, suspicious_positions, scale_x, scale_y):

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x, y) in suspicious_positions:
        x = int(x * scale_y)
        y = int(y * scale_x)
        cv2.rectangle(mask, (y, x), (y+20, x+20), 255, -1)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def draw_dotted_boundary(image, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        for i in range(0, len(cnt), 8):
            cv2.circle(image, tuple(cnt[i][0]), 3, (0, 0, 255), -1)

    return image