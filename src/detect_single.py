# detect_single.py

import cv2
import os
from src.preprocessing import load_image
from src.dct_detector import dct_copy_move_detection


def detect_image(input_path):

    image, gray = load_image(input_path)

    cluster, suspicious_positions = dct_copy_move_detection(gray)

    result = image.copy()

    scale_x = image.shape[1] / 256
    scale_y = image.shape[0] / 256

    for (x, y) in suspicious_positions:

        # Scale back to original size
        x = int(x * scale_y)
        y = int(y * scale_x)

        cv2.rectangle(result,
                      (y, x),
                      (y+20, x+20),
                      (0, 0, 255),
                      2)

    if cluster > 10:
        label = "Forgery Detected"
        color = (0, 0, 255)
    else:
        label = "No Forgery Detected"
        color = (0, 255, 0)

    cv2.putText(result,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/result.png", result)

    print("Detection Completed.")
    print("Output saved in output/result.png")


if __name__ == "__main__":
    detect_image("test_image.png")  # change to your input image