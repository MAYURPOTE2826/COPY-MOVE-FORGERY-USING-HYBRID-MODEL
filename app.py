# app.py

from flask import Flask, render_template, request
import os
import cv2
import numpy as np

from src.preprocessing import load_image
# from src.sift_detector import sift_copy_move_detection
from src.hybrid_detector import hybrid_copy_move_detection

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def generate_mask(image_shape, suspicious_positions):

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x, y) in suspicious_positions:
        x = int(y)
        y = int(x)
        cv2.circle(mask, (y, x), 20, 255, -1)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def draw_dotted_boundary(image, mask):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        for i in range(0, len(cnt), 6):
            cv2.circle(image, tuple(cnt[i][0]), 3, (0, 0, 255), -1)

    return image


def generate_mask(image_shape, suspicious_positions):

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x, y) in suspicious_positions:
        cv2.circle(mask, (int(x), int(y)), 25, 255, -1)

    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def draw_highlight(image, mask):

    overlay = image.copy()
    overlay[mask == 255] = [0, 0, 255]

    alpha = 0.4
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            cv2.drawContours(image, [cnt], -1, (0, 0, 255), 3)

    return image
@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        image, gray = load_image(filepath)

        # Run SIFT detection
        cluster_count, suspicious_positions = hybrid_copy_move_detection(gray)
        print("Cluster Count:", cluster_count)
        print("Suspicious Points:", len(suspicious_positions))

        result = image.copy()

        if cluster_count > 0 and len(suspicious_positions) > 0:

            mask = generate_mask(image.shape, suspicious_positions)
            result = draw_dotted_boundary(result, mask)

            status = f"Forgery Detected (Cluster Size: {cluster_count})"
        else:
            status = "No Forgery Detected"

        output_filename = "result.png"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        cv2.imwrite(output_path, result)

        return render_template(
            "index.html",
            input_image=file.filename,
            result_image=output_filename,
            status=status
        )

    return render_template("index.html")



    

if __name__ == "__main__":
    app.run(debug=True)