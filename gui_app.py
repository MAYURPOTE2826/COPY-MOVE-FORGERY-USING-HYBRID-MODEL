# gui_app.py

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
from src.preprocessing import load_image
from src.dct_detector import dct_copy_move_detection
from tkinterdnd2 import TkinterDnD, DND_FILES


def detect_image(path):
    image, gray = load_image(path)
    cluster, suspicious_positions = dct_copy_move_detection(gray)

    result = image.copy()

    scale_x = image.shape[1] / 256
    scale_y = image.shape[0] / 256

    for (x, y) in suspicious_positions:
        x = int(x * scale_y)
        y = int(y * scale_x)

        cv2.rectangle(result, (y, x), (y+20, x+20), (0, 0, 255), 2)

    if cluster > 10:
        label_text.set("Forgery Detected ❌")
    else:
        label_text.set("No Forgery Detected ✅")

    cv2.imwrite("output_gui.png", result)
    show_image("output_gui.png")


def show_image(path):
    img = Image.open(path)
    img = img.resize((400, 300))
    img = ImageTk.PhotoImage(img)
    panel.configure(image=img)
    panel.image = img


def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detect_image(file_path)


root = TkinterDnD.Tk()
root.title("Copy-Move Forgery Detector")
root.geometry("500x500")

btn = tk.Button(root, text="Select Image", command=upload_image)
btn.pack(pady=10)

label_text = tk.StringVar()
label = tk.Label(root, textvariable=label_text, font=("Arial", 14))
label.pack()

panel = tk.Label(root)
panel.pack(pady=10)
panel.drop_target_register(DND_FILES)
panel.dnd_bind('<<Drop>>', lambda e: detect_image(e.data))
root.mainloop()