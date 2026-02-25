# evaluate_dataset.py

import os
import cv2
import numpy as np
from src.preprocessing import load_image
from src.hybrid_detector import hybrid_copy_move_detection
from src.evaluation import calculate_metrics


DATASET_IMAGE_FOLDER = "dataset/images"
DATASET_MASK_FOLDER = "dataset/masks"


def generate_mask(image_shape, suspicious_positions):

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for (x, y) in suspicious_positions:
        cv2.circle(mask, (int(x), int(y)), 30, 255, -1)

    kernel = np.ones((30, 30), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask


def evaluate_dataset():

    image_files = os.listdir(DATASET_IMAGE_FOLDER)

    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0

    count = 0

    for filename in image_files:

        image_path = os.path.join(DATASET_IMAGE_FOLDER, filename)
        mask_path = os.path.join(DATASET_MASK_FOLDER, filename)

        if not os.path.exists(mask_path):
            continue

        image, gray = load_image(image_path)

        cluster_count, suspicious_positions = hybrid_copy_move_detection(gray)

        pred_mask = generate_mask(image.shape, suspicious_positions)

        gt_mask = cv2.imread(mask_path, 0)

        accuracy, precision, recall, f1, iou = calculate_metrics(pred_mask, gt_mask)

        print(f"Image: {filename}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"IoU: {iou:.4f}")
        print("-" * 30)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_iou += iou

        count += 1

    if count == 0:
        print("No valid images found.")
        return

    print("\n===== FINAL AVERAGE RESULTS =====")
    print(f"Average Accuracy: {total_accuracy / count:.4f}")
    print(f"Average Precision: {total_precision / count:.4f}")
    print(f"Average Recall: {total_recall / count:.4f}")
    print(f"Average F1 Score: {total_f1 / count:.4f}")
    print(f"Average IoU: {total_iou / count:.4f}")


if __name__ == "__main__":
    evaluate_dataset()