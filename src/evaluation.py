# src/evaluation.py

import cv2
import numpy as np


def calculate_metrics(pred_mask, gt_mask):

    pred_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)[1]
    gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)[1]

    TP = np.sum((pred_mask == 255) & (gt_mask == 255))
    TN = np.sum((pred_mask == 0) & (gt_mask == 0))
    FP = np.sum((pred_mask == 255) & (gt_mask == 0))
    FN = np.sum((pred_mask == 0) & (gt_mask == 255))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)

    return accuracy, precision, recall, f1, iou