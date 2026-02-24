# src/sift_detector.py

import cv2
import numpy as np


def sift_copy_move_detection(gray):

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 2:
        return 0, []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors, descriptors, k=2)

    good_matches = []

    for m, n in matches:
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 20:
        return 0, []

    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if mask is None:
        return 0, []

    inlier_pts = src_pts[mask.ravel() == 1]

    if len(inlier_pts) < 20:
        return 0, []

    suspicious_points = inlier_pts.reshape(-1,2)

    return len(suspicious_points), suspicious_points.tolist()