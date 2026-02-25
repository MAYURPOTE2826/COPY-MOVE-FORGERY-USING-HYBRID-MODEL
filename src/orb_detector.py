# src/orb_detector.py

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def orb_copy_move_detection(gray):

    orb = cv2.ORB_create(nfeatures=6000)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(descriptors) < 2:
        return 0, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors, descriptors, k=2)

    good_matches = []

    for m, n in matches:
        if m.queryIdx == m.trainIdx:
            continue
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 30:
        return 0, []

    src_pts = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

    # Remove near matches (small displacement)
    displacement = dst_pts - src_pts
    mask_large_shift = np.linalg.norm(displacement, axis=1) > 15

    src_pts = src_pts[mask_large_shift]
    dst_pts = dst_pts[mask_large_shift]

    if len(src_pts) < 25:
        return 0, []

    src_pts = src_pts.reshape(-1,1,2)
    dst_pts = dst_pts.reshape(-1,1,2)

    # RANSAC geometric filtering
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

    if mask is None:
        return 0, []

    inliers = src_pts[mask.ravel() == 1]

    if len(inliers) < 25:
        return 0, []

    points = inliers.reshape(-1,2)

    # DBSCAN spatial clustering
    clustering = DBSCAN(eps=30, min_samples=10).fit(points)
    labels = clustering.labels_

    unique_labels = set(labels)
    best_cluster = None
    max_count = 0

    for label in unique_labels:
        if label == -1:
            continue
        count = np.sum(labels == label)
        if count > max_count:
            max_count = count
            best_cluster = label

    if best_cluster is None:
        return 0, []

    suspicious_points = points[labels == best_cluster]

    if len(suspicious_points) < 40:
        return 0, []

    return len(suspicious_points), suspicious_points.tolist()