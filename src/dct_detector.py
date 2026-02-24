# src/dct_detector.py

import cv2
import numpy as np
from scipy.fftpack import dct
from sklearn.decomposition import PCA


def compute_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def dct_copy_move_detection(gray,
                            block_size=8,
                            step=8,
                            similarity_threshold=0.98):

    gray = cv2.resize(gray, (256, 256))
    h, w = gray.shape

    features = []
    positions = []

    for i in range(0, h - block_size, step):
        for j in range(0, w - block_size, step):

            block = gray[i:i+block_size, j:j+block_size]

            if np.var(block) < 25:
                continue

            dct_block = compute_dct(block)
            feature = dct_block[:4, :4].flatten()

            features.append(feature)
            positions.append((i, j))

    if len(features) < 2:
        return 0, []

    features = np.array(features)

    pca = PCA(n_components=6)
    features = pca.fit_transform(features)

    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    sorted_idx = np.lexsort(features.T)
    features = features[sorted_idx]
    positions = [positions[i] for i in sorted_idx]

    suspicious_positions = []
    window = 10

    for i in range(len(features)):
        for j in range(i+1, min(i+window, len(features))):

            similarity = np.dot(features[i], features[j])

            if similarity > similarity_threshold:
                suspicious_positions.append(positions[i])
                suspicious_positions.append(positions[j])

    if len(suspicious_positions) < 20:
        return 0, []

    return len(suspicious_positions), suspicious_positions