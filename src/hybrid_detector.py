# src/hybrid_detector.py

import numpy as np
from .dct_detector import dct_copy_move_detection
from .sift_detector import sift_copy_move_detection
from .orb_detector import orb_copy_move_detection


def compute_texture_score(gray):
    return np.var(gray)


def hybrid_copy_move_detection(gray):

    texture_score = compute_texture_score(gray)
    print("Texture Score:", texture_score)

    # Adaptive threshold
    if texture_score > 600:
        print("Using DCT + SIFT")

        cluster_dct, pts_dct = dct_copy_move_detection(gray)
        cluster_sift, pts_sift = sift_copy_move_detection(gray)

        # Select stronger detection
        if cluster_sift > cluster_dct:
            return cluster_sift, pts_sift
        else:
            return cluster_dct, pts_dct

    else:
        print("Using ORB (Low Texture Image)")
        return orb_copy_move_detection(gray)