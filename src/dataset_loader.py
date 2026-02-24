# src/dataset_loader.py

import os

def load_dataset(dataset_path="dataset",
                 only_clean=True,
                 limit=None):
    """
    Load dataset images and labels.

    0 = Original
    1 = Forged
    """

    original_path = os.path.join(dataset_path, "original")
    forged_path = os.path.join(dataset_path, "forged")

    valid_ext = (".png", ".jpg", ".jpeg")

    original_images = []
    forged_images = []

    # -----------------------------
    # Load ORIGINAL images
    # -----------------------------
    for file in sorted(os.listdir(original_path)):

        if not file.lower().endswith(valid_ext):
            continue

        if only_clean:
            if file.endswith("_O.png") or file.endswith("_O.jpg"):
                original_images.append(os.path.join(original_path, file))
        else:
            if "_O" in file:
                original_images.append(os.path.join(original_path, file))

    # -----------------------------
    # Load FORGED images
    # -----------------------------
    for file in sorted(os.listdir(forged_path)):

        if not file.lower().endswith(valid_ext):
            continue

        if only_clean:
            if file.endswith("_F.png") or file.endswith("_F.jpg"):
                forged_images.append(os.path.join(forged_path, file))
        else:
            if "_F" in file:
                forged_images.append(os.path.join(forged_path, file))

    # -----------------------------
    # Apply limit (balanced)
    # -----------------------------
    if limit is not None:
        half = limit // 2
        original_images = original_images[:half]
        forged_images = forged_images[:half]

    # Combine
    image_paths = original_images + forged_images
    labels = [0]*len(original_images) + [1]*len(forged_images)

    # -----------------------------
    # Print Summary
    # -----------------------------
    print("====================================")
    print(" Dataset Loaded Successfully ")
    print(f" Total Images: {len(image_paths)}")
    print(f" Originals: {len(original_images)}")
    print(f" Forged: {len(forged_images)}")
    print("====================================")

    return image_paths, labels