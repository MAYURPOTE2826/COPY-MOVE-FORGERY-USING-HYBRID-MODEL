
import os
import cv2
from src.preprocessing import load_image
from src.dataset_loader import load_dataset
from src.evaluation import evaluate
from src.dct_detector import dct_copy_move_detection


def main():

    image_paths, labels = load_dataset(
        dataset_path="dataset",
        only_clean=True,
        limit=None   # Use full dataset
    )

    predictions = []

    os.makedirs("results", exist_ok=True)

    print("\nStarting Optimized Classical DCT Detection...\n")

    for idx, path in enumerate(image_paths):

        image, gray = load_image(path)

        dct_cluster = dct_copy_move_detection(gray)

        # Tuned Threshold (adjust slightly if needed)
        if dct_cluster >= 9:
            prediction = 1
            label_text = "Forged"
            color = (0, 0, 255)
        else:
            prediction = 0
            label_text = "Original"
            color = (0, 255, 0)

        predictions.append(prediction)

        result = image.copy()

        cv2.putText(result,
                    f"{label_text} | Cluster: {dct_cluster}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2)

        filename = os.path.basename(path)
        cv2.imwrite(os.path.join("results", filename), result)

        print(f"[{idx+1}/{len(image_paths)}] {filename} → "
              f"Cluster:{dct_cluster} → {label_text}")

    acc, prec, rec, f1 = evaluate(labels, predictions)

    print("\n================ FINAL REPORT ================")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("==============================================\n")

    print("✅ Optimized Classical DCT Detection Completed Successfully!")
    print("Check 'results' folder for output images.")


if __name__ == "__main__":
    main()