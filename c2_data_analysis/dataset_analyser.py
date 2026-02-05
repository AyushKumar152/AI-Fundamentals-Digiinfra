import numpy as np
import cv2
import os

def calculate_class_distribution(mask_folder):
    class_counts = {}
    total_pixels = 0

    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

    for file in os.listdir(mask_folder):

        # Only process image files
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        mask_path = os.path.join(mask_folder, file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"⚠️ Skipping unreadable mask: {file}")
            continue

        unique, counts = np.unique(mask, return_counts=True)

        for u, c in zip(unique, counts):
            class_counts[u] = class_counts.get(u, 0) + c
            total_pixels += c

    if total_pixels == 0:
        raise ValueError("No valid pixels found in masks")

    return {int(k): (v / total_pixels) * 100 for k, v in class_counts.items()}

def find_class_imbalance_ratio(mask_folder):
    dist = calculate_class_distribution(mask_folder)

    if not dist or len(dist) <= 1:
        return 0

    if 0 in dist:
        dist = {k: v for k, v in dist.items() if k != 0}

    if not dist:
        return 0

    percentages = list(dist.values())
    min_pct = min(percentages)
    max_pct = max(percentages)

    if min_pct == 0:
        return float("inf")

    return max_pct / min_pct

def calculate_mean_object_size(mask_folder):
    
    class_sizes = {}  

    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

    for filename in os.listdir(mask_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue

        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Skipping unreadable mask: {filename}")
        continue


        for cid in np.unique(mask):
            if cid == 0:
                continue 

            binary = (mask == cid).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

            areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]

            if areas:
                class_sizes.setdefault(int(cid), []).extend(areas)

    mean_sizes = {
        cid: float(np.mean(areas))
        for cid, areas in class_sizes.items()
        if areas
    }

    return mean_sizes

def detect_small_objects(mask_folder, threshold=32):
    suspicious_masks = []

    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

    for filename in os.listdir(mask_folder):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Skipping unreadable mask: {filename}")
            continue

        unique, counts = np.unique(mask, return_counts=True)

        for val, count in zip(unique, counts):
            if val == 0:
                continue

            if count < threshold:
                suspicious_masks.append({
                    "file": filename,
                    "class_id": int(val),
                    "pixel_count": int(count)
                })
                break

    return suspicious_masks

def calculate_class_cooccurrence(mask_folder, num_classes):
   
    if not os.path.exists(mask_folder):
        raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for filename in os.listdir(mask_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue

        mask_path = os.path.join(mask_folder, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping unreadable mask: {filename}")
            continue


        present = np.unique(mask)

        for i in present:
            for j in present:
                if i < num_classes and j < num_classes:
                    matrix[int(i), int(j)] += 1

    return matrix


if __name__ == "__main__":
    stats = calculate_class_distribution("sample_data/masks")
    print("Class Distribution:", stats)

    ratio = find_class_imbalance_ratio("sample_data/masks")
    print("Imbalance Ratio:", ratio)

    mean_sizes = calculate_mean_object_size("sample_data/masks")
    print("Mean Object Sizes:", mean_sizes)

    small = detect_small_objects("sample_data/masks")
    print("Small Object Masks:", small)

    co_matrix = calculate_class_cooccurrence("sample_data/masks",num_classes=3)
    print("Cooccurrence Matrices:", co_matrix)

