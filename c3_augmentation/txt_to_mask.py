import cv2
import numpy as np
import os

def txt_to_mask(txt_path, image_shape):
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)

    with open(txt_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]

            polygon = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w)
                y = int(coords[i + 1] * h)
                polygon.append([x, y])

            polygon = np.array([polygon], dtype=np.int32)
            cv2.fillPoly(mask, polygon, class_id)

    return mask


def remap_mask(mask):
    unique_vals = np.unique(mask)
    mapping = {val: idx for idx, val in enumerate(unique_vals)}

    remapped = np.zeros_like(mask)
    for old, new in mapping.items():
        remapped[mask == old] = new

    return remapped, mapping


IMAGE_DIR = "sample_data/images"
LABEL_DIR = "sample_data/labels"
MASK_DIR = "sample_data/masks"

os.makedirs(MASK_DIR, exist_ok=True)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    base_name = os.path.splitext(img_name)[0]

    img_path = os.path.join(IMAGE_DIR, img_name)
    txt_path = os.path.join(LABEL_DIR, base_name + ".txt")

    if not os.path.exists(txt_path):
        print(f"Label missing for {img_name}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Cannot read image {img_name}")
        continue

    mask = txt_to_mask(txt_path, image.shape[:2])
    mask, class_map = remap_mask(mask)

    save_path = os.path.join(MASK_DIR, base_name + ".png")
    cv2.imwrite(save_path, mask)

    print(f"{img_name} → classes:", np.unique(mask))
    print("Mapping:", class_map)

print("All masks generated.")