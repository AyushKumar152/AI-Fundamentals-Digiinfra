import os
import cv2
import numpy as np

def rotate_image_90(image):
    n = len(image)

    for i in range(n):
        for j in range(i + 1, n):
            image[i][j], image[j][i] = image[j][i], image[i][j]

    for i in range(n):
        image[i].reverse()


matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

rotate_image_90(matrix)
print("Rotated matrix:")
print(matrix)

def manual_rgb_to_grayscale(image):
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]

    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    return gray.astype(np.uint8)


def apply_custom_kernel(image, kernel):
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2

    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            roi = padded[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            output[i, j] = np.sum(roi * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)


def create_image_pyramid(image, levels=4):
    pyramid = [image]
    current = image

    for _ in range(1, levels):
        h, w = current.shape[:2]
        current = cv2.resize(current, (w // 2, h // 2))
        pyramid.append(current)

    return pyramid

IMAGE_DIR = "sample_data/images"
OUTPUT_DIR = "test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

edge_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Skipping {img_name}")
        continue

    base = os.path.splitext(img_name)[0]

    # Grayscale
    gray = manual_rgb_to_grayscale(image)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base}_gray.jpg"),
        gray
    )

    # Edge detection
    edges = apply_custom_kernel(gray, edge_kernel)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, f"{base}_edges.jpg"),
        edges
    )

    # Image pyramid
    pyramid = create_image_pyramid(image)
    for i, p in enumerate(pyramid):
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, f"{base}_pyramid_{i}.jpg"),
            p
        )

    print(f"Processed -> {img_name}")

def visualize_segmentation_mask_v2(mask, num_classes):
    np.random.seed(42)

    colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    return colors[mask]


mask = np.zeros((256, 256), dtype=np.uint8)
mask[40:140, 40:140] = 1
mask[120:220, 120:220] = 2

color_mask = visualize_segmentation_mask_v2(mask, num_classes=3)
cv2.imwrite("segmentation_visualization_v2.png", color_mask)
