import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
import os


class SegmentationAugmenter:
    def __init__(self, image, mask):
        self.image = image
        self.mask = mask

    def random_flip(self, horizontal=True, vertical=False):
        img, msk = self.image.copy(), self.mask.copy()

        if horizontal and random.random() > 0.5:
            img = cv2.flip(img, 1)
            msk = cv2.flip(msk, 1)

        if vertical and random.random() > 0.5:
            img = cv2.flip(img, 0)
            msk = cv2.flip(msk, 0)

        return img, msk

    def random_rotation(self, max_angle=30):
        angle = random.uniform(-max_angle, max_angle)
        h, w = self.image.shape[:2]

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)

        img = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_LINEAR)
        msk = cv2.warpAffine(self.mask, M, (w, h), flags=cv2.INTER_NEAREST)

        return img, msk

    def elastic_deformation(self, alpha=20, sigma=3):
        shape = self.image.shape[:2]

        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        img = np.zeros_like(self.image)
        for i in range(3):
            img[:, :, i] = map_coordinates(
                self.image[:, :, i], indices, order=1
            ).reshape(shape)

        msk = map_coordinates(self.mask, indices, order=0).reshape(shape)

        return img, msk

    def augment_batch(self, n=3):
        results = []
        for _ in range(n):
            img, msk = self.random_flip()
            if random.random() > 0.5:
                img, msk = self.random_rotation()
            results.append((img, msk))
        return results

    @staticmethod
    def visualize_segmentation_mask(mask, num_classes):
        np.random.seed(42)
        colors = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]

        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for cid in range(num_classes):
            color_mask[mask == cid] = colors[cid]

        return color_mask

    @staticmethod
    def overlay_mask_on_image(image, mask, num_classes, alpha=0.5):
        """
        Overlay mask on original image (no black background)
        """
        color_mask = SegmentationAugmenter.visualize_segmentation_mask(
            mask, num_classes
        )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()

        mask_region = mask != 0
        overlay[mask_region] = (
            image_rgb[mask_region] * (1 - alpha)
            + color_mask[mask_region] * alpha
        ).astype(np.uint8)

        return overlay

    def visualize_augmentations(self, num_classes=5, n_augmentations=3,
                                save_path="augmentation_examples/augmentation_visual_check.png"):

        augmented = self.augment_batch(n_augmentations)
        rows = n_augmentations + 1

        fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))

        # Original
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")

        overlay_orig = self.overlay_mask_on_image(
            self.image, self.mask, num_classes
        )
        axes[0, 1].imshow(overlay_orig)
        axes[0, 1].set_title("Original Mask Overlay")

        # Augmented
        for i, (img, msk) in enumerate(augmented):
            axes[i + 1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[i + 1, 0].set_title(f"Augmented Image {i + 1}")

            overlay_aug = self.overlay_mask_on_image(
                img, msk, num_classes
            )
            axes[i + 1, 1].imshow(overlay_aug)
            axes[i + 1, 1].set_title(f"Augmented Mask Overlay {i + 1}")

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path)   
        plt.show()

        print(f"Saved visualization → {save_path}")


image = cv2.imread("sample_data/images/img1.jpg")
mask = cv2.imread("sample_data/masks/img1.png", cv2.IMREAD_GRAYSCALE)

augmenter = SegmentationAugmenter(image, mask)
augmenter.visualize_augmentations(
    num_classes=len(np.unique(mask)),
    n_augmentations=3
)
