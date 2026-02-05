import numpy as np

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    for t, p in zip(y_true_flat, y_pred_flat):
        if t < num_classes and p < num_classes:
            matrix[t, p] += 1
    return matrix


def calculate_iou(mask_true, mask_pred, class_id):
    true_class = (mask_true == class_id)
    pred_class = (mask_pred == class_id)

    intersection = np.logical_and(true_class, pred_class).sum()
    union = np.logical_or(true_class, pred_class).sum()

    return 1.0 if union == 0 else intersection / union


def calculate_dice_coefficient(mask_true, mask_pred, class_id):
    true_class = (mask_true == class_id)
    pred_class = (mask_pred == class_id)

    intersection = np.logical_and(true_class, pred_class).sum()
    sum_areas = true_class.sum() + pred_class.sum()

    return 1.0 if sum_areas == 0 else (2 * intersection) / sum_areas


def calculate_mean_iou(mask_true, mask_pred, num_classes):
    ious = []
    for cid in range(num_classes):
        ious.append(calculate_iou(mask_true, mask_pred, cid))
    return np.mean(ious)


def precision_recall_f1(y_true, y_pred, class_id):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_pred == class_id) & (y_true == class_id))
    fp = np.sum((y_pred == class_id) & (y_true != class_id))
    fn = np.sum((y_pred != class_id) & (y_true == class_id))

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1

if __name__ == "__main__":

    # Dummy segmentation masks
    mask_true = np.array([
        [0, 1, 1],
        [0, 2, 2],
        [0, 2, 2]
    ])

    mask_pred = np.array([
        [0, 1, 0],
        [0, 2, 2],
        [0, 2, 1]
    ])

    num_classes = 3  # background + 2 classes

    # 1️⃣ Confusion Matrix
    cm = calculate_confusion_matrix(mask_true, mask_pred, num_classes)
    print("Confusion Matrix:\n", cm)

    # 2️⃣ IoU & Dice (per class)
    for cid in range(num_classes):
        iou = calculate_iou(mask_true, mask_pred, cid)
        dice = calculate_dice_coefficient(mask_true, mask_pred, cid)
        print(f"Class {cid} -> IoU: {iou:.3f}, Dice: {dice:.3f}")

    # 3️⃣ Mean IoU
    miou = calculate_mean_iou(mask_true, mask_pred, num_classes)
    print("Mean IoU:", miou)

    # 4️⃣ Precision / Recall / F1 (example for class 1)
    p, r, f1 = precision_recall_f1(
        mask_true.flatten(),
        mask_pred.flatten(),
        class_id=1
    )
    print(f"Class 1 -> Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
