"""
Full evaluation of a trained autoencoder checkpoint against the MVTec AD
tile test set.

Fixes vs. the original notebook:
  - Evaluates ALL defect categories (crack, glue_strip, gray_stroke, oil,
    rough), not just "good vs rough".
  - Replaces the hardcoded threshold (0.01) with two calibrated options:
      1. "unsupervised" threshold: a percentile of the held-out good
         images' reconstruction error (no defect labels needed --
         this is what you'd actually use in production).
      2. "supervised" threshold: the value that maximizes F1 (or Youden's
         J) on the labeled test set -- useful for reporting best-case
         performance, not for picking blind in deployment.
  - Reports ROC-AUC, PR-AUC, per-class accuracy, and a full confusion
    matrix + classification report across all classes.
  - Computes pixel-level anomaly heatmaps and scores them against the
    ground-truth defect masks (pixel-level ROC-AUC), which is the
    standard MVTec AD localization metric and is what the GUI's heatmap
    overlay is based on.

Usage:
    python evaluate.py --data-root /path/to/tile --checkpoint best_baseline_autoencoder.pth
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix, classification_report,
    accuracy_score, f1_score,
)

from dataset import TileDefectDataset, get_full_test_set, get_good_split, TileDataset
from model import load_autoencoder


def per_image_errors(model, loader, device):
    """Returns (errors, labels, pixel_error_maps, masks) for a loader of
    (input, target, mask, label) tuples."""
    errors, labels, pixel_maps, masks = [], [], [], []
    with torch.no_grad():
        for inputs, targets, mask, label in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            per_pixel = (outputs - targets) ** 2  # [B, 3, H, W]
            per_image = per_pixel.mean(dim=(1, 2, 3))  # [B]

            errors.extend(per_image.cpu().numpy())
            labels.extend(label.numpy() if torch.is_tensor(label) else list(label))
            pixel_maps.append(per_pixel.mean(dim=1).cpu().numpy())  # [B, H, W]
            masks.append(mask.squeeze(1).numpy())  # [B, H, W]

    return (
        np.array(errors),
        np.array(labels),
        np.concatenate(pixel_maps, axis=0),
        np.concatenate(masks, axis=0),
    )


def find_best_f1_threshold(errors, labels):
    thresholds = np.unique(errors)
    best_f1, best_t = -1.0, thresholds[0]
    for t in thresholds:
        preds = (errors > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def find_youden_threshold(errors, labels):
    fpr, tpr, thresholds = roc_curve(labels, errors)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return thresholds[best_idx]


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_autoencoder(args.checkpoint, device=device)

    # -- Calibrate an unsupervised threshold from held-out good images --
    splits = get_good_split(args.data_root, seed=args.seed)
    calib_ds = TileDataset(splits["test_good_images"], splits["good_dir"])
    calib_loader = DataLoader(calib_ds, batch_size=args.batch_size, shuffle=False)

    calib_errors = []
    with torch.no_grad():
        for inputs, targets in calib_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            err = ((outputs - targets) ** 2).mean(dim=(1, 2, 3))
            calib_errors.extend(err.cpu().numpy())
    calib_errors = np.array(calib_errors)

    unsupervised_threshold = float(np.percentile(calib_errors, args.percentile))
    print(f"Unsupervised threshold ({args.percentile}th percentile of held-out good errors): "
          f"{unsupervised_threshold:.6f}")

    # -- Build the full labeled test set (all defect classes + good) --
    per_class = get_full_test_set(args.data_root)

    all_errors, all_labels, all_pixel_maps, all_masks, class_names = [], [], [], [], []
    per_class_report = {}

    for class_name, (image_paths, mask_paths) in per_class.items():
        label = 0 if class_name == "good" else 1
        ds = TileDefectDataset(image_paths, mask_paths, label=label)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        errors, labels, pixel_maps, masks = per_image_errors(model, loader, device)
        all_errors.append(errors)
        all_labels.append(labels)
        all_pixel_maps.append(pixel_maps)
        all_masks.append(masks)
        class_names.extend([class_name] * len(errors))

        per_class_report[class_name] = {
            "n_images": len(errors),
            "mean_error": float(errors.mean()),
            "min_error": float(errors.min()),
            "max_error": float(errors.max()),
        }

    all_errors = np.concatenate(all_errors)
    all_labels = np.concatenate(all_labels)
    all_pixel_maps = np.concatenate(all_pixel_maps, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # -- Image-level metrics --
    roc_auc = roc_auc_score(all_labels, all_errors)
    pr_auc = average_precision_score(all_labels, all_errors)
    supervised_threshold, best_f1 = find_best_f1_threshold(all_errors, all_labels)
    youden_threshold = float(find_youden_threshold(all_errors, all_labels))

    print(f"\nImage-level ROC-AUC: {roc_auc:.4f}")
    print(f"Image-level PR-AUC:  {pr_auc:.4f}")
    print(f"F1-optimal threshold: {supervised_threshold:.6f} (F1={best_f1:.4f})")
    print(f"Youden's-J threshold: {youden_threshold:.6f}")

    results = {}
    for name, threshold in [
        ("unsupervised (percentile)", unsupervised_threshold),
        ("supervised (F1-optimal)", supervised_threshold),
        ("supervised (Youden's J)", youden_threshold),
    ]:
        preds = (all_errors > threshold).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)
        cm = confusion_matrix(all_labels, preds).tolist()
        report = classification_report(all_labels, preds, target_names=["good", "defective"],
                                        zero_division=0, output_dict=True)
        results[name] = {"threshold": float(threshold), "accuracy": acc, "f1": f1,
                          "confusion_matrix": cm, "classification_report": report}
        print(f"\n--- Threshold strategy: {name} ({threshold:.6f}) ---")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:", cm)

    # -- Pixel-level localization (only meaningful for defective images with a mask) --
    defective_mask_idx = np.array(class_names) != "good"
    if defective_mask_idx.any() and all_masks[defective_mask_idx].sum() > 0:
        pixel_scores = all_pixel_maps[defective_mask_idx].reshape(-1)
        pixel_labels = (all_masks[defective_mask_idx].reshape(-1) > 0.5).astype(int)
        pixel_auc = roc_auc_score(pixel_labels, pixel_scores)
        print(f"\nPixel-level localization ROC-AUC (defect regions vs. ground truth masks): {pixel_auc:.4f}")
    else:
        pixel_auc = None

    # -- Save everything the GUI needs --
    calibration = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "image_level_roc_auc": roc_auc,
        "image_level_pr_auc": pr_auc,
        "pixel_level_roc_auc": pixel_auc,
        "recommended_threshold": unsupervised_threshold,
        "thresholds": {
            "unsupervised_percentile": unsupervised_threshold,
            "supervised_f1_optimal": float(supervised_threshold),
            "supervised_youden_j": youden_threshold,
        },
        "per_class_report": per_class_report,
        "detailed_results": results,
    }

    with open(args.out, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nSaved calibration + full report to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to the MVTec 'tile' folder")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="calibration.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--percentile", type=float, default=95.0,
                         help="Percentile of held-out good-image errors used for the unsupervised threshold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args)