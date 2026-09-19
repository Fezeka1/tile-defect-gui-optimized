"""
Full evaluation of the optimized autoencoder checkpoint against
the MVTec AD tile test set.

The script evaluates two image-level anomaly scoring methods:

1. mean
   Mean reconstruction error across the full image.

2. top1_mean
   Mean reconstruction error of the highest-error 1% of pixels.

Both methods are evaluated using the same dataset, split and metrics
as the baseline model so that the results can be compared fairly.

Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - PR-AUC
    - Confusion matrix
    - Classification report
    - Pixel-level ROC-AUC

Usage:
    python evaluate.py --data-root data/tile --checkpoint optimization2_model.pth
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from dataset import (
    TileDefectDataset,
    get_full_test_set,
    get_good_split,
    TileDataset,
)

from model import load_autoencoder


# ============================================================
# SCORING METHODS
# ============================================================

def mean_image_score(pixel_maps):
    """
    Mean reconstruction error across the entire image.

    pixel_maps shape:
        [B, H, W]
    """

    return pixel_maps.mean(axis=(1, 2))


def top1_mean_image_score(pixel_maps):
    """
    Mean of the highest-error 1% of pixels in each image.

    This emphasizes small/localized anomalies instead of allowing
    their error to be diluted by the normal background.

    pixel_maps shape:
        [B, H, W]
    """

    batch_scores = []

    for pixel_map in pixel_maps:

        flat = pixel_map.reshape(-1)

        # Number of pixels corresponding to 1%
        k = max(
            1,
            int(np.ceil(flat.size * 0.01))
        )

        # Get the k largest errors
        top_values = np.partition(
            flat,
            -k
        )[-k:]

        batch_scores.append(
            float(top_values.mean())
        )

    return np.array(batch_scores)


# ============================================================
# MODEL INFERENCE
# ============================================================

def per_image_errors(model, loader, device):
    """
    Run inference and return reconstruction-error maps together
    with labels and ground-truth masks.

    Returns:
        mean_errors
        top1_errors
        labels
        pixel_error_maps
        masks
    """

    mean_errors = []
    top1_errors = []
    labels = []
    pixel_maps = []
    masks = []

    with torch.no_grad():

        for inputs, targets, mask, label in loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Reconstruction squared error
            per_pixel_rgb = (
                outputs - targets
            ) ** 2

            # Average RGB channels
            #
            # Shape:
            # [B, H, W]
            batch_pixel_maps = (
                per_pixel_rgb
                .mean(dim=1)
                .cpu()
                .numpy()
            )

            batch_mean = mean_image_score(
                batch_pixel_maps
            )

            batch_top1 = top1_mean_image_score(
                batch_pixel_maps
            )

            mean_errors.extend(
                batch_mean
            )

            top1_errors.extend(
                batch_top1
            )

            if torch.is_tensor(label):
                labels.extend(
                    label.numpy()
                )
            else:
                labels.extend(
                    list(label)
                )

            pixel_maps.append(
                batch_pixel_maps
            )

            masks.append(
                mask.squeeze(1).numpy()
            )

    return (
        np.array(mean_errors),
        np.array(top1_errors),
        np.array(labels),
        np.concatenate(pixel_maps, axis=0),
        np.concatenate(masks, axis=0),
    )


# ============================================================
# THRESHOLD FUNCTIONS
# ============================================================

def find_best_f1_threshold(errors, labels):
    """
    Find the threshold that maximizes F1-score.

    This is supervised because defect labels are used.
    """

    thresholds = np.unique(errors)

    best_f1 = -1.0
    best_threshold = thresholds[0]

    for threshold in thresholds:

        predictions = (
            errors > threshold
        ).astype(int)

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_f1)
    )


def find_youden_threshold(errors, labels):
    """
    Find threshold maximizing Youden's J:

        J = TPR - FPR
    """

    fpr, tpr, thresholds = roc_curve(
        labels,
        errors
    )

    j_scores = tpr - fpr

    best_index = int(
        np.argmax(j_scores)
    )

    return float(
        thresholds[best_index]
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    errors,
    labels,
    threshold
):
    predictions = (
        errors > threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        labels,
        predictions
    ).tolist()

    report = classification_report(
        labels,
        predictions,
        target_names=[
            "good",
            "defective"
        ],
        zero_division=0,
        output_dict=True
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ============================================================
# EVALUATE ONE SCORING METHOD
# ============================================================

def evaluate_scoring_method(
    name,
    all_errors,
    all_labels,
    calibration_errors,
    percentile
):
    print("\n")
    print("=" * 70)
    print(f"SCORING METHOD: {name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Unsupervised threshold
    # --------------------------------------------------------

    unsupervised_threshold = float(
        np.percentile(
            calibration_errors,
            percentile
        )
    )

    # --------------------------------------------------------
    # Supervised thresholds
    # --------------------------------------------------------

    f1_threshold, best_f1 = (
        find_best_f1_threshold(
            all_errors,
            all_labels
        )
    )

    youden_threshold = (
        find_youden_threshold(
            all_errors,
            all_labels
        )
    )

    # --------------------------------------------------------
    # Threshold-independent metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        all_labels,
        all_errors
    )

    pr_auc = average_precision_score(
        all_labels,
        all_errors
    )

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    print(
        f"Unsupervised threshold "
        f"({percentile}th percentile): "
        f"{unsupervised_threshold:.6f}"
    )

    print(
        f"F1-optimal threshold: "
        f"{f1_threshold:.6f} "
        f"(F1={best_f1:.4f})"
    )

    print(
        f"Youden's-J threshold: "
        f"{youden_threshold:.6f}"
    )

    # --------------------------------------------------------
    # Evaluate all threshold strategies
    # --------------------------------------------------------

    results = {}

    strategies = [
        (
            "unsupervised_percentile",
            unsupervised_threshold
        ),
        (
            "supervised_f1_optimal",
            f1_threshold
        ),
        (
            "supervised_youden_j",
            youden_threshold
        ),
    ]

    for strategy_name, threshold in strategies:

        result = evaluate_threshold(
            all_errors,
            all_labels,
            threshold
        )

        results[strategy_name] = result

        print(
            f"\n--- {strategy_name} ---"
        )

        print(
            f"Threshold : "
            f"{threshold:.6f}"
        )

        print(
            f"Accuracy  : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"Precision : "
            f"{result['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{result['recall']:.4f}"
        )

        print(
            f"F1-score  : "
            f"{result['f1']:.4f}"
        )

        print(
            "Confusion matrix "
            "[[TN, FP], [FN, TP]]:"
        )

        print(
            result[
                "confusion_matrix"
            ]
        )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),

        "thresholds": {
            "unsupervised_percentile":
                unsupervised_threshold,

            "supervised_f1_optimal":
                f1_threshold,

            "supervised_youden_j":
                youden_threshold,
        },

        "results":
            results,
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        f"\nLoading checkpoint: "
        f"{args.checkpoint}"
    )

    model = load_autoencoder(
        args.checkpoint,
        device=device
    )

    print(
        "Optimized model loaded successfully."
    )

    # ========================================================
    # CALIBRATION SET
    # ========================================================

    print(
        "\nPreparing held-out good "
        "calibration images..."
    )

    splits = get_good_split(
        args.data_root,
        seed=args.seed
    )

    calib_ds = TileDataset(
        splits["test_good_images"],
        splits["good_dir"]
    )

    calib_loader = DataLoader(
        calib_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    calibration_mean = []
    calibration_top1 = []

    with torch.no_grad():

        for inputs, targets in calib_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            per_pixel_rgb = (
                outputs - targets
            ) ** 2

            pixel_maps = (
                per_pixel_rgb
                .mean(dim=1)
                .cpu()
                .numpy()
            )

            calibration_mean.extend(
                mean_image_score(
                    pixel_maps
                )
            )

            calibration_top1.extend(
                top1_mean_image_score(
                    pixel_maps
                )
            )

    calibration_mean = np.array(
        calibration_mean
    )

    calibration_top1 = np.array(
        calibration_top1
    )

    # ========================================================
    # FULL TEST SET
    # ========================================================

    print(
        "\nLoading complete MVTec Tile test set..."
    )

    per_class = get_full_test_set(
        args.data_root
    )

    all_mean_errors = []
    all_top1_errors = []
    all_labels = []

    all_pixel_maps = []
    all_masks = []

    class_names = []

    per_class_report = {}

    for class_name, (
        image_paths,
        mask_paths
    ) in per_class.items():

        label = (
            0
            if class_name == "good"
            else 1
        )

        dataset = TileDefectDataset(
            image_paths,
            mask_paths,
            label=label
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        (
            mean_errors,
            top1_errors,
            labels,
            pixel_maps,
            masks,
        ) = per_image_errors(
            model,
            loader,
            device
        )

        all_mean_errors.append(
            mean_errors
        )

        all_top1_errors.append(
            top1_errors
        )

        all_labels.append(
            labels
        )

        all_pixel_maps.append(
            pixel_maps
        )

        all_masks.append(
            masks
        )

        class_names.extend(
            [class_name] * len(labels)
        )

        per_class_report[class_name] = {
            "n_images":
                int(len(labels)),

            "mean_full_image_error":
                float(
                    mean_errors.mean()
                ),

            "mean_top1_error":
                float(
                    top1_errors.mean()
                ),

            "minimum_top1_error":
                float(
                    top1_errors.min()
                ),

            "maximum_top1_error":
                float(
                    top1_errors.max()
                ),
        }

        print(
            f"{class_name:<15} "
            f"images={len(labels):3d} "
            f"mean={mean_errors.mean():.6f} "
            f"top1={top1_errors.mean():.6f}"
        )

    # Combine classes

    all_mean_errors = np.concatenate(
        all_mean_errors
    )

    all_top1_errors = np.concatenate(
        all_top1_errors
    )

    all_labels = np.concatenate(
        all_labels
    )

    all_pixel_maps = np.concatenate(
        all_pixel_maps,
        axis=0
    )

    all_masks = np.concatenate(
        all_masks,
        axis=0
    )

    # ========================================================
    # EVALUATE BOTH SCORING METHODS
    # ========================================================

    mean_results = evaluate_scoring_method(

        name="mean",

        all_errors=
            all_mean_errors,

        all_labels=
            all_labels,

        calibration_errors=
            calibration_mean,

        percentile=
            args.percentile
    )

    top1_results = evaluate_scoring_method(

        name="top1_mean",

        all_errors=
            all_top1_errors,

        all_labels=
            all_labels,

        calibration_errors=
            calibration_top1,

        percentile=
            args.percentile
    )

    # ========================================================
    # PIXEL-LEVEL LOCALIZATION
    # ========================================================

    defective_indices = (
        np.array(class_names)
        != "good"
    )

    if (
        defective_indices.any()
        and
        all_masks[
            defective_indices
        ].sum() > 0
    ):

        pixel_scores = (
            all_pixel_maps[
                defective_indices
            ]
            .reshape(-1)
        )

        pixel_labels = (
            all_masks[
                defective_indices
            ]
            .reshape(-1)
            > 0.5
        ).astype(int)

        pixel_auc = roc_auc_score(
            pixel_labels,
            pixel_scores
        )

        print("\n")
        print("=" * 70)
        print("PIXEL-LEVEL LOCALIZATION")
        print("=" * 70)

        print(
            f"Pixel-level ROC-AUC: "
            f"{pixel_auc:.4f}"
        )

    else:

        pixel_auc = None

        print(
            "\nNo valid defect masks found "
            "for pixel-level evaluation."
        )

    # ========================================================
    # CHOOSE RECOMMENDED SCORING METHOD
    # ========================================================

    if (
        top1_results[
            "results"
        ][
            "supervised_f1_optimal"
        ][
            "f1"
        ]
        >=
        mean_results[
            "results"
        ][
            "supervised_f1_optimal"
        ][
            "f1"
        ]
    ):

        recommended_method = (
            "top1_mean"
        )

        selected_results = (
            top1_results
        )

    else:

        recommended_method = (
            "mean"
        )

        selected_results = (
            mean_results
        )

    print("\n")
    print("=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)

    print(
        f"Recommended scoring method: "
        f"{recommended_method}"
    )

    print(
        "Recommended deployment threshold "
        "(unsupervised): "
        f"{selected_results['thresholds']['unsupervised_percentile']:.6f}"
    )

    print(
        "Balanced/reporting threshold "
        "(Youden's J): "
        f"{selected_results['thresholds']['supervised_youden_j']:.6f}"
    )

    # ========================================================
    # SAVE CALIBRATION JSON
    # ========================================================

    calibration = {

        "checkpoint":
            os.path.abspath(
                args.checkpoint
            ),

        "recommended_scoring_method":
            recommended_method,

        "recommended_threshold":
            selected_results[
                "thresholds"
            ][
                "unsupervised_percentile"
            ],

        "recommended_balanced_threshold":
            selected_results[
                "thresholds"
            ][
                "supervised_youden_j"
            ],

        "pixel_level_roc_auc":
            (
                float(pixel_auc)
                if pixel_auc is not None
                else None
            ),

        "mean":
            mean_results,

        "top1_mean":
            top1_results,

        "per_class_report":
            per_class_report,
    }

    with open(
        args.out,
        "w"
    ) as file:

        json.dump(
            calibration,
            file,
            indent=2
        )

    print(
        f"\nSaved complete evaluation to:"
    )

    print(
        args.out
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help=(
            "Path to the MVTec "
            "'tile' folder"
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pth model"
    )

    parser.add_argument(
        "--out",
        default="calibration.json"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    evaluate(args)