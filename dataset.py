"""
Dataset utilities for the MVTec AD "tile" category.

Expected directory layout:

<root>/
    train/
        good/
            *.png

    test/
        good/
            *.png

        crack/
            *.png

        glue_strip/
            *.png

        gray_stroke/
            *.png

        oil/
            *.png

        rough/
            *.png

    ground_truth/
        crack/
            *_mask.png
        glue_strip/
            *_mask.png
        gray_stroke/
            *_mask.png
        oil/
            *_mask.png
        rough/
            *_mask.png
"""

import os
import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ============================================================
# IMAGE SIZE
# ============================================================

# Optimization Model 2 uses 128 x 128 images.
IMG_SIZE = 128


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

# Encoder input:
#
# 1. Resize to 128 x 128
# 2. Convert image to tensor
# 3. Apply ImageNet normalization because the encoder is
#    an ImageNet-pretrained ResNet18.
encoder_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    ),
])


# Reconstruction target:
#
# Decoder output uses Sigmoid, so target values remain
# in the [0, 1] range.
target_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),
])


# Ground-truth mask:
#
# Nearest-neighbour interpolation preserves binary mask values.
mask_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE),

        interpolation=(
            transforms
            .InterpolationMode
            .NEAREST
        )
    ),

    transforms.ToTensor(),
])


# ============================================================
# DATA AUGMENTATION
# ============================================================

# Augmentation is applied to the source PIL image BEFORE
# encoder_transform and target_transform so that the input
# and reconstruction target stay aligned.
augment_transform = transforms.Compose([

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
])


# ============================================================
# NORMAL TILE DATASET
# ============================================================

class TileDataset(Dataset):
    """
    Dataset used for normal tile images.

    Returns:

        encoder_input,
        reconstruction_target
    """

    def __init__(
        self,
        image_names,
        image_dir,
        augment: bool = False
    ):

        self.image_names = image_names
        self.image_dir = image_dir
        self.augment = augment


    def __len__(self):

        return len(
            self.image_names
        )


    def __getitem__(
        self,
        index
    ):

        image_name = (
            self.image_names[
                index
            ]
        )

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = (
            Image
            .open(
                image_path
            )
            .convert("RGB")
        )


        # ----------------------------------------------------
        # Apply augmentation during training if enabled
        # ----------------------------------------------------

        if self.augment:

            image = augment_transform(
                image
            )


        # ----------------------------------------------------
        # Encoder input
        # ----------------------------------------------------

        encoder_image = encoder_transform(
            image
        )


        # ----------------------------------------------------
        # Reconstruction target
        # ----------------------------------------------------

        target_image = target_transform(
            image
        )


        return (
            encoder_image,
            target_image
        )


# ============================================================
# DEFECT EVALUATION DATASET
# ============================================================

class TileDefectDataset(Dataset):
    """
    Dataset used during evaluation.

    Returns:

        encoder_input,
        reconstruction_target,
        ground_truth_mask,
        label

    Labels:

        0 = good
        1 = defective

    Good images receive an all-zero mask because there is
    no ground-truth defect region.
    """

    def __init__(
        self,
        image_paths,
        mask_paths=None,
        label: int = 0
    ):

        self.image_paths = image_paths

        self.mask_paths = mask_paths

        self.label = label


    def __len__(self):

        return len(
            self.image_paths
        )


    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = (
            Image
            .open(
                self.image_paths[
                    index
                ]
            )
            .convert("RGB")
        )


        # ----------------------------------------------------
        # Prepare encoder input
        # ----------------------------------------------------

        encoder_image = encoder_transform(
            image
        )


        # ----------------------------------------------------
        # Prepare reconstruction target
        # ----------------------------------------------------

        target_image = target_transform(
            image
        )


        # ----------------------------------------------------
        # Ground-truth mask
        # ----------------------------------------------------

        if (
            self.mask_paths is not None
            and
            self.mask_paths[
                index
            ] is not None
        ):

            mask = (
                Image
                .open(
                    self.mask_paths[
                        index
                    ]
                )
                .convert("L")
            )


            mask = mask_transform(
                mask
            )


            # Convert to strict binary mask
            mask = (
                mask > 0.5
            ).float()


        else:

            # Good image -> no defective pixels
            mask = torch.zeros(
                (
                    1,
                    IMG_SIZE,
                    IMG_SIZE
                )
            )


        return (
            encoder_image,
            target_image,
            mask,
            self.label
        )


# ============================================================
# GOOD-IMAGE SPLIT
# ============================================================

def get_good_split(
    tile_root: str,
    seed: int = 42,
    ratios=(
        0.60,
        0.20,
        0.20
    )
):
    """
    Split train/good into:

        60% training
        20% validation
        20% held-out good images

    Uses a fixed random seed so the split is reproducible.
    """

    good_path = os.path.join(
        tile_root,
        "train",
        "good"
    )


    good_images = sorted([

        filename

        for filename
        in os.listdir(
            good_path
        )

        if filename
        .lower()
        .endswith(
            ".png"
        )
    ])


    rng = random.Random(
        seed
    )

    rng.shuffle(
        good_images
    )


    n = len(
        good_images
    )


    train_end = int(
        ratios[0]
        *
        n
    )


    val_end = (

        train_end

        +

        int(
            ratios[1]
            *
            n
        )
    )


    train_images = (
        good_images[
            :train_end
        ]
    )


    val_images = (
        good_images[
            train_end:
            val_end
        ]
    )


    test_good_images = (
        good_images[
            val_end:
        ]
    )


    return {

        "good_dir":
            good_path,

        "train_images":
            train_images,

        "val_images":
            val_images,

        "test_good_images":
            test_good_images,
    }


# ============================================================
# COMPLETE MVTec TEST SET
# ============================================================

def get_full_test_set(
    tile_root: str,
    defect_types=None
):
    """
    Build the complete labeled MVTec test set.

    Includes:

        test/good

    and every defect folder under:

        test/<defect_type>

    Labels:

        good      -> 0
        defective -> 1

    Each defect image is paired with its ground-truth mask
    when one exists.
    """

    test_dir = os.path.join(
        tile_root,
        "test"
    )


    gt_dir = os.path.join(
        tile_root,
        "ground_truth"
    )


    # --------------------------------------------------------
    # Automatically discover defect classes
    # --------------------------------------------------------

    if defect_types is None:

        defect_types = sorted(

            directory

            for directory
            in os.listdir(
                test_dir
            )

            if (
                os.path.isdir(
                    os.path.join(
                        test_dir,
                        directory
                    )
                )

                and

                directory != "good"
            )
        )


    # --------------------------------------------------------
    # Good test images
    # --------------------------------------------------------

    good_dir = os.path.join(
        test_dir,
        "good"
    )


    good_paths = sorted(

        os.path.join(
            good_dir,
            filename
        )

        for filename
        in os.listdir(
            good_dir
        )

        if filename
        .lower()
        .endswith(
            ".png"
        )
    )


    per_class = {

        "good": (
            good_paths,
            [None] * len(
                good_paths
            )
        )
    }


    # --------------------------------------------------------
    # Defective test images
    # --------------------------------------------------------

    for defect in defect_types:

        defect_dir = os.path.join(
            test_dir,
            defect
        )


        defect_gt_dir = os.path.join(
            gt_dir,
            defect
        )


        image_files = sorted(

            filename

            for filename
            in os.listdir(
                defect_dir
            )

            if filename
            .lower()
            .endswith(
                ".png"
            )
        )


        image_paths = [

            os.path.join(
                defect_dir,
                filename
            )

            for filename
            in image_files
        ]


        mask_paths = []


        for filename in image_files:

            stem = os.path.splitext(
                filename
            )[0]


            mask_path = os.path.join(

                defect_gt_dir,

                f"{stem}_mask.png"
            )


            if os.path.exists(
                mask_path
            ):

                mask_paths.append(
                    mask_path
                )

            else:

                mask_paths.append(
                    None
                )


        per_class[
            defect
        ] = (
            image_paths,
            mask_paths
        )


    return per_class