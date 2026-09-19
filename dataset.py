"""
Dataset utilities for the MVTec AD "tile" (or "bottle"/other) category.
 
Expected directory layout (standard MVTec AD):
    <root>/
        train/good/*.png
        test/good/*.png
        test/<defect_type>/*.png          (e.g. crack, glue_strip, gray_stroke, oil, rough)
        ground_truth/<defect_type>/*_mask.png
"""
import os
import random
 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
 
IMG_SIZE = 128
 
# Encoder input: resized + ImageNet-normalized (matches the pretrained ResNet18 stem)
encoder_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
# Reconstruction target: resized, [0, 1] range (matches the decoder's Sigmoid output)
target_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
 
# Mask transform: nearest-neighbour resize to preserve binary mask values
mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])
 
# Light augmentation applied to the *source* PIL image before either transform,
# so encoder input and reconstruction target stay pixel-aligned.
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])
 
 
class TileDataset(Dataset):
    """Returns (encoder_input, reconstruction_target) for a list of image files."""
 
    def __init__(self, image_names, image_dir, augment: bool = False):
        self.image_names = image_names
        self.image_dir = image_dir
        self.augment = augment
 
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
 
        if self.augment:
            image = augment_transform(image)
 
        encoder_image = encoder_transform(image)
        target_image = target_transform(image)
        return encoder_image, target_image
 
 
class TileDefectDataset(Dataset):
    """Returns (encoder_input, target, mask, label) for evaluation, where
    label = 0 for good, 1 for defective. mask is a zeros tensor for good
    images (no ground truth defect region)."""
 
    def __init__(self, image_paths, mask_paths=None, label: int = 0):
        self.image_paths = image_paths
        self.mask_paths = mask_paths  # None, or same length as image_paths (may contain None entries)
        self.label = label
 
    def __len__(self):
        return len(self.image_paths)
 
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        encoder_image = encoder_transform(image)
        target_image = target_transform(image)
 
        if self.mask_paths is not None and self.mask_paths[index] is not None:
            mask = Image.open(self.mask_paths[index]).convert("L")
            mask = mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros((1, IMG_SIZE, IMG_SIZE))
 
        return encoder_image, target_image, mask, self.label
 
 
def get_good_split(tile_root: str, seed: int = 42, ratios=(0.60, 0.20, 0.20)):
    """Reproduces the notebook's 60/20/20 split of train/good/*.png,
    with the same random.seed(42) shuffle so results are comparable."""
    good_path = os.path.join(tile_root, "train", "good")
    good_images = sorted([f for f in os.listdir(good_path) if f.lower().endswith(".png")])
 
    rng = random.Random(seed)
    rng.shuffle(good_images)
 
    n = len(good_images)
    train_end = int(ratios[0] * n)
    val_end = train_end + int(ratios[1] * n)
 
    train_images = good_images[:train_end]
    val_images = good_images[train_end:val_end]
    test_good_images = good_images[val_end:]
 
    return {
        "good_dir": good_path,
        "train_images": train_images,
        "val_images": val_images,
        "test_good_images": test_good_images,
    }
 
 
def get_full_test_set(tile_root: str, defect_types=None):
    """Builds the full labeled test set: test/good (label 0) plus every
    defect subfolder under test/ (label 1), each paired with its
    ground_truth mask when available."""
    test_dir = os.path.join(tile_root, "test")
    gt_dir = os.path.join(tile_root, "ground_truth")
 
    if defect_types is None:
        defect_types = sorted(
            d for d in os.listdir(test_dir)
            if os.path.isdir(os.path.join(test_dir, d)) and d != "good"
        )
 
    good_dir = os.path.join(test_dir, "good")
    good_paths = sorted(
        os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.lower().endswith(".png")
    )
 
    per_class = {"good": (good_paths, [None] * len(good_paths))}
 
    for defect in defect_types:
        defect_dir = os.path.join(test_dir, defect)
        defect_gt_dir = os.path.join(gt_dir, defect)
        image_files = sorted(f for f in os.listdir(defect_dir) if f.lower().endswith(".png"))
 
        image_paths = [os.path.join(defect_dir, f) for f in image_files]
        mask_paths = []
        for f in image_files:
            stem = os.path.splitext(f)[0]
            mask_path = os.path.join(defect_gt_dir, f"{stem}_mask.png")
            mask_paths.append(mask_path if os.path.exists(mask_path) else None)
 
        per_class[defect] = (image_paths, mask_paths)
 
    return per_class
