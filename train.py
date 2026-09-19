"""
Training script for the tile/glass defect autoencoder.

Fixes vs. the original notebook:
  - Training loop actually completes (val loss is accumulated, divided,
    and the best checkpoint is saved -- the original loop was cut off
    before this happened).
  - Adds a validation-loss LR scheduler (ReduceLROnPlateau) and early
    stopping, since the decoder is small and the training set is tiny
    (~138 images) -- left unchecked, 50 fixed epochs risks overfitting.
  - Adds a combined MSE + SSIM loss. Pure per-pixel MSE tends to reward
    blurry, averaged-out reconstructions; SSIM rewards preserving local
    structure/edges, which is exactly what makes cracks and scratches
    show up as reconstruction error at inference time.
  - Optional light augmentation (flips/rotation/jitter) on the "good"
    training images, since the real training set is small.

Usage:
    python train.py --data-root /path/to/tile --out best_autoencoder.pth
"""
import argparse
import copy
import json
import os
import time

import torch

# Works around a known PyTorch CPU-backend (oneDNN) crash on some Windows/Intel
# setups: "RuntimeError: could not execute a primitive" during backward() on
# Conv/BatchNorm layers, usually intermittent rather than immediate. Disabling
# mkldnn costs a bit of CPU speed but avoids the crash entirely.
torch.backends.mkldnn.enabled = False

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TileDataset, get_good_split
from model import AutoencoderAD


def gaussian_window(window_size: int, sigma: float, channels: int, device):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size: int = 11, sigma: float = 1.5):
    """Standard single-scale SSIM, averaged over the batch. Returns a
    value in [0, 1] where 1 = identical images."""
    channels = img1.size(1)
    window = gaussian_window(window_size, sigma, channels, img1.device)
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def combined_loss(output, target, alpha: float = 0.7):
    """alpha * MSE + (1 - alpha) * (1 - SSIM)."""
    mse = F.mse_loss(output, target)
    ssim_val = ssim(output, target)
    return alpha * mse + (1 - alpha) * (1 - ssim_val), mse.item(), ssim_val.item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    splits = get_good_split(args.data_root, seed=args.seed)
    train_ds = TileDataset(splits["train_images"], splits["good_dir"], augment=args.augment)
    val_ds = TileDataset(splits["val_images"], splits["good_dir"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = AutoencoderAD(freeze_encoder=not args.finetune_encoder).to(device)

    if args.finetune_encoder:
        # Small LR for the pretrained encoder, normal LR for the decoder.
        optimizer = optim.Adam([
            {"params": model.encoder.parameters(), "lr": args.lr * 0.1},
            {"params": model.decoder.parameters(), "lr": args.lr},
        ])
    else:
        optimizer = optim.Adam(model.decoder.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": [], "lr": []}
    start_epoch = 0

    if args.resume and os.path.exists(args.resume_path):
        print(f"Resuming from {args.resume_path}")
        ckpt = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        best_val_loss = ckpt["best_val_loss"]
        best_state = ckpt["best_state"]
        epochs_without_improvement = ckpt["epochs_without_improvement"]
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming at epoch {start_epoch}, best_val_loss so far={best_val_loss:.5f}")

    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()

        train_loss_sum, train_mse_sum = 0.0, 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, mse_val, _ = combined_loss(outputs, targets, alpha=args.mse_weight)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * inputs.size(0)
            train_mse_sum += mse_val * inputs.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_mse = train_mse_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum, val_mse_sum = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, mse_val, _ = combined_loss(outputs, targets, alpha=args.mse_weight)
                val_loss_sum += loss.item() * inputs.size(0)
                val_mse_sum += mse_val * inputs.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_mse = val_mse_sum / len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["lr"].append(current_lr)

        print(f"Epoch {epoch+1:03d}/{args.epochs} | "
              f"train_loss={train_loss:.5f} (mse={train_mse:.5f}) | "
              f"val_loss={val_loss:.5f} (mse={val_mse:.5f}) | lr={current_lr:.2e}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save a full resumable checkpoint after every epoch, so a crash
        # never costs more than one epoch of progress.
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }, args.resume_path)

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs).")
            break

    elapsed = time.time() - start
    print(f"Training finished in {elapsed:.1f}s. Best val_loss={best_val_loss:.5f}")

    torch.save(best_state, args.out)
    print(f"Saved best checkpoint to {args.out}")

    with open(args.history_out, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {args.history_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Path to the MVTec 'tile' folder")
    parser.add_argument("--out", default="optimized_autoencoder.pth")
    parser.add_argument("--history-out", default="training_history.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mse-weight", type=float, default=0.7, help="alpha in alpha*MSE + (1-alpha)*(1-SSIM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", help="Enable flip/rotation/jitter augmentation")
    parser.add_argument("--finetune-encoder", action="store_true", help="Unfreeze encoder with a low LR")
    parser.add_argument("--resume", action="store_true", help="Resume from --resume-path if it exists")
    parser.add_argument("--resume-path", default="training_checkpoint.pt",
                         help="Where per-epoch resumable checkpoints are saved/loaded")
    args = parser.parse_args()
    train(args)