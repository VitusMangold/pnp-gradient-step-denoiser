# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         train_gsd.py
#
# Purpose: Training script for the Gradient Step Denoiser (GSD). Optimizes
#          the weights theta of the potential network psi_theta such that
#          the resulting denoiser D_theta(x) = x - nabla_x psi_theta(x)
#          solves the AWGN denoising problem.
#
# Methodological particularity (double backpropagation):
#   The forward pass of the GSD already contains a derivative
#   (nabla_x psi). The weight update via loss.backward() therefore requires
#   differentiating THROUGH this derivative (second derivative). This is
#   enabled by the flag create_graph=True in the GSD forward pass (active
#   in train() mode).
#
# References:
#   - GSD / double backpropagation: Hurault et al. (2022), ICLR.
#   - Residual-learning target:     Zhang et al. (2017), IEEE TIP (DnCNN).
#   - Training setup (BSD500, 40x40 patches, Adam, lr=1e-4):
#       Ryu et al. (2019), ICML, Sec. 4.3.
# ============================================================================

import os
import sys
import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add the project root to the path so that package imports work.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)

from data.dataset import BSD500DenoisingDataset
from models.architectures import PotentialNetwork, GradientStepDenoiser


def parse_args() -> argparse.Namespace:
    """Defines and parses the command-line arguments.

    Returns:
        Namespace with all training hyperparameters and paths.
    """
    parser = argparse.ArgumentParser(
        description="Training of the Gradient Step Denoiser (GSD)."
    )
    # Paths (defaults relative to the project root, no absolute paths).
    parser.add_argument(
        "--train-dir",
        type=str,
        default=os.path.join(BASE_DIR, "data", "archive", "images", "train"),
        help="Directory with training images.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=os.path.join(BASE_DIR, "data", "archive", "images", "val"),
        help="Directory with validation images.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(BASE_DIR, "saved_models"),
        help="Target directory for the best model.",
    )

    # Hyperparameters.
    parser.add_argument("--sigma", type=int, default=15,
                        help="Noise level (pixel space [0,255]) for training.")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of the Adam optimizer.")
    parser.add_argument("--patch-size", type=int, default=40)
    parser.add_argument("--features", type=int, default=64,
                        help="Number of convolutional filters in the potential network.")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max norm for gradient clipping (0 = disabled).")
    parser.add_argument("--scheduler-step", type=int, default=200,
                        help="StepLR: epoch interval for halving the LR.")

    # Infrastructure.
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 recommended on macOS/MPS).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Sets all relevant random seeds for reproducible runs.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Selects the best available compute device (CUDA > MPS > CPU).

    Returns:
        The selected ``torch.device``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mse_to_psnr(mse: float) -> float:
    """Converts an MSE value (on [0,1] images) to PSNR (dB).

    Args:
        mse: Mean squared error.

    Returns:
        PSNR in decibels; +inf for mse == 0.
    """
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def run_epoch(
    model: GradientStepDenoiser,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer = None,
    grad_clip: float = 0.0,
) -> float:
    """Runs one training or validation epoch.

    Implements the dynamic "target shift": the dataset returns
    (noisy image y, noise target v). Since the GSD outputs the DENOISED
    image, the ground truth is reconstructed at run time as x = y - v
    (residual learning, Zhang et al. 2017).

    Args:
        model: The GSD (wrapper around the potential network).
        loader: DataLoader for training or validation.
        criterion: Loss function (MSE).
        device: Compute device.
        optimizer: Adam optimizer. If None, the epoch runs in evaluation
            mode (no weight updates).
        grad_clip: Max norm for gradient clipping (>0 enables it). Only
            relevant in training mode. Necessary because double
            backpropagation can produce exploding gradients.

    Returns:
        Average loss over all batches of the epoch.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    # torch.set_grad_enabled controls the GLOBAL autograd state. The GSD
    # internally re-enables enable_grad() for the psi derivative; in
    # evaluation mode, however, create_graph remains False (no 2nd
    # derivative).
    with torch.set_grad_enabled(is_training):
        for noisy_imgs, target_noise in loader:
            noisy_imgs = noisy_imgs.to(device, non_blocking=True)
            target_noise = target_noise.to(device, non_blocking=True)

            # Target shift: ground truth = noisy image - noise.
            clean_imgs = noisy_imgs - target_noise

            denoised_imgs = model(noisy_imgs)
            loss = criterion(denoised_imgs, clean_imgs)

            if is_training:
                optimizer.zero_grad()
                loss.backward()  # 2nd derivative (double backprop) via create_graph=True
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.potential_net.parameters(), max_norm=grad_clip
                    )
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(1, len(loader))


def train_gsd(args: argparse.Namespace) -> None:
    """Main training routine for the GSD.

    Trains the potential network over ``args.epochs`` epochs, validates
    periodically, and saves only the model with the lowest validation loss
    (best checkpointing to avoid overfitting).

    Args:
        args: Parsed command-line arguments.
    """
    set_seed(args.seed)
    device = get_device()
    print(f"[Train] Device: {device} | Seed: {args.seed}")

    # --- 1. DATASETS ---
    print("[Train] Loading training and validation data ...")
    train_dataset = BSD500DenoisingDataset(
        image_dir=args.train_dir, patch_size=args.patch_size, sigma=args.sigma
    )
    val_dataset = BSD500DenoisingDataset(
        image_dir=args.val_dir, patch_size=args.patch_size, sigma=args.sigma
    )

    # pin_memory is only useful with CUDA (speeds up host->device transfer).
    use_pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=use_pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_pin,
    )

    # --- 2. MODEL, LOSS, OPTIMIZER, SCHEDULER ---
    potential_net = PotentialNetwork(
        in_channels=1, features=args.features
    ).to(device)
    model = GradientStepDenoiser(potential_net).to(device)

    criterion = nn.MSELoss()
    # The wrapper has no parameters of its own -> the optimizer explicitly
    # receives the weights of the embedded potential network.
    optimizer = optim.Adam(model.potential_net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=0.5
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, f"GSD_sigma{args.sigma}.pth")
    best_val_loss = float("inf")

    # --- 3. TRAINING/VALIDATION LOOP ---
    print(f"[Train] Starting double-backpropagation training for "
          f"{args.epochs} epochs ...")

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, grad_clip=args.grad_clip,
        )
        val_loss = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Best checkpointing: only the best model is persisted.
        saved_flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.potential_net.state_dict(), best_model_path)
            saved_flag = "  -> [BEST MODEL SAVED]"

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:3d}/{args.epochs}] | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_loss:.6f} (PSNR {mse_to_psnr(val_loss):.2f} dB) | "
            f"LR: {current_lr:.2e}{saved_flag}"
        )

    print(f"\n[Train] Finished. Best model (val MSE={best_val_loss:.6f}) "
          f"saved to: {best_model_path}")


if __name__ == "__main__":
    train_gsd(parse_args())
