# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         analyze_scale_effect.py  (NEW, Revision 2)
#
# Purpose: Experimental demonstration of the size scaling of the GSD
#          gradient caused by global average pooling. Hypothesis:
#
#              |nabla_x psi|_ij ~ 1/(H*W)
#              => the denoising gain collapses on images much larger than
#                 the training patch size.
#
#          We measure the PSNR gain (denoised vs. noisy) for the same model
#          in three configurations:
#            (1) patch mode:      40x40 crops (training condition),
#            (2) full-image mode: crop_size x crop_size, without rescaling,
#            (3) full-image mode: with gradient rescaling (variant A).
#
#          If the gain drops sharply from (1) to (2) and is (approximately)
#          restored by (3), the scaling diagnosis is confirmed. The result
#          quantitatively supports the discussion in Section 5.2 of the
#          thesis.
# ============================================================================

import os
import sys
import math
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)

from models.architectures import PotentialNetwork, GradientStepDenoiser


def parse_args() -> argparse.Namespace:
    """Defines and parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstration of the GAP size-scaling effect in the GSD."
    )
    parser.add_argument(
        "--test-dir", type=str,
        default=os.path.join(BASE_DIR, "data", "archive", "images", "test"),
    )
    parser.add_argument(
        "--model-path", type=str,
        default=os.path.join(BASE_DIR, "saved_models", "GSD_sigma15.pth"),
    )
    parser.add_argument("--out-dir", type=str, default=SCRIPT_DIR)
    parser.add_argument("--sigma", type=float, default=15.0,
                        help="Test noise level in pixel space [0,255]; should "
                             "match the training level of the model.")
    parser.add_argument("--patch-size", type=int, default=40,
                        help="Training patch size (reference scale).")
    parser.add_argument("--crop-size", type=int, default=256,
                        help="Full-image edge length for the comparison.")
    parser.add_argument("--patches-per-image", type=int, default=4,
                        help="Random 40x40 patches per image (mode 1).")
    parser.add_argument("--max-images", type=int, default=50,
                        help="Number of test images (0 = all).")
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_device() -> torch.device:
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """PSNR (dB) between two [0,1] image tensors."""
    mse = torch.nn.functional.mse_loss(img1, img2).item()
    return 100.0 if mse <= 0 else 10.0 * math.log10(1.0 / mse)


def denoise_gain(
    denoiser: GradientStepDenoiser, x_clean: torch.Tensor, sigma01: float,
    generator: torch.Generator,
) -> float:
    """Measures the PSNR gain PSNR(D(y), x) - PSNR(y, x) for y = x + v.

    Args:
        denoiser: GSD in eval mode.
        x_clean: Clean image (1, 1, H, W).
        sigma01: Noise level on the [0,1] scale.
        generator: CPU generator for reproducible noise.

    Returns:
        PSNR gain in dB (positive = denoiser improves).
    """
    noise = torch.randn(
        x_clean.shape, generator=generator
    ).to(x_clean.device) * sigma01
    y = torch.clamp(x_clean + noise, 0.0, 1.0)
    with torch.no_grad():
        x_hat = torch.clamp(denoiser(y), 0.0, 1.0)
    return psnr(x_hat, x_clean) - psnr(y, x_clean)


def run_scale_analysis(args: argparse.Namespace) -> None:
    """Runs the threefold comparison (patch / full image / full image + rescale)."""
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = get_device()
    sigma01 = args.sigma / 255.0
    print(f"[Scale] Device: {device} | sigma_test={args.sigma} | "
          f"patch {args.patch_size}px vs. full image {args.crop_size}px")

    # Load the model (average-pooling checkpoint).
    net = PotentialNetwork(in_channels=1, features=args.features).to(device)
    net.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    denoiser = GradientStepDenoiser(
        net, rescale_gradient=False, train_patch_size=args.patch_size
    ).to(device)
    denoiser.eval()

    to_gray = transforms.Grayscale(num_output_channels=1)
    center_crop = transforms.CenterCrop(args.crop_size)
    rand_crop = transforms.RandomCrop(args.patch_size)
    to_tensor = transforms.ToTensor()

    valid_ext = (".jpg", ".jpeg", ".png")
    files = sorted(
        f for f in os.listdir(args.test_dir) if f.lower().endswith(valid_ext)
    )
    if args.max_images > 0:
        files = files[: args.max_images]
    if not files:
        print(f"[Scale] No images in {args.test_dir}. Aborting.")
        return

    gains_patch: List[float] = []
    gains_full: List[float] = []
    gains_full_rescaled: List[float] = []

    print(f"[Scale] Evaluating over {len(files)} images ...")
    for idx, fname in enumerate(files):
        try:
            img = to_gray(
                Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
            )
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[Scale] File skipped ({fname}): {exc}")
            continue

        # (1) Patch mode: training condition (40x40).
        denoiser.rescale_gradient = False
        for _ in range(args.patches_per_image):
            x_patch = to_tensor(rand_crop(img)).unsqueeze(0).to(device)
            gains_patch.append(denoise_gain(denoiser, x_patch, sigma01, gen))

        # Prepare the full image.
        x_full = to_tensor(center_crop(img)).unsqueeze(0).to(device)

        # (2) Full image without rescaling.
        denoiser.rescale_gradient = False
        gains_full.append(denoise_gain(denoiser, x_full, sigma01, gen))

        # (3) Full image with rescaling (variant A).
        denoiser.rescale_gradient = True
        gains_full_rescaled.append(denoise_gain(denoiser, x_full, sigma01, gen))

        if (idx + 1) % 10 == 0 or (idx + 1) == len(files):
            print(f"[Scale] {idx + 1}/{len(files)} images processed ...")

    denoiser.rescale_gradient = False  # reset state.

    # Results.
    labels = [
        f"Patch {args.patch_size}px\n(training condition)",
        f"Full image {args.crop_size}px\n(without rescaling)",
        f"Full image {args.crop_size}px\n(with rescaling)",
    ]
    data = [gains_patch, gains_full, gains_full_rescaled]

    print("\n[Scale] --- RESULTS (PSNR gain from denoising) ---")
    for lab, d in zip(labels, data):
        lab_flat = lab.replace("\n", " ")
        print(f"[Scale] {lab_flat:45s} | mean {np.mean(d):+6.2f} dB "
              f"(std {np.std(d):.2f}, N={len(d)})")

    ratio = np.mean(gains_full) / max(np.mean(gains_patch), 1e-9)
    print(f"[Scale] Ratio full image/patch (without rescaling): {ratio:.3f} "
          f"-- values << 1 confirm the 1/(H*W) scaling of the gradient.")

    # Plot.
    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=labels)
    plt.axhline(0.0, color="gray", linewidth=0.8)
    plt.ylabel("PSNR gain from denoising (dB)")
    plt.title(f"GAP size scaling of the GSD gradient "
              f"($\\sigma$={args.sigma:.0f}, N={len(files)} images)")
    plt.grid(True, axis="y", alpha=0.5)
    path = os.path.join(args.out_dir, "gsd_scale_effect.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Scale] Plot saved to: {path}")


if __name__ == "__main__":
    run_scale_analysis(parse_args())
