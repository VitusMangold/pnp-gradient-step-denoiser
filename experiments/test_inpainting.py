# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         test_inpainting.py
#
# Purpose: Experimental validation of the PnP-FBS framework on the
#          inpainting problem (image completion) with extreme information
#          loss:
#
#              y = M x + v,
#
#          where M is a diagonal projection operator (binary mask). Pixels
#          with M_ii = 0 lie in the null space and are irreversibly
#          deleted; their reconstruction relies entirely on the GSD prior.
#
# Mathematical note on the step size:
#   The mask M is idempotent (M^2 = M) and self-adjoint (M* = M), hence the
#   gradient is nabla f(x) = M(x - y) and the Lipschitz constant is L = 1.
#   The Ryu condition alpha < 2/L = 2 is satisfied with alpha = 1.
#
# Generated figures (for the thesis):
#   1. Visual comparison (original / masked+noisy / reconstruction).
#   2. Convergence curve of the relative residual (Hurault 2022, Fig. 1h).
#   3. PSNR distribution over the test set (box plot).
#
# References:
#   - Inpainting in the PnP context:  Ryu et al. (2019); Hurault et al.
#     (2022), J.3.
# ============================================================================

import os
import sys
import math
import argparse
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)

from models.architectures import PotentialNetwork, GradientStepDenoiser
from models.pnp_solvers import pnp_fbs


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Computes the PSNR (dB) between two [0,1] image tensors.

    Args:
        img1: First image (e.g., reconstruction).
        img2: Second image (e.g., ground truth). Same shape as img1.

    Returns:
        PSNR in decibels; +inf for identical images.
    """
    mse = torch.nn.functional.mse_loss(img1, img2)
    if mse.item() == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse.item())


def make_denoiser_wrapper(
    model: GradientStepDenoiser,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Binds the model early and ensures the clamping to [0,1].

    Args:
        model: The GSD model to bind.

    Returns:
        Wrapper that only clamps; the GSD manages autograd internally.
    """
    def wrapper(x_in: torch.Tensor) -> torch.Tensor:
        return model(torch.clamp(x_in, 0.0, 1.0))
    return wrapper


def make_grad_inpainting(
    mask: torch.Tensor, y_meas: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Creates the data-term gradient for inpainting.

    Since M is idempotent and self-adjoint, the gradient of
    f(x) = 1/2 ||M x - y||^2 simplifies to nabla f(x) = M(x - y).

    Args:
        mask: Binary projection mask M (1 = observed, 0 = deleted).
        y_meas: Masked, noisy measurement y.

    Returns:
        Callable grad_f(x).
    """
    def grad_f(x_in: torch.Tensor) -> torch.Tensor:
        return mask * (x_in - y_meas)
    return grad_f


def parse_args() -> argparse.Namespace:
    """Defines and parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PnP-FBS inpainting experiment (image completion)."
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
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--drop-prob", type=float, default=0.70,
                        help="Fraction of deleted pixels (0.70 = 70%% missing).")
    parser.add_argument("--noise-level", type=float, default=0.05,
                        help="Additive noise (on the [0,1] scale).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Step size (alpha<2/L=2, since L=1 for the mask).")
    parser.add_argument("--num-iter", type=int, default=400)
    parser.add_argument("--tol", type=float, default=1e-5,
                        help="Relative stopping criterion for the residual.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_device() -> torch.device:
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str, features: int,
               device: torch.device) -> GradientStepDenoiser:
    """Loads the trained GSD model.

    Args:
        model_path: Path to the .pth file.
        features: Filter count of the potential network.
        device: Target device.

    Returns:
        Loaded GSD in eval mode (with gradient rescaling, variant A).

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    net = PotentialNetwork(in_channels=1, features=features).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    denoiser = GradientStepDenoiser(net, rescale_gradient=True, train_patch_size=40).to(device)
    denoiser.eval()
    return denoiser


# ----------------------------------------------------------------------------
# Plot functions
# ----------------------------------------------------------------------------
def plot_visual_comparison(
    img_true: np.ndarray, img_masked: np.ndarray, img_recon: np.ndarray,
    psnr_masked: float, psnr_recon: float, drop_prob: float, out_dir: str,
) -> str:
    """Creates the three-panel visual comparison for the inpainting."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(img_true, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_masked, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Masked ({drop_prob*100:.0f}% missing)\n"
                      f"{psnr_masked:.2f} dB")
    axes[1].axis("off")

    axes[2].imshow(img_recon, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"PnP-FBS reconstruction\n{psnr_recon:.2f} dB")
    axes[2].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "inpainting_visual_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_convergence(history: List[float], out_dir: str) -> str:
    """Plots the residual curve as empirical convergence evidence."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(history) + 1), history,
                 marker=".", color="navy", linewidth=1.5)
    plt.title("PnP-FBS convergence (inpainting)")
    plt.xlabel("Iteration $k$")
    plt.ylabel(r"Rel. residual $\|x_{k+1}-x_k\| / \|x_k\|$")
    plt.grid(True, which="both", alpha=0.5)
    path = os.path.join(out_dir, "inpainting_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_psnr_distribution(psnr_masked: List[float], psnr_recon: List[float],
                           out_dir: str) -> str:
    """Plots the PSNR distribution (box plot) over the entire test set."""
    plt.figure(figsize=(7, 5))
    plt.boxplot([psnr_masked, psnr_recon],
                tick_labels=["Masked (input)", "Reconstructed"])
    plt.ylabel("PSNR (dB)")
    plt.title(f"PnP-FBS inpainting: PSNR distribution "
              f"(N={len(psnr_recon)} images)")
    plt.grid(True, axis="y", alpha=0.5)
    path = os.path.join(out_dir, "inpainting_psnr_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------
def run_inpainting_test(args: argparse.Namespace) -> None:
    """Runs the full inpainting experiment over the test set.

    Args:
        args: Parsed command-line arguments.
    """
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[Inpaint] Device: {device} | Seed: {args.seed}")
    print(f"[Inpaint] alpha={args.alpha} (L=1 for the idempotent mask, "
          f"Ryu condition alpha<2 satisfied)")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
    ])

    denoiser = load_model(args.model_path, args.features, device)

    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(args.test_dir) if f.lower().endswith(valid_ext)
    )
    if not image_files:
        print(f"[Inpaint] No images in {args.test_dir}. Aborting.")
        return

    psnr_masked_all: List[float] = []
    psnr_recon_all: List[float] = []

    # Visualization data from the first image.
    vis = {}
    convergence_history: List[float] = []

    print(f"[Inpaint] Evaluating over {len(image_files)} images ...")

    for idx, fname in enumerate(image_files):
        try:
            img = Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[Inpaint] File skipped ({fname}): {exc}")
            continue

        x_true = transform(img).unsqueeze(0).to(device)

        # Simulate the forward model y = M (x + v).
        mask = (torch.rand_like(x_true) > args.drop_prob).float()
        noise = torch.randn_like(x_true) * args.noise_level
        y_meas = mask * (x_true + noise)

        grad_f = make_grad_inpainting(mask, y_meas)
        wrapper = make_denoiser_wrapper(denoiser)

        # Record the convergence history only for the first image.
        want_history = (idx == 0)
        x_recon, history = pnp_fbs(
            y_meas=y_meas, denoiser=wrapper, grad_f=grad_f,
            alpha=args.alpha, num_iter=args.num_iter, tol=args.tol,
            return_history=want_history,
        )
        if want_history and history is not None:
            convergence_history = history

        y_meas_c = torch.clamp(y_meas, 0.0, 1.0)
        x_recon_c = torch.clamp(x_recon, 0.0, 1.0)

        psnr_masked_all.append(calculate_psnr(y_meas_c, x_true))
        psnr_recon_all.append(calculate_psnr(x_recon_c, x_true))

        if idx == 0:
            vis = {
                "true": x_true.squeeze().cpu().numpy(),
                "masked": y_meas_c.squeeze().cpu().numpy(),
                "recon": x_recon_c.squeeze().cpu().numpy(),
                "psnr_masked": psnr_masked_all[-1],
                "psnr_recon": psnr_recon_all[-1],
            }

        if (idx + 1) % 5 == 0 or (idx + 1) == len(image_files):
            print(f"[Inpaint] {idx + 1}/{len(image_files)} images processed ...")

    # Results.
    print("\n[Inpaint] --- RESULTS ---")
    print(f"[Inpaint] PSNR masked (mean):        "
          f"{np.mean(psnr_masked_all):.2f} dB")
    print(f"[Inpaint] PSNR reconstructed (mean): "
          f"{np.mean(psnr_recon_all):.2f} dB "
          f"(std {np.std(psnr_recon_all):.2f})")
    print(f"[Inpaint] Mean gain: "
          f"{np.mean(psnr_recon_all) - np.mean(psnr_masked_all):.2f} dB")

    # Plots.
    os.makedirs(args.out_dir, exist_ok=True)
    if vis:
        p1 = plot_visual_comparison(
            vis["true"], vis["masked"], vis["recon"],
            vis["psnr_masked"], vis["psnr_recon"], args.drop_prob, args.out_dir,
        )
        print(f"[Inpaint] Visual comparison: {p1}")

    if convergence_history:
        p2 = plot_convergence(convergence_history, args.out_dir)
        print(f"[Inpaint] Convergence curve: {p2}")

    p3 = plot_psnr_distribution(psnr_masked_all, psnr_recon_all, args.out_dir)
    print(f"[Inpaint] PSNR distribution: {p3}")


if __name__ == "__main__":
    run_inpainting_test(parse_args())
