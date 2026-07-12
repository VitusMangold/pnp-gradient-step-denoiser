# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         test_gsd_benchmark.py  (FINAL VERSION)
#
# Purpose: Stress test of the PnP framework under NON-Gaussian,
#          signal-dependent Poisson noise (relevant for medical imaging and
#          low-light photography). Compares PnP-ADMM and PnP-FBS.
#
#          Observation model (photon counting):
#              y_i ~ Poisson(peak * x_i).
#
#          Data term (negative Poisson log-likelihood / generalized
#          Kullback-Leibler divergence):
#              f(x) = sum_i ( x_i - y_i log x_i ).
#          Its proximal operator has a closed-form solution (positive root
#          of a quadratic equation; Ryu et al. 2019, Sec. 5), which ADMM
#          uses here. For FBS, the gradient nabla f(x) = 1 - y/x is used
#          (numerically stabilized via an epsilon).
#
# FINAL CONFIGURATION (Revision 3):
#   * Both methods receive the same iteration budget (20 iterations) so
#     that the comparison is fair.
#   * Fixed, documented parameters (admm-alpha=1.0, fbs-alpha=0.1, each
#     scaled by peak); NO data-dependent tuning. Single-image tuning
#     generalized poorly to the test set in preliminary runs and was
#     therefore discarded for the final evaluation.
#   * Gradient rescaling (variant A) active (see architectures.py).
#
# IMPORTANT METHODOLOGICAL NOTES:
#   (a) Domain mismatch: the denoiser was trained on AWGN (sigma=15) but
#       here regularizes a completely different (Poisson) noise model.
#       That this works underlines the flexibility of PnP compared to
#       end-to-end trained networks.
#   (b) Scaling: the photon counts y are immediately normalized to [0,1]
#       by peak so that the denoiser operates in its training domain.
#       Consequently, the step sizes are co-scaled by 1/peak so that the
#       convergence conditions (Ryu 2019, Thm. 1/2) hold in the normalized
#       space.
#   (c) Reporting: both methods are evaluated after the fixed budget of 20
#       iterations (no "best iterate"); the state closer to the fixed
#       point is the relevant object for a convergent regularization.
#
# Generated figures (for the thesis):
#   1. Convergence comparison ADMM vs. FBS (residual, log scale).
#   2. Visual comparison (original / Poisson measurement / reconstruction).
#   3. PSNR and SSIM distribution over the test set (box plots).
#
# References:
#   - Poisson data term, prox, PnP-ADMM/FBS:  Ryu et al. (2019), Sec. 5.
#   - GSD prior:                              Hurault et al. (2022).
# ============================================================================

import os
import sys
import argparse
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)

from models.architectures import PotentialNetwork, GradientStepDenoiser
from models.pnp_solvers import pnp_admm, pnp_fbs


# ----------------------------------------------------------------------------
# Metrics and operator factories
# ----------------------------------------------------------------------------
def calculate_metrics(img1: torch.Tensor, img2: torch.Tensor) -> Tuple[float, float]:
    """Computes PSNR (dB) and SSIM between two [0,1] images.

    Args:
        img1: Reconstruction (B, 1, H, W).
        img2: Ground truth (B, 1, H, W).

    Returns:
        Tuple (psnr, ssim). PSNR=100 for a perfect match.
    """
    mse = torch.nn.functional.mse_loss(img1, img2).item()
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else 100.0

    im1 = img1.squeeze().cpu().numpy()
    im2 = img2.squeeze().cpu().numpy()
    ssim = ssim_func(im1, im2, data_range=1.0)
    return psnr, ssim


def make_prox_poisson(
    y_meas: torch.Tensor, alpha: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Creates the proximal operator of the Poisson data term (for ADMM).

    Solves argmin_x { x - y log x + 1/(2 alpha) ||x - v||^2 } in closed
    form via the positive root of x^2 + (alpha - v) x - alpha y = 0
    (Ryu et al., 2019, Sec. 5). The argument x_in corresponds to the dual
    variable v.

    Args:
        y_meas: Normalized Poisson measurement.
        alpha: Penalty/step-size parameter in the normalized space.

    Returns:
        Callable prox(v).
    """
    def prox(v: torch.Tensor) -> torch.Tensor:
        term = v - alpha
        return 0.5 * (term + torch.sqrt(term**2 + 4 * alpha * y_meas))
    return prox


def make_grad_poisson(
    y_meas: torch.Tensor, eps: float = 1e-6
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Creates the gradient of the Poisson log-likelihood (for FBS).

    nabla f(x) = 1 - y / x. The epsilon term stabilizes the division for
    x -> 0 (the gradient is formally singular there; FBS therefore tends
    to be less stable than ADMM under Poisson noise, cf. Ryu et al. 2019,
    Fig. 2).

    Args:
        y_meas: Normalized Poisson measurement.
        eps: Stabilization constant.

    Returns:
        Callable grad(x).
    """
    def grad(x_in: torch.Tensor) -> torch.Tensor:
        return 1.0 - y_meas / (x_in + eps)
    return grad


# ----------------------------------------------------------------------------
# Infrastructure
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Defines and parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GSD benchmark under Poisson noise (PnP-ADMM vs. FBS)."
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
    parser.add_argument("--method", type=str, choices=["ADMM", "FBS", "BOTH"],
                        default="BOTH", help="PnP algorithm to test.")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--peak", type=float, default=60.0,
                        help="Max. photon count per pixel (smaller = noisier).")
    # FINAL: identical budget for both methods (fair comparison).
    parser.add_argument("--admm-iter", type=int, default=20)
    parser.add_argument("--fbs-iter", type=int, default=20)
    parser.add_argument("--admm-alpha", type=float, default=1.0,
                        help="ADMM penalty (internally scaled by peak).")
    parser.add_argument("--fbs-alpha", type=float, default=0.1,
                        help="FBS step size (internally scaled by peak).")
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
    denoiser = GradientStepDenoiser(
        net, rescale_gradient=True, train_patch_size=40
    ).to(device)
    denoiser.eval()
    return denoiser


# ----------------------------------------------------------------------------
# Plot functions
# ----------------------------------------------------------------------------
def plot_convergence_comparison(histories: Dict[str, List[float]],
                                out_dir: str) -> str:
    """Plots the residual curves of all tested methods in comparison."""
    plt.figure(figsize=(8, 5))
    colors = {"ADMM": "navy", "FBS": "darkorange"}
    for name, hist in histories.items():
        if hist:
            plt.semilogy(range(1, len(hist) + 1), hist, marker=".",
                         linewidth=1.5, label=f"PnP-{name}",
                         color=colors.get(name, None))
    plt.title("PnP convergence under Poisson noise: ADMM vs. FBS")
    plt.xlabel("Iteration $k$")
    plt.ylabel(r"Rel. residual $\|x_{k+1}-x_k\| / \|x_k\|$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.5)
    path = os.path.join(out_dir, "poisson_convergence_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_visual_comparison(img_true: np.ndarray, img_noisy: np.ndarray,
                           recon: Dict[str, np.ndarray],
                           recon_metrics: Dict[str, Tuple[float, float]],
                           out_dir: str) -> str:
    """Visual comparison original / Poisson measurement / reconstruction(s)."""
    n_panels = 2 + len(recon)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(img_true, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_noisy, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Poisson measurement")
    axes[1].axis("off")

    for ax, name in zip(axes[2:], recon.keys()):
        psnr, ssim = recon_metrics[name]
        ax.imshow(recon[name], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"PnP-{name}\n{psnr:.2f} dB | SSIM {ssim:.3f}")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "poisson_visual_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_metric_distribution(metrics: Dict[str, Dict[str, List[float]]],
                             out_dir: str) -> str:
    """Box plots of the PSNR and SSIM distribution per method."""
    methods = list(metrics.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.boxplot([metrics[m]["psnr"] for m in methods], tick_labels=methods)
    ax1.set_title("PSNR distribution")
    ax1.set_ylabel("PSNR (dB)")
    ax1.grid(True, axis="y", alpha=0.5)

    ax2.boxplot([metrics[m]["ssim"] for m in methods], tick_labels=methods)
    ax2.set_title("SSIM distribution")
    ax2.set_ylabel("SSIM")
    ax2.grid(True, axis="y", alpha=0.5)

    plt.suptitle(f"GSD benchmark under Poisson noise "
                 f"(N={len(metrics[methods[0]]['psnr'])} images)")
    plt.tight_layout()
    path = os.path.join(out_dir, "poisson_metric_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------
def run_gsd_benchmark(args: argparse.Namespace) -> None:
    """Runs the Poisson benchmark over the test set.

    Args:
        args: Parsed command-line arguments.
    """
    torch.manual_seed(args.seed)
    device = get_device()
    methods = ["ADMM", "FBS"] if args.method == "BOTH" else [args.method]
    print(f"[Poisson] Device: {device} | Methods: {methods} | "
          f"peak={args.peak} | Seed: {args.seed}")
    print(f"[Poisson] Budget: ADMM {args.admm_iter} iter. / "
          f"FBS {args.fbs_iter} iter. | admm-alpha={args.admm_alpha}, "
          f"fbs-alpha={args.fbs_alpha} (each scaled by /peak)")
    print("[Poisson] Note: denoiser trained on AWGN (sigma=15), here "
          "regularizes Poisson noise (PnP flexibility).")

    # Scale the step sizes into the normalized [0,1] space.
    alpha_admm = args.admm_alpha / args.peak
    alpha_fbs = args.fbs_alpha / args.peak

    denoiser = load_model(args.model_path, args.features, device)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
    ])

    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(args.test_dir) if f.lower().endswith(valid_ext)
    )
    if not image_files:
        print(f"[Poisson] No images in {args.test_dir}. Aborting.")
        return

    # Metric collectors per method.
    metrics = {m: {"psnr": [], "ssim": []} for m in methods}
    # Visualization and convergence data from the first image.
    vis_recon: Dict[str, np.ndarray] = {}
    vis_metrics: Dict[str, Tuple[float, float]] = {}
    histories: Dict[str, List[float]] = {}
    vis_true = vis_noisy = None

    print(f"[Poisson] Evaluating over {len(image_files)} images ...")

    for idx, fname in enumerate(image_files):
        try:
            img = Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[Poisson] File skipped ({fname}): {exc}")
            continue

        x_true = transform(img).unsqueeze(0).to(device)

        # Forward model: Poisson measurement, immediately normalized to
        # [0,1]. torch.poisson runs on the CPU (MPS compatibility).
        y_raw = torch.poisson((x_true * args.peak).cpu()).to(device)
        y_norm = y_raw / args.peak

        for name in methods:
            want_history = (idx == 0)
            with torch.no_grad():
                if name == "ADMM":
                    prox = make_prox_poisson(y_norm, alpha_admm)
                    x_recon, hist = pnp_admm(
                        y_norm, denoiser, prox,
                        num_iter=args.admm_iter, return_history=want_history,
                    )
                else:  # FBS
                    grad = make_grad_poisson(y_norm)
                    x_recon, hist = pnp_fbs(
                        y_norm, denoiser, grad, alpha=alpha_fbs,
                        num_iter=args.fbs_iter, return_history=want_history,
                    )

            x_recon_c = torch.clamp(x_recon, 0.0, 1.0)
            psnr, ssim = calculate_metrics(x_recon_c, x_true)
            metrics[name]["psnr"].append(psnr)
            metrics[name]["ssim"].append(ssim)

            if idx == 0:
                vis_recon[name] = x_recon_c.squeeze().cpu().numpy()
                vis_metrics[name] = (psnr, ssim)
                if want_history and hist is not None:
                    histories[name] = hist

        if idx == 0:
            vis_true = x_true.squeeze().cpu().numpy()
            vis_noisy = torch.clamp(y_norm, 0.0, 1.0).squeeze().cpu().numpy()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"[Poisson] {idx + 1}/{len(image_files)} images processed ...")

    # Results.
    print("\n[Poisson] --- RESULTS ---")
    for name in methods:
        p = metrics[name]["psnr"]
        s = metrics[name]["ssim"]
        print(f"[Poisson] PnP-{name:4s} | "
              f"PSNR {np.mean(p):.2f} dB (std {np.std(p):.2f}) | "
              f"SSIM {np.mean(s):.4f} (std {np.std(s):.4f})")

    # Plots.
    os.makedirs(args.out_dir, exist_ok=True)
    if histories:
        p1 = plot_convergence_comparison(histories, args.out_dir)
        print(f"[Poisson] Convergence comparison: {p1}")

    if vis_recon and vis_true is not None:
        p2 = plot_visual_comparison(
            vis_true, vis_noisy, vis_recon, vis_metrics, args.out_dir
        )
        print(f"[Poisson] Visual comparison: {p2}")

    p3 = plot_metric_distribution(metrics, args.out_dir)
    print(f"[Poisson] Metric distribution: {p3}")


if __name__ == "__main__":
    run_gsd_benchmark(parse_args())
