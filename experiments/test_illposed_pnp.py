# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         test_illposed_pnp.py  (FINAL VERSION)
#
# Purpose: Experimental validation of the PnP-FBS framework on the linear,
#          ill-posed inverse problem of image deconvolution (deblurring):
#
#              y = A x + v,
#
#          where A is a Gaussian convolution operator (blur) and v additive
#          noise. Since A damps high-frequency information, direct
#          inversion is unstable; the GSD prior regularizes the problem.
#
# FINAL CONFIGURATION (Revision 3):
#   * Denoiser grid sigma in {2, 5, 10, 15, 20, 25, 30, 35}: uses all
#     available trained models and brackets the optimum on both sides.
#   * Measurement noise level 25/255: the operating point is deliberately
#     chosen such that the regularization trade-off (bias-variance) becomes
#     visible: too small a sigma under-regularizes (noise amplification),
#     too large a sigma over-regularizes (loss of detail); the PSNR optimum
#     lies in the interior of the grid. The noise level must be stated in
#     the figure caption of the thesis.
#   * Gradient rescaling (variant A) active: compensates the 1/(H*W)
#     scaling of the GAP gradient (see architectures.py, Revision 2, and
#     analyze_scale_effect.py).
#
# Generated figures (for the thesis):
#   1. PSNR over denoiser level sigma  -> regularization trade-off.
#   2. Visual comparison (original / measurement / reconstructions for
#      under-, near-optimally, and over-regularized).
#   3. Convergence curve of the relative residual -> empirical evidence of
#      convergence of the PnP-FBS iteration (cf. Hurault 2022, Fig. 1h).
#
# References:
#   - PnP-FBS, step size alpha < 2/L:  Ryu et al. (2019), Theorem 1.
#   - GSD prior:                       Hurault et al. (2022).
# ============================================================================

import os
import sys
import math
import argparse
from typing import Callable, Dict, List, Tuple

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
# Helper functions: metrics and operators
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


def gaussian_kernel(size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    """Creates a normalized 2D Gaussian kernel (point spread function).

    Args:
        size: Edge length of the square kernel in pixels.
        sigma: Standard deviation of the Gaussian.

    Returns:
        Kernel tensor of shape (1, 1, size, size), normalized to sum 1.
    """
    coords = torch.arange(size) - size // 2
    x, y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def blur_operator(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Applies the forward operator A (convolution) to an image.

    Args:
        x: Image tensor (B, 1, H, W).
        kernel: Convolution kernel (1, 1, k, k).

    Returns:
        Convolved (blurred) image of the same spatial size.
    """
    padding = kernel.size(2) // 2
    return torch.nn.functional.conv2d(x, kernel, padding=padding)


def blur_adjoint(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Applies the adjoint operator A*.

    For a symmetric Gaussian kernel, A = A* (self-adjoint), so the same
    convolution can be used.

    Args:
        x: Image tensor (B, 1, H, W).
        kernel: Convolution kernel (1, 1, k, k).

    Returns:
        Result of A* x.
    """
    padding = kernel.size(2) // 2
    return torch.nn.functional.conv2d(x, kernel, padding=padding)


def compute_spectral_norm_conv(
    kernel: torch.Tensor, img_shape: Tuple[int, int] = (256, 256)
) -> float:
    """Computes the Lipschitz constant L = ||A^T A||_2 of the convolution.

    For a circular convolution, A is diagonal in Fourier space; the largest
    eigenvalues of A^T A are given by max |H(omega)|^2, where H is the DFT
    of the (zero-padded) kernel. The computation runs on the CPU because
    complex-valued FFT can be unstable on Apple-Silicon MPS.

    Args:
        kernel: Convolution kernel (1, 1, k, k).
        img_shape: Spatial image size (H, W).

    Returns:
        Spectral norm L as float. The step size should satisfy alpha < 2/L
        (Ryu et al., 2019, Theorem 1).
    """
    kernel_cpu = kernel.cpu()
    padded_kernel = torch.zeros(img_shape, device="cpu")
    k_h, k_w = kernel_cpu.shape[-2], kernel_cpu.shape[-1]
    padded_kernel[:k_h, :k_w] = kernel_cpu.squeeze()
    H = torch.fft.fft2(padded_kernel)
    return torch.max(torch.abs(H) ** 2).item()


def make_denoiser_wrapper(
    denoiser: GradientStepDenoiser,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Binds a concrete denoiser early (avoids the closure bug).

    Args:
        denoiser: The GSD model to bind.

    Returns:
        Wrapper that only ensures the physical clamping to [0,1]; the GSD
        manages the autograd context internally itself.
    """
    def wrapper(x_in: torch.Tensor) -> torch.Tensor:
        return denoiser(torch.clamp(x_in, 0.0, 1.0))
    return wrapper


def make_grad_f_deblur(
    kernel: torch.Tensor, y_meas: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Creates the data-term gradient and binds kernel + measurement.

    Implements nabla f(x) = A*(A x - y) for f(x) = 1/2 ||A x - y||^2.

    Args:
        kernel: Convolution kernel of the forward operator A.
        y_meas: Noisy measurement y.

    Returns:
        Callable grad_f(x) computing the data-term gradient.
    """
    def grad_f(x_in: torch.Tensor) -> torch.Tensor:
        residual = blur_operator(x_in, kernel) - y_meas
        return blur_adjoint(residual, kernel)
    return grad_f


# ----------------------------------------------------------------------------
# Infrastructure
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Defines and parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PnP-FBS deblurring experiment (ill-posed)."
    )
    parser.add_argument(
        "--test-dir", type=str,
        default=os.path.join(BASE_DIR, "data", "archive", "images", "test"),
    )
    parser.add_argument(
        "--model-dir", type=str,
        default=os.path.join(BASE_DIR, "saved_models"),
    )
    parser.add_argument("--out-dir", type=str, default=SCRIPT_DIR)
    # FINAL: full grid -- all trained models, optimum bracketed on both
    # sides.
    parser.add_argument("--sigmas", type=int, nargs="+",
                        default=[2, 5, 10, 15, 20, 25, 30, 35])
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--kernel-sigma", type=float, default=2.0)
    # FINAL: operating point with substantial measurement noise so that the
    # regularization trade-off shows an interior optimum (see header).
    parser.add_argument("--noise-level", type=float, default=25.0,
                        help="Measurement noise level in pixel space [0,255].")
    parser.add_argument("--num-iter", type=int, default=100)
    # FINAL: under- / near-optimally / over-regularized.
    parser.add_argument("--vis-sigmas", type=int, nargs="+", default=[2, 15, 35],
                        help="Sigmas for the visual comparison (1st image).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_device() -> torch.device:
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(
    model_dir: str, sigmas: List[int], features: int, device: torch.device
) -> Dict[int, GradientStepDenoiser]:
    """Loads all available GSD models into memory once.

    Args:
        model_dir: Directory with the .pth files.
        sigmas: List of the requested noise levels.
        features: Filter count of the potential network.
        device: Target device.

    Returns:
        Mapping sigma -> loaded GSD (in eval mode).
    """
    models: Dict[int, GradientStepDenoiser] = {}
    for s in sigmas:
        path = os.path.join(model_dir, f"GSD_sigma{s}.pth")
        if not os.path.exists(path):
            print(f"[Deblur] Warning: model sigma={s} missing, skipped.")
            continue
        net = PotentialNetwork(in_channels=1, features=features).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        denoiser = GradientStepDenoiser(
            net, rescale_gradient=True, train_patch_size=40
        ).to(device)
        denoiser.eval()
        models[s] = denoiser
    return models


# ----------------------------------------------------------------------------
# Plot functions
# ----------------------------------------------------------------------------
def plot_psnr_curve(avg_results: Dict[int, float], avg_meas: float,
                    n_imgs: int, out_dir: str) -> str:
    """Plots the average PSNR over the denoiser level sigma."""
    sigmas = sorted(avg_results.keys())
    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, [avg_results[s] for s in sigmas],
             marker="o", color="green", linewidth=2, label="PnP-FBS reconstruction")
    plt.axhline(y=avg_meas, color="r", linestyle="--",
                label="Baseline (measurement)")
    plt.title(f"PnP-FBS deblurring: PSNR vs. regularization "
              f"(average over {n_imgs} images)")
    plt.xlabel(r"Denoiser level $\sigma$")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, alpha=0.6)
    plt.xticks(sigmas)
    path = os.path.join(out_dir, "illposed_pnp_psnr_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_visual_comparison(
    img_true: np.ndarray, img_meas: np.ndarray, psnr_meas: float,
    recon_images: Dict[int, np.ndarray], recon_psnr: Dict[int, float],
    out_dir: str,
) -> str:
    """Creates the visual comparison original/measurement/reconstructions."""
    n_panels = 2 + len(recon_images)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(img_true, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_meas, cmap="gray")
    axes[1].set_title(f"Measurement\n{psnr_meas:.2f} dB")
    axes[1].axis("off")

    for ax, s in zip(axes[2:], sorted(recon_images.keys())):
        ax.imshow(recon_images[s], cmap="gray")
        ax.set_title(f"Reconstruction\n$\\sigma={s}$ | {recon_psnr[s]:.2f} dB")
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "illposed_pnp_visual_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_convergence(history: List[float], sigma: int, out_dir: str) -> str:
    """Plots the residual curve as empirical convergence evidence."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(history) + 1), history,
                 marker=".", color="navy", linewidth=1.5)
    plt.title(f"PnP-FBS convergence (deblurring, $\\sigma={sigma}$)")
    plt.xlabel("Iteration $k$")
    plt.ylabel(r"Rel. residual $\|x_{k+1}-x_k\| / \|x_k\|$")
    plt.grid(True, which="both", alpha=0.5)
    path = os.path.join(out_dir, "illposed_pnp_convergence.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# ----------------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------------
def run_illposed_test(args: argparse.Namespace) -> None:
    """Runs the full deblurring experiment.

    Averages the reconstruction quality over the entire test set and
    creates the PSNR curve, a visual comparison, and a convergence curve.

    Args:
        args: Parsed command-line arguments.
    """
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[Deblur] Device: {device} | Seed: {args.seed}")
    print(f"[Deblur] Measurement noise level: {args.noise_level}/255 | "
          f"Denoiser grid: {args.sigmas}")

    # Operator and theory-guided step size.
    kernel = gaussian_kernel(args.kernel_size, args.kernel_sigma).to(device)
    L_op = compute_spectral_norm_conv(kernel, (args.crop_size, args.crop_size))
    alpha = 1.0 / L_op
    print(f"[Deblur] Spectral norm L={L_op:.4f} -> alpha=1/L={alpha:.4f} "
          f"(Ryu et al. 2019, Theorem 1)")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
    ])

    models = load_models(args.model_dir, args.sigmas, args.features, device)
    valid_sigmas = sorted(models.keys())
    if not valid_sigmas:
        print("[Deblur] No models loaded. Aborting.")
        return

    valid_ext = (".jpg", ".jpeg", ".png")
    image_files = sorted(
        f for f in os.listdir(args.test_dir) if f.lower().endswith(valid_ext)
    )
    if not image_files:
        print(f"[Deblur] No images in {args.test_dir}. Aborting.")
        return

    avg_meas = 0.0
    avg_results = {s: 0.0 for s in valid_sigmas}

    # Data for the thesis-relevant plots (from the first image).
    vis_true = vis_meas = None
    vis_psnr_meas = 0.0
    vis_recon: Dict[int, np.ndarray] = {}
    vis_recon_psnr: Dict[int, float] = {}
    convergence_history: List[float] = []
    convergence_sigma = valid_sigmas[len(valid_sigmas) // 2]  # middle sigma

    print(f"[Deblur] Evaluating over {len(image_files)} images ...")

    for idx, fname in enumerate(image_files):
        try:
            img = Image.open(os.path.join(args.test_dir, fname)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[Deblur] File skipped ({fname}): {exc}")
            continue

        x_true = transform(img).unsqueeze(0).to(device)

        # Simulate the forward model y = A x + v.
        x_blurred = blur_operator(x_true, kernel)
        noise = torch.randn_like(x_true) * (args.noise_level / 255.0)
        y_meas = x_blurred + noise

        psnr_meas = calculate_psnr(torch.clamp(y_meas, 0.0, 1.0), x_true)
        avg_meas += psnr_meas

        grad_f = make_grad_f_deblur(kernel, y_meas)

        for s in valid_sigmas:
            wrapper = make_denoiser_wrapper(models[s])
            # Record the convergence history only for the first image and
            # the middle sigma.
            want_history = (idx == 0 and s == convergence_sigma)
            x_recon, history = pnp_fbs(
                y_meas=y_meas, denoiser=wrapper, grad_f=grad_f,
                alpha=alpha, num_iter=args.num_iter,
                return_history=want_history,
            )
            if want_history and history is not None:
                convergence_history = history

            x_recon_c = torch.clamp(x_recon, 0.0, 1.0)
            psnr_recon = calculate_psnr(x_recon_c, x_true)
            avg_results[s] += psnr_recon

            # Save visual examples from the first image.
            if idx == 0 and s in args.vis_sigmas:
                vis_recon[s] = x_recon_c.squeeze().cpu().numpy()
                vis_recon_psnr[s] = psnr_recon

        if idx == 0:
            vis_true = x_true.squeeze().cpu().numpy()
            vis_meas = torch.clamp(y_meas, 0.0, 1.0).squeeze().cpu().numpy()
            vis_psnr_meas = psnr_meas

        if (idx + 1) % 5 == 0 or (idx + 1) == len(image_files):
            print(f"[Deblur] {idx + 1}/{len(image_files)} images processed ...")

    # Averages.
    n = len(image_files)
    avg_meas /= n
    for s in valid_sigmas:
        avg_results[s] /= n

    print("\n[Deblur] --- RESULTS ---")
    print(f"[Deblur] Baseline PSNR (measurement, mean): {avg_meas:.2f} dB")
    for s in valid_sigmas:
        print(f"[Deblur] sigma={s:2d} | PSNR={avg_results[s]:.2f} dB")
    best_s = max(avg_results, key=avg_results.get)
    print(f"[Deblur] Best sigma: {best_s} ({avg_results[best_s]:.2f} dB)")
    if best_s in (valid_sigmas[0], valid_sigmas[-1]):
        print("[Deblur] NOTE: optimum lies at the grid boundary -- consider "
              "discussing this as a boundary case in the thesis.")

    # Create plots.
    os.makedirs(args.out_dir, exist_ok=True)
    p1 = plot_psnr_curve(avg_results, avg_meas, n, args.out_dir)
    print(f"[Deblur] PSNR curve: {p1}")

    if vis_recon and vis_true is not None:
        p2 = plot_visual_comparison(
            vis_true, vis_meas, vis_psnr_meas, vis_recon, vis_recon_psnr,
            args.out_dir,
        )
        print(f"[Deblur] Visual comparison: {p2}")

    if convergence_history:
        p3 = plot_convergence(convergence_history, convergence_sigma, args.out_dir)
        print(f"[Deblur] Convergence curve: {p3}")


if __name__ == "__main__":
    run_illposed_test(parse_args())
