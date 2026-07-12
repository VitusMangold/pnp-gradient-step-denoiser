# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         analyze_denoiser.py
#
# Purpose: Empirical verification of the identity property of the GSD
#          operator. For a classical regularization method, the denoiser
#          must converge to the identity in the limit of vanishing
#          regularization strength:
#
#              lim_{sigma -> 0} Phi_sigma(y) = y   for all y.
#
#          We measure the mean L2 deviation ||Phi_sigma(y) - y||^2 over a
#          family of GSD models trained at decreasing noise levels. A value
#          decaying to 0 as sigma -> 0 confirms the property.
#
# IMPORTANT METHODOLOGICAL NOTE (test noise):
#   The identity property is a statement about the OPERATOR Phi_sigma, not
#   about its denoising performance. To measure it cleanly, the test noise
#   must be chosen consistently:
#     * mode 'clean' : input = clean image (test noise = 0). Measures the
#                      operator deviation directly at the origin -- the
#                      theoretically most correct variant.
#     * mode 'match' : test noise sigma_test = sigma_train. Measures how
#                      close the denoiser is to the identity at its DESIGN
#                      noise level.
#   A FIXED test noise (e.g., always sigma=25) is NOT suitable, since it
#   creates an out-of-distribution scenario for small sigma_train and thus
#   distorts the identity property.
#
# Reference:
#   - Limit behavior of regularization methods / proximal operators:
#       Engl, Hanke, Neubauer (1996); cf. Sections 2.2 and 5.5.1 of the
#       thesis.
# ============================================================================

import os
import sys
import argparse
from typing import Dict, List

import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(BASE_DIR)

from models.architectures import PotentialNetwork, GradientStepDenoiser


def parse_args() -> argparse.Namespace:
    """Defines and parses the command-line arguments.

    Returns:
        Namespace with paths and analysis parameters.
    """
    parser = argparse.ArgumentParser(
        description="Analysis of the identity property of the GSD denoiser."
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=os.path.join(BASE_DIR, "data", "archive", "images", "test"),
        help="Directory with (unseen) test images.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(BASE_DIR, "saved_models"),
        help="Directory with the trained GSD models.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=SCRIPT_DIR,
        help="Target directory for the result plot.",
    )
    parser.add_argument(
        "--sigmas",
        type=int,
        nargs="+",
        default=[2, 5, 10, 15, 20, 25, 30, 35],
        help="List of trained noise levels (model file names).",
    )
    parser.add_argument(
        "--crop-size", type=int, default=256, help="Central image crop."
    )
    parser.add_argument(
        "--features", type=int, default=64,
        help="Filter count of the potential network (must match training).",
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["clean", "match"],
        default="clean",
        help="Test noise: 'clean' (=0) or 'match' (=sigma_train).",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_device() -> torch.device:
    """Selects the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_test_images(test_dir: str, transform: transforms.Compose,
                     device: torch.device) -> List[torch.Tensor]:
    """Loads and transforms all test images into memory once.

    Args:
        test_dir: Directory with test images.
        transform: Preprocessing pipeline (grayscale, crop, ToTensor).
        device: Target device for the tensors.

    Returns:
        List of preprocessed image tensors of shape (1, 1, H, W).

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If no valid images are found.
    """
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    valid_ext = (".jpg", ".jpeg", ".png")
    files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith(valid_ext))

    images: List[torch.Tensor] = []
    for fname in files:
        try:
            img = Image.open(os.path.join(test_dir, fname)).convert("RGB")
            images.append(transform(img).unsqueeze(0).to(device))
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[Analysis] Warning: file skipped ({fname}): {exc}")

    if not images:
        raise ValueError(f"No loadable test images in: {test_dir}")
    return images


def measure_identity_deviation(
    denoiser: GradientStepDenoiser,
    images: List[torch.Tensor],
    sigma_test: float,
) -> float:
    """Measures the mean deviation ||Phi(y) - y||^2 over all test images.

    Args:
        denoiser: The GSD under test (in eval mode).
        images: List of clean image tensors.
        sigma_test: Standard deviation of the test noise (on [0,1] scale).
            At 0 the clean image is fed in directly.

    Returns:
        Average MSE between denoiser output and input.
    """
    total_mse = 0.0
    # Outer no_grad: no training graph needed. The GSD re-enables
    # enable_grad() INTERNALLY only for the psi derivative -- an additional
    # enable_grad() or requires_grad_() from outside is unnecessary.
    with torch.no_grad():
        for x_clean in images:
            if sigma_test > 0.0:
                y = x_clean + torch.randn_like(x_clean) * sigma_test
            else:
                y = x_clean

            y_denoised = denoiser(y)

            # Deviation from the identity: ||Phi_sigma(y) - y||^2.
            # Deliberately NOT against the clean image but against the INPUT.
            mse = torch.nn.functional.mse_loss(y_denoised, y).item()
            total_mse += mse

    return total_mse / len(images)


def plot_results(results: Dict[int, float], out_dir: str, mode: str) -> str:
    """Creates the convergence plot of the identity deviation.

    Args:
        results: Mapping sigma_train -> mean MSE.
        out_dir: Target directory for the PNG file.
        mode: Test mode used ('clean' or 'match'), for the title.

    Returns:
        Path of the saved plot file.
    """
    sigmas = sorted(results.keys())
    values = [results[s] for s in sigmas]

    plt.figure(figsize=(8, 5))
    plt.plot(sigmas, values, marker="s", markersize=8, linestyle="-",
             linewidth=2, color="darkblue")
    plt.title(
        r"Convergence to Identity: $\Phi_{\sigma}(y) \approx y$ as "
        r"$\sigma \to 0$" + f"  (test mode: {mode})",
        fontsize=14,
    )
    plt.xlabel(r"Trained Noise Level $\sigma$", fontsize=12)
    plt.ylabel(r"Average $\|\Phi_{\sigma}(y) - y\|^2$ (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(sigmas)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "gsd_identity_convergence.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Analysis] Plot saved to: {save_path}")
    return save_path


def run_analysis(args: argparse.Namespace) -> None:
    """Runs the full identity analysis over all models.

    Args:
        args: Parsed command-line arguments.
    """
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"[Analysis] Device: {device} | Test mode: {args.test_mode}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
    ])

    images = load_test_images(args.test_dir, transform, device)
    print(f"[Analysis] Loaded {len(images)} test images.")

    avg_mse_results: Dict[int, float] = {}

    for s in args.sigmas:
        model_path = os.path.join(args.model_dir, f"GSD_sigma{s}.pth")
        if not os.path.exists(model_path):
            print(f"[Analysis] Model sigma={s} missing. Skipping ...")
            continue

        potential_net = PotentialNetwork(
            in_channels=1, features=args.features
        ).to(device)
        potential_net.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        denoiser = GradientStepDenoiser(potential_net, rescale_gradient=True, train_patch_size=40).to(device)
        denoiser.eval()

        # Couple the test noise to sigma_train or set it to zero, depending
        # on the mode.
        sigma_test = 0.0 if args.test_mode == "clean" else (s / 255.0)

        avg_mse = measure_identity_deviation(denoiser, images, sigma_test)
        avg_mse_results[s] = avg_mse
        print(f"[Analysis] sigma={s:2d} | mean identity deviation "
              f"||Phi(y) - y||^2 = {avg_mse:.6f}")

    if avg_mse_results:
        plot_results(avg_mse_results, args.out_dir, args.test_mode)
    else:
        print("[Analysis] No models found -- no plot created.")


if __name__ == "__main__":
    run_analysis(parse_args())
