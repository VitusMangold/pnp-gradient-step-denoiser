# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         dataset.py
#
# Purpose: PyTorch dataset simulating the additive white Gaussian noise
#          (AWGN) model for training the Gradient Step Denoiser (GSD).
#
# References:
#   - AWGN noise model and patch-based training:
#       Ryu et al. (2019), "Plug-and-Play Methods Provably Converge with
#       Properly Trained Denoisers", ICML. (40x40 patches, BSD500)
#   - Residual learning scheme (network learns the noise v instead of x):
#       Zhang et al. (2017), "Beyond a Gaussian Denoiser: Residual Learning
#       of Deep CNN for Image Denoising", IEEE TIP. (original source)
#   - Dataset:
#       Martin et al. (2001), "A Database of Human Segmented Natural Images"
#       (Berkeley Segmentation Dataset, BSD500).
# ============================================================================

import os
import glob
from typing import Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


class BSD500DenoisingDataset(Dataset):
    """Dataset simulating the AWGN denoising problem on BSD500.

    Implements the classical linear perturbation model

        y = x + v,    v ~ N(0, sigma^2 * I)

    where x denotes the clean signal and v additive white Gaussian noise
    (AWGN). On each access, a random image crop (patch) is extracted,
    converted to grayscale, normalized to [0, 1], and corrupted with
    freshly drawn noise.

    The dataset follows the *residual learning* principle (Zhang et al.,
    2017): the training target is not the clean image x but the isolated
    noise tensor v. The network thus approximates the residual R(y) ~ v;
    the image reconstruction follows algebraically as x_rec = y - R(y).
    This choice simplifies the loss landscape and is structurally required
    to later formulate the denoiser as the gradient of a scalar potential
    (cf. Hurault et al., 2022).

    Note on epoch length:
        Since a *random* crop is drawn on each access, the network sees many
        different patches per original image. The parameter
        ``samples_per_image`` artificially stretches the nominal dataset
        length so that several crops per image are drawn per epoch. This
        decouples the definition of an "epoch" from the raw image count and
        must be taken into account when tuning the learning-rate scheduler.

    Attributes:
        image_paths (list[str]): Sorted list of paths of all indexed image
            files.
        sigma (float): Standard deviation of the noise, scaled to the value
            range [0, 1] (i.e., sigma_pixel / 255).
        samples_per_image (int): Multiplier for the nominal dataset length
            (number of random crops per image and epoch).
        clamp_output (bool): If True, the noisy image is clipped to [0, 1]
            to avoid a train/inference mismatch with clamped PnP inputs.
        transform (torchvision.transforms.Compose): Augmentation pipeline.
    """

    VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png")

    def __init__(
        self,
        image_dir: str,
        patch_size: int = 40,
        sigma: float = 15.0,
        samples_per_image: int = 10,
        clamp_output: bool = False,
    ) -> None:
        """Initializes the dataset and indexes all image files.

        Args:
            image_dir: Path to the folder with training images (e.g., the
                BSD500 training split).
            patch_size: Edge length of the square patches in pixels.
                Default 40 following Ryu et al. (2019).
            sigma: Noise level with respect to the pixel value range
                [0, 255]. Internally normalized to [0, 1] (sigma / 255).
            samples_per_image: Number of random crops per image per epoch.
                Stretches the nominal dataset length accordingly.
            clamp_output: If True, the noisy input image is clipped to
                [0, 1]. Recommended if the PnP solver uses clamped inputs
                (train/inference consistency).

        Raises:
            FileNotFoundError: If ``image_dir`` does not exist.
            ValueError: If no valid image files are found in the directory.
        """
        super().__init__()

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Robust, platform-independent search over all valid extensions.
        # Sorting guarantees a deterministic index order (important for
        # reproducibility).
        self.image_paths = sorted(
            p
            for ext in self.VALID_EXTENSIONS
            for p in glob.glob(os.path.join(image_dir, f"*{ext}"))
        )

        if len(self.image_paths) == 0:
            raise ValueError(
                f"No image files ({', '.join(self.VALID_EXTENSIONS)}) "
                f"found in directory: {image_dir}"
            )

        print(
            f"[Dataset] Indexed {len(self.image_paths)} images in "
            f"'{image_dir}' (sigma={sigma}, patch_size={patch_size})."
        )

        self.sigma = sigma / 255.0  # normalize to the tensor value range [0, 1]
        self.samples_per_image = max(1, samples_per_image)
        self.clamp_output = clamp_output

        # Augmentation pipeline following Ryu et al. (2019):
        #   - Grayscale: the GSD is designed for single-channel intensity images.
        #   - RandomCrop: enforces translation invariance of the learned filters.
        #   - Flips: make the prior isotropic w.r.t. these symmetries.
        #   - ToTensor: normalizes discrete pixel intensities to [0, 1].
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(patch_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """Returns the nominal dataset length.

        Returns:
            Product of image count and ``samples_per_image``. This value
            controls how many patches the DataLoader draws per epoch.
        """
        return len(self.image_paths) * self.samples_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads an image, extracts a patch, and simulates AWGN.

        Mirrors the mathematical equation y = x + v step by step: a clean
        patch x is augmented, a fresh noise tensor v ~ N(0, sigma^2 * I) is
        drawn and added.

        Args:
            idx: Linear index in [0, len(self)). Mapped onto the actual
                image count via modulo, so each real index is drawn multiple
                times (with different random crops).

        Returns:
            A tuple ``(noisy_patch, noise)``:
                - noisy_patch (Tensor): noisy input image y of shape
                  (1, patch_size, patch_size).
                - noise (Tensor): isolated noise target v of the same shape
                  (residual-learning target).
        """
        # Modulo maps the stretched index onto a real image.
        real_idx = idx % len(self.image_paths)
        img_path = self.image_paths[real_idx]

        # Robust loading: broken files are skipped iteratively.
        # An iterative (rather than recursive) search prevents a stack
        # overflow when several consecutive images are broken.
        img = self._load_image_safe(real_idx)

        # 1. Clean reference patch x (augmented, normalized to [0, 1]).
        clean_patch = self.transform(img)

        # 2. Additive white Gaussian noise v ~ N(0, sigma^2 * I).
        #    randn_like samples from N(0, 1); scaling by sigma sets the
        #    target standard deviation.
        noise = torch.randn_like(clean_patch) * self.sigma

        # 3. Noisy observation signal y = x + v.
        noisy_patch = clean_patch + noise

        # Optional: clip to [0, 1] if the PnP solver expects clamped inputs
        # (avoids a train/inference mismatch).
        if self.clamp_output:
            noisy_patch = torch.clamp(noisy_patch, 0.0, 1.0)
            # Adjust the target to the clamped y so that y - noise = x holds.
            noise = noisy_patch - clean_patch

        # Residual-learning convention (Zhang et al., 2017):
        #   input = y (noisy_patch), target = v (noise).
        return noisy_patch, noise

    def _load_image_safe(self, start_idx: int) -> Image.Image:
        """Loads an image robustly and skips broken files iteratively.

        Args:
            start_idx: Start index into ``self.image_paths``.

        Returns:
            The loaded PIL image in RGB mode.

        Raises:
            RuntimeError: If not a single file can be loaded.
        """
        n = len(self.image_paths)
        for offset in range(n):
            current_idx = (start_idx + offset) % n
            img_path = self.image_paths[current_idx]
            try:
                return Image.open(img_path).convert("RGB")
            except (UnidentifiedImageError, OSError) as exc:
                print(f"[Dataset] Warning: file not loadable ({img_path}): {exc}")

        raise RuntimeError(
            "No loadable image file found in the dataset (all broken?)."
        )


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # Configuration via argparse (no absolute paths in the code).
    # This block serves as a smoke test: it indexes the dataset, draws a
    # single sample, and prints its statistics.
    # ------------------------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test for BSD500DenoisingDataset (AWGN simulation)."
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Path to the folder with training images (e.g., BSD500 train).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=40, help="Edge length of the patches."
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=15.0,
        help="Noise level in pixel space [0, 255] (normalized to [0,1]).",
    )
    parser.add_argument(
        "--samples-per-image",
        type=int,
        default=10,
        help="Random crops per image and epoch.",
    )
    parser.add_argument(
        "--clamp-output",
        action="store_true",
        help="Clip the noisy image to [0, 1] (consistency with PnP).",
    )
    args = parser.parse_args()

    dataset = BSD500DenoisingDataset(
        image_dir=args.image_dir,
        patch_size=args.patch_size,
        sigma=args.sigma,
        samples_per_image=args.samples_per_image,
        clamp_output=args.clamp_output,
    )

    print(f"[Smoke test] Nominal dataset length: {len(dataset)}")

    noisy, target_noise = dataset[0]
    print(
        f"[Smoke test] Sample 0 | "
        f"noisy shape={tuple(noisy.shape)}, "
        f"noisy range=[{noisy.min():.3f}, {noisy.max():.3f}] | "
        f"noise std={target_noise.std():.4f} "
        f"(expected ~{args.sigma / 255.0:.4f})"
    )
