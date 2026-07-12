# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         architectures.py
#
# Purpose: Definition of the Gradient Step Denoiser (GSD). The denoising
#          operator is realized analytically as a gradient step on a learned
#          scalar energy landscape:
#
#              D_theta(x) = x - nabla_x psi_theta(x).
#
# CHANGES (Revision 2) -- fixing the size scaling of the gradient:
#   With global average pooling one has, approximately,
#       |nabla_x psi|_ij ~ 1/(H*W),
#   i.e., the strength of the denoising step decays inversely with the pixel
#   count. Training uses 40x40 patches (1,600 pixels) while testing uses
#   128x128 to 256x256 images (up to 65,536 pixels) -- at test time the
#   denoiser is therefore up to ~40x too weak. Two remedies are implemented:
#
#   VARIANT A (no retraining required):
#       GradientStepDenoiser(..., rescale_gradient=True, train_patch_size=40)
#       rescales the gradient at run time by (H*W)/train_patch_size^2 and
#       thereby exactly restores the training scale. Mathematically,
#       D(x) = x - c * nabla psi(x) is still a gradient step (on the scaled
#       potential c*psi), so the operator remains conservative.
#       Use only with average-pooling checkpoints!
#
#   VARIANT B (clean, requires retraining):
#       PotentialNetwork(..., pooling="sum") replaces average by sum
#       pooling. The gradient is then size-independent by construction.
#       After retraining, keep rescale_gradient=False.
#
#   IMPORTANT: Do NOT combine variants A and B (double scaling).
#
# References:
#   - Gradient Step Denoiser / conservative vector field as denoiser:
#       Hurault et al. (2022), "Gradient Step Denoiser for Convergent
#       Plug-and-Play", ICLR. (central methodological reference)
#   - Convergence of PnP with Lipschitz-bounded denoisers:
#       Ryu et al. (2019), ICML.
#   - Differentiable activation (ELU): Hurault et al. (2022), Sec. 5.1.
# ============================================================================

import torch
import torch.nn as nn
import torch.autograd as autograd


# ----------------------------------------------------------------------------
# PART 1: The potential network (scalar energy landscape psi_theta)
# ----------------------------------------------------------------------------
class PotentialNetwork(nn.Module):
    """Scalar-valued potential network psi_theta: R^(C x H x W) -> R.

    In contrast to a classical image-to-image CNN, this network maps an
    entire image tensor to a single real energy value. The corresponding
    denoiser is obtained from the gradient of this energy (see
    ``GradientStepDenoiser``).

    Architecture (three-stage "information funnel"):
        1. Convolutional backbone with ELU activation extracts local
           features without changing the spatial resolution (padding=1).
        2. Spatial pooling reduces (H, W) to a feature vector:
           - "avg": average pooling (original; gradient ~ 1/(H*W), leading
             to an image-size-dependent denoising strength).
           - "sum": sum pooling (Revision 2; gradient independent of the
             image size, recommended for retraining).
        3. A linear layer projects the feature vector onto the scalar
           potential value psi.

    Note on the activation function:
        ELU instead of ReLU, since double backpropagation requires an
        everywhere-differentiable activation with a Lipschitz-continuous
        gradient (cf. Hurault et al., 2022, Appendix B).

    Checkpoint compatibility:
        The pooling has no trainable parameters; existing state_dicts
        (conv_layers.*, fc.*) load in both pooling modes. The LEARNED
        weights are, however, adapted to the pooling mode used during
        training -- do not use average checkpoints in sum mode and vice
        versa.

    Attributes:
        conv_layers (nn.Sequential): Convolutional backbone with ELU
            activations.
        pooling (str): Pooling mode, "avg" or "sum".
        fc (nn.Linear): Final projection onto the scalar psi.
    """

    def __init__(
        self,
        in_channels: int = 1,
        features: int = 64,
        pooling: str = "avg",
    ) -> None:
        """Initializes the potential network.

        Args:
            in_channels: Number of input channels (1 for grayscale images).
            features: Number of convolutional filters per layer.
            pooling: "avg" (original, compatible with existing checkpoints)
                or "sum" (size-independent gradient, requires retraining).

        Raises:
            ValueError: For an unknown pooling mode.
        """
        super().__init__()

        if pooling not in ("avg", "sum"):
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        self.pooling = pooling

        # Convolutional backbone: three 3x3 convolutions with ELU.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ELU(),
        )

        # Linear projection: (B, features) -> (B, 1).
        self.fc = nn.Linear(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the scalar potential value psi(x).

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Potential tensor of shape (B, 1).
        """
        # 1. Local feature extraction.
        features = self.conv_layers(x)

        # 2. Spatial aggregation: (B, C', H, W) -> (B, C').
        if self.pooling == "sum":
            pooled = features.sum(dim=(2, 3))
        else:  # "avg"
            pooled = features.mean(dim=(2, 3))

        # 3. Final scalar projection psi.
        psi = self.fc(pooled)
        return psi


# ----------------------------------------------------------------------------
# PART 2: The Gradient Step Denoiser (GSD)
# ----------------------------------------------------------------------------
class GradientStepDenoiser(nn.Module):
    """Denoiser as analytical gradient step: D(x) = x - c * nabla psi(x).

    Implements the Gradient Step Denoiser following Hurault et al. (2022).
    By construction as the gradient of a scalar potential, the denoiser is
    a conservative vector field by definition. This also holds with active
    rescaling (factor c > 0), since c * nabla psi = nabla (c * psi) is
    still the gradient of a scalar potential.

    This wrapper has no trainable parameters of its own; it only
    manipulates the computation graph of the autograd engine.

    Attributes:
        potential_net (PotentialNetwork): The encapsulated potential network.
        rescale_gradient (bool): If True, the gradient is scaled by
            (H*W)/train_patch_size^2 (variant A; only for average-pooling
            checkpoints).
        train_patch_size (int): Patch edge length used during training
            (reference scale for the rescaling).
    """

    def __init__(
        self,
        potential_net: PotentialNetwork,
        rescale_gradient: bool = False,
        train_patch_size: int = 40,
    ) -> None:
        """Initializes the GSD with a potential network.

        Args:
            potential_net: Instance of ``PotentialNetwork``.
            rescale_gradient: Activates the run-time compensation of the
                GAP size scaling (variant A). ONLY use if potential_net was
                trained with average pooling. For sum-pooling models
                (variant B), keep this False.
            train_patch_size: Edge length of the training patches (default
                40, cf. Ryu et al. 2019).

        Raises:
            ValueError: If rescaling is combined with a sum-pooling network
                (double scaling).
        """
        super().__init__()
        if rescale_gradient and getattr(potential_net, "pooling", "avg") == "sum":
            raise ValueError(
                "rescale_gradient=True must not be combined with "
                "pooling='sum' (double size scaling)."
            )
        self.potential_net = potential_net
        self.rescale_gradient = rescale_gradient
        self.train_patch_size = train_patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a gradient step on the energy landscape.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            Denoised image tensor D(x) = x - c * nabla_x psi(x), same shape
            as the input. c = (H*W)/train_patch_size^2 if
            rescale_gradient=True, otherwise c = 1.
        """
        # CRITICAL MECHANISM:
        # Inside a PnP loop the surrounding code typically runs under
        # torch.no_grad(). Since the denoising step, however, *is* a
        # derivative by definition, the graph must be enforced for this
        # block via torch.enable_grad().
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)

            # 1. Scalar energy psi(x).
            psi = self.potential_net(x_in)

            # 2. Analytical gradient nabla_x psi(x).
            #    create_graph=self.training enables double backpropagation
            #    during training and discards the graph in evaluation (see
            #    original documentation).
            grad_psi = autograd.grad(
                outputs=psi,
                inputs=x_in,
                grad_outputs=torch.ones_like(psi),
                create_graph=self.training,
                only_inputs=True,
            )[0]

        # 2b. VARIANT A: compensation of the GAP size scaling.
        #     With average pooling, |grad_psi| ~ 1/(H*W); the factor
        #     (H*W)/train_patch_size^2 restores the gradient magnitude of
        #     the training condition (40x40 patches).
        if self.rescale_gradient:
            h, w = x_in.shape[-2], x_in.shape[-1]
            scale = (h * w) / float(self.train_patch_size ** 2)
            grad_psi = grad_psi * scale

        # 3. Formal denoising step.
        x_denoised = x_in - grad_psi

        return x_denoised


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # Smoke test: verifies (a) the scalar output of psi, (b) the shape
    # preservation of the GSD, (c) double backpropagation, and (d) the size
    # independence of the gradient with rescaling or sum pooling.
    # ------------------------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test for PotentialNetwork and GradientStepDenoiser."
    )
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=40)
    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[Smoke test] Device: {device}")

    potential_net = PotentialNetwork(
        in_channels=args.in_channels, features=args.features
    ).to(device)
    gsd = GradientStepDenoiser(potential_net).to(device)

    dummy = torch.rand(
        args.batch_size, args.in_channels, args.patch_size, args.patch_size,
        device=device,
    )

    # (a) Potential yields one scalar per batch element.
    psi = potential_net(dummy)
    print(f"[Smoke test] psi shape = {tuple(psi.shape)} (expected ({args.batch_size}, 1))")

    # (b) Inference mode: the GSD preserves the shape.
    gsd.eval()
    out_eval = gsd(dummy)
    print(f"[Smoke test] D(x) shape (eval) = {tuple(out_eval.shape)}")

    # (c) Training mode: double backpropagation must run through.
    gsd.train()
    out_train = gsd(dummy)
    loss = (out_train - dummy).pow(2).mean()
    loss.backward()
    grad_norm = potential_net.fc.weight.grad.norm().item()
    print(f"[Smoke test] Double backprop OK | loss={loss.item():.4f} | "
          f"||grad fc.weight||={grad_norm:.4e}")

    # (d) Size scaling: gradient magnitude on 40x40 vs. 256x256.
    gsd.eval()
    with torch.no_grad():
        for size in (args.patch_size, 256):
            big = torch.rand(1, args.in_channels, size, size, device=device)
            # Without rescaling.
            gsd.rescale_gradient = False
            d_plain = (gsd(big) - big).abs().mean().item()
            # With rescaling (variant A).
            gsd.rescale_gradient = True
            d_rescaled = (gsd(big) - big).abs().mean().item()
            print(f"[Smoke test] {size}x{size} | mean |D(x)-x| "
                  f"without rescale={d_plain:.3e}, with rescale={d_rescaled:.3e}")
    gsd.rescale_gradient = False
