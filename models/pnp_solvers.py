# ============================================================================
# Master's thesis: "Plug-and-Play in Image Processing:
#                   Convergent Regularization using Gradient Step Denoisers"
# Author:       Vitus Mangold, University of Mannheim, 2026
# Chair:        Mathematical Optimization (Prof. Dr. M. Staudigl)
# File:         pnp_solvers.py
#
# Purpose: Generic meta-algorithms of the Plug-and-Play (PnP) family. Both
#          the physical forward model (via grad_f or prox_f) and the
#          denoiser are passed as executable functions (callables). This
#          mirrors the mathematical decoupling of data term and prior term
#          and makes the solvers independent of the concrete inverse
#          problem.
#
# CHANGES (Revision 2):
#   (1) PnP-ADMM now initializes x with the measurement (instead of zero).
#       Reason: with x^0 = 0 one has ||x_old|| = 0 in the first step; the
#       relative residual is then dominated by the epsilon denominator and
#       explodes artificially (~1e14 in the plot), even though there is no
#       divergence.
#   (2) _relative_residual uses a symmetrized denominator
#       max(||x_old||, ||x_new||) that provides a sensible scale even when
#       the norm of x_old vanishes.
#
# References:
#   - PnP-FBS and PnP-ADMM, convergence conditions:
#       Ryu et al. (2019), "Plug-and-Play Methods Provably Converge with
#       Properly Trained Denoisers", ICML.
#         * PnP-FBS:  x^{k+1} = H(I - alpha * grad f)(x^k)        (Sec. 2)
#         * PnP-ADMM: x^{k+1} = H(y^k - u^k)
#                     y^{k+1} = Prox_{alpha f}(x^{k+1} + u^k)
#                     u^{k+1} = u^k + x^{k+1} - y^{k+1}           (Sec. 2)
#   - Convergence as a regularization method / residual -> 0:
#       Ebner & Haltmeier (2024); Hurault et al. (2022), Fig. 1(h).
# ============================================================================

from typing import Callable, List, Optional, Tuple

import torch


def pnp_fbs(
    y_meas: torch.Tensor,
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    grad_f: Callable[[torch.Tensor], torch.Tensor],
    alpha: float,
    num_iter: int = 50,
    tol: float = 0.0,
    verbose: bool = False,
    return_history: bool = False,
) -> Tuple[torch.Tensor, Optional[List[float]]]:
    """Plug-and-Play Forward-Backward Splitting (PnP-FBS).

    Implements the fixed-point iteration following Ryu et al. (2019), Sec. 2:

        x^{k+1} = H( x^k - alpha * grad_f(x^k) ).

    The iteration decomposes into two steps:
        1. Forward step (data fidelity): z = x - alpha * grad_f(x). Pulls
           the image towards the measurement; amplifies noise in doing so.
        2. Backward step (regularization): x = denoiser(z). Replaces the
           classical proximal step by a forward pass of the (Gradient Step)
           denoiser.

    Convergence: By Ryu et al. (2019), Theorem 1, the iteration converges
    linearly to a unique fixed point, provided f is mu-strongly convex with
    L-Lipschitz gradient, the denoiser satisfies Assumption (A), and the
    step size alpha lies in the interval specified there. The returned
    residual history serves as an empirical convergence certificate (cf.
    Hurault et al., 2022, Fig. 1h).

    Args:
        y_meas: Noisy measurement; also serves as the initialization x^0.
        denoiser: Callable executing the (GSD) denoising step. The GSD
            manages its autograd context internally.
        grad_f: Callable computing the gradient of the data term
            nabla f(x) (e.g., A^*(Ax - y) for f(x) = 1/2 ||Ax - y||^2).
        alpha: Step size of the forward step. Should be chosen according to
            the Lipschitz constant L of grad_f (Ryu Theorem 1).
        num_iter: Maximum number of iterations.
        tol: Relative stopping criterion for the residual
            ||x^{k+1} - x^k|| / max(||x^k||, ||x^{k+1}||). Disabled at 0.0
            (full num_iter).
        verbose: If True, the residual is printed per iteration.
        return_history: If True, the list of relative residuals is returned
            in addition (for convergence plots).

    Returns:
        Tuple ``(x, history)``:
            - x (Tensor): reconstructed image (fixed-point approximation).
            - history (list[float] | None): relative residuals per
              iteration if ``return_history=True``, otherwise None.
    """
    x = y_meas.clone()
    history: List[float] = []

    for k in range(num_iter):
        # 1. Forward step (data fidelity).
        #    grad_f is evaluated under no_grad: the data-term gradient
        #    requires no autograd graph; otherwise PyTorch builds
        #    unnecessary graphs over 100+ iterations (memory/time overhead).
        with torch.no_grad():
            gradient = grad_f(x)
            z = x - alpha * gradient

        # 2. Backward step (regularization).
        #    The GSD internally re-enables torch.enable_grad() exactly for
        #    the computation of nabla_x psi and discards the graph after.
        with torch.no_grad():
            x_new = denoiser(z)

        # Relative residual as convergence indicator.
        residual = _relative_residual(x_new, x)
        history.append(residual)
        if verbose:
            print(f"[PnP-FBS] Iter {k + 1:3d} | rel. residual = {residual:.3e}")

        x = x_new

        # Early stopping when the tolerance is reached.
        if tol > 0.0 and residual < tol:
            if verbose:
                print(f"[PnP-FBS] Converged after {k + 1} iterations "
                      f"(residual < {tol:.1e}).")
            break

    return (x, history) if return_history else (x, None)


def pnp_admm(
    y_meas: torch.Tensor,
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    prox_f: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int = 50,
    tol: float = 0.0,
    verbose: bool = False,
    return_history: bool = False,
) -> Tuple[torch.Tensor, Optional[List[float]]]:
    """Plug-and-Play Alternating Direction Method of Multipliers (PnP-ADMM).

    Implements the update scheme following Ryu et al. (2019), Sec. 2:

        x^{k+1} = H( y^k - u^k )                 (prior/denoising step)
        y^{k+1} = Prox_{alpha f}( x^{k+1} + u^k )  (data-fidelity step)
        u^{k+1} = u^k + x^{k+1} - y^{k+1}          (dual step)

    PnP-ADMM evaluates the data term implicitly via a proximal operator and
    is therefore more robust for non-smooth or ill-conditioned forward
    models (e.g., Poisson noise), where explicit forward-backward splitting
    becomes numerically unstable.

    Initialization (Revision 2): Both the y-variable and the x-variable are
    initialized with the measurement, the dual variable u with zero. The
    previous initialization x^0 = 0 produced an artificially exploding
    relative residual in the first iteration (division by ||x^0|| = 0),
    which falsely looked like an instability. Asymptotically, the
    initialization has no influence on the fixed point (at the fixed point
    x = y holds).

    Args:
        y_meas: Noisy measurement; initializes the x- and y-variables.
        denoiser: Callable for the (GSD) denoising step.
        prox_f: Callable evaluating the proximal operator of the data term
            Prox_{alpha f} (e.g., closed-form solution for the Poisson/KL
            data term, cf. Ryu et al., 2019, Sec. 5).
        num_iter: Maximum number of iterations.
        tol: Relative stopping criterion for the residual
            ||x^{k+1} - x^k|| / max(||x^k||, ||x^{k+1}||). Disabled at 0.0.
        verbose: If True, the residual is printed per iteration.
        return_history: If True, the list of relative residuals is returned
            in addition.

    Returns:
        Tuple ``(x, history)``:
            - x (Tensor): reconstructed image (denoising output of the last
              step).
            - history (list[float] | None): relative residuals per
              iteration if ``return_history=True``, otherwise None.
    """
    # Initialization: x^0 = y^0 = measurement, u^0 = 0 (see docstring).
    y = y_meas.clone()
    u = torch.zeros_like(y_meas)
    x = y_meas.clone()

    history: List[float] = []

    for k in range(num_iter):
        x_prev = x

        # 1. Prior/denoising step (x-update).
        #    The GSD re-enables its gradient tracking internally; the outer
        #    no_grad keeps the rest of the loop memory-efficient.
        with torch.no_grad():
            x = denoiser(y - u)

        # 2. Data-fidelity step (y-update via proximal operator).
        y = prox_f(x + u)

        # 3. Dual step (u-update): accumulates the consensus residual
        #    x - y and drives both subproblems into a common fixed point.
        u = u + x - y

        # Relative residual of the prior output as convergence indicator.
        residual = _relative_residual(x, x_prev)
        history.append(residual)
        if verbose:
            print(f"[PnP-ADMM] Iter {k + 1:3d} | rel. residual = {residual:.3e}")

        if tol > 0.0 and residual < tol:
            if verbose:
                print(f"[PnP-ADMM] Converged after {k + 1} iterations "
                      f"(residual < {tol:.1e}).")
            break

    return (x, history) if return_history else (x, None)


def _relative_residual(x_new: torch.Tensor, x_old: torch.Tensor) -> float:
    """Computes a robust relative residual.

    Definition (symmetrized denominator):

        r = ||x_new - x_old|| / max(||x_old||, ||x_new||, eps).

    The symmetrized denominator prevents an artificial explosion of the
    residual when one of the two iterates (e.g., the initialization) has a
    vanishing norm. A residual decaying to 0 is the empirical confirmation
    of asymptotic regularity (cf. Theorem 3.5 (ii) of the thesis).

    Args:
        x_new: Iterate of the current step.
        x_old: Iterate of the previous step.

    Returns:
        Relative residual as float.
    """
    numerator = torch.norm(x_new - x_old)
    denominator = torch.maximum(torch.norm(x_old), torch.norm(x_new))
    denominator = torch.clamp(denominator, min=1e-8)
    return (numerator / denominator).item()


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # Smoke test: simple linear model f(x) = 1/2 ||x - y||^2 (identity
    # forward operator A = I). Expectation: the residual decays
    # monotonically and the solver converges to the measurement when the
    # denoiser is the identity.
    # ------------------------------------------------------------------------
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke test for pnp_fbs and pnp_admm (identity model)."
    )
    parser.add_argument("--num-iter", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    torch.manual_seed(0)
    y = torch.rand(1, 1, 16, 16)

    # Trivial denoiser (identity) and data term for A = I.
    identity_denoiser = lambda v: v
    grad_identity = lambda v: v - y              # nabla (1/2 ||x - y||^2)
    prox_identity = lambda v: 0.5 * (v + y)      # prox of 1/2 ||x - y||^2, alpha=1

    x_fbs, hist_fbs = pnp_fbs(
        y, identity_denoiser, grad_identity, alpha=args.alpha,
        num_iter=args.num_iter, verbose=True, return_history=True,
    )
    print(f"[Smoke test] FBS final residual = {hist_fbs[-1]:.3e}\n")

    x_admm, hist_admm = pnp_admm(
        y, identity_denoiser, prox_identity,
        num_iter=args.num_iter, verbose=True, return_history=True,
    )
    print(f"[Smoke test] ADMM final residual = {hist_admm[-1]:.3e}")
