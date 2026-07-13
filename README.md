# Plug-and-Play Image Reconstruction with Gradient Step Denoisers

Code accompanying the master's thesis

> **"Plug-and-Play in Image Processing: Convergent Regularization using
> Gradient Step Denoisers"**
> Vitus Mangold, University of Mannheim, 2026
> Chair of Mathematical Optimization (Prof. Dr. M. Staudigl)

The repository implements a **Gradient Step Denoiser (GSD)**, a denoiser
defined analytically as a gradient step on a learned scalar energy
landscape, `D(x) = x − ∇ψ(x)` (Hurault et al., 2022) and plugs it into
the **Plug-and-Play (PnP)** solvers **FBS** and **ADMM** (Ryu et al., 2019)
to solve three inverse problems: deblurring, inpainting, and reconstruction
under Poisson noise.

## Key finding: GAP size scaling of the gradient

The potential network aggregates features via **global average pooling
(GAP)**. As a consequence, the per-pixel gradient magnitude scales like
`1/(H·W)`: a model trained on 40×40 patches is ~40× too weak on 256×256
test images. We verify this quantitatively, the measured full-image/patch
denoising-gain ratio of **0.023** matches the theoretical prediction
`40²/256² ≈ 0.024` and compensate it at run time by rescaling the
gradient with `(H·W)/40²` ("variant A", which keeps the operator a
conservative vector field). See `experiments/analyze_scale_effect.py` and
Section 5.2 of the thesis.

## Repository structure

```
.
├── data/
│   └── dataset.py              # BSD500 AWGN dataset (residual learning)
├── models/
│   ├── architectures.py        # PotentialNetwork + GradientStepDenoiser
│   └── pnp_solvers.py          # Generic PnP-FBS and PnP-ADMM
├── training/
│   └── train_gsd.py            # GSD training (double backpropagation)
├── experiments/
│   ├── analyze_denoiser.py     # Identity property Φ_σ(y) → y as σ → 0
│   ├── analyze_scale_effect.py # GAP size-scaling demonstration
│   ├── test_illposed_pnp.py    # Deblurring (PnP-FBS, σ-sweep)
│   ├── test_inpainting.py      # Inpainting, 70% missing pixels (PnP-FBS)
│   └── test_gsd_benchmark.py   # Poisson noise: PnP-ADMM vs. PnP-FBS
└── saved_models/               # Trained checkpoints GSD_sigma{σ}.pth
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 and PyTorch ≥ 2.x on Apple Silicon (MPS); CUDA and
CPU are selected automatically.

**Data.** Download the [Berkeley Segmentation Dataset
(BSD500)](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)
and place the images under `data/archive/images/{train,val,test}/`. The
dataset itself is not distributed with this repository.

## Usage

Train one GSD per noise level (checkpoints land in `saved_models/`):

```bash
python training/train_gsd.py --sigma 15      # repeat for 2 5 10 15 20 25 30 35
```

Reproduce all experiments and figures (all runs are seeded, seed 42):

```bash
python experiments/analyze_scale_effect.py   # GAP scaling demonstration
python experiments/analyze_denoiser.py       # identity property
python experiments/test_illposed_pnp.py      # deblurring σ-sweep
python experiments/test_inpainting.py        # inpainting, 70% missing
python experiments/test_gsd_benchmark.py     # Poisson: ADMM vs. FBS
```

## Results (BSD500 test set, N = 200 images, seed 42)

| Experiment | Setting | Result |
|---|---|---|
| Denoising, patch vs. full image | σ = 15 | gain +6.77 dB (40×40) vs. +0.15 dB (256², no rescale) vs. +5.57 dB (with rescale); ratio 0.023 ≈ 40²/256² |
| Deblurring (PnP-FBS) | Gaussian blur 9×9, σ_blur = 2, meas. noise 25/255 | baseline 18.30 dB → best 23.62 dB at denoiser σ = 10 (interior optimum of the σ-sweep) |
| Inpainting (PnP-FBS) | 70 % pixels missing, noise 0.05 | 8.23 dB → 24.51 dB (mean gain +16.28 dB) |
| Poisson (peak = 60, 20 iter.) | PnP-ADMM vs. PnP-FBS | ADMM 22.95 dB / SSIM 0.546 vs. FBS 22.05 dB / SSIM 0.496 |
| Identity property | clean inputs, 256² | ‖Φ_σ(y) − y‖² decays monotonically from 2.1e-3 (σ=35) to 1.0e-5 (σ=2) |

## References

- S. Hurault, A. Leclaire, N. Papadakis (2022). *Gradient Step Denoiser
  for Convergent Plug-and-Play.* ICLR.
- E. K. Ryu, J. Liu, S. Wang, X. Chen, Z. Wang, W. Yin (2019).
  *Plug-and-Play Methods Provably Converge with Properly Trained
  Denoisers.* ICML.
- A. Ebner, M. Haltmeier (2022). *Plug-and-Play image reconstruction is a convergent regularization method.*
- K. Zhang, W. Zuo, Y. Chen, D. Meng, L. Zhang (2017). *Beyond a Gaussian
  Denoiser: Residual Learning of Deep CNN for Image Denoising.* IEEE TIP.
- D. Martin, C. Fowlkes, D. Tal, J. Malik (2001). *A Database of Human
  Segmented Natural Images.* ICCV (BSD500).
- H. W. Engl, M. Hanke, A. Neubauer (1996). *Regularization of Inverse
  Problems.* Kluwer.
