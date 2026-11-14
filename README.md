# BlindDecon.jl

A **blind deconvolution toolkit** for 1-D signals (seismic/sparse spike) and
2-D images, written in **100% Julia standard library**. No external packages:
hand-written FFT (radix-2 Cooley-Tukey + Bluestein for arbitrary lengths),
hand-written 24-bit BMP reader/writer, hand-written SVG figures, all solvers
built from scratch. Julia >= 1.9, single file, zero dependencies.

```
julia BlindDecon.jl selftest                    # 33 deterministic checks, ~13 s (incl. JIT)
julia BlindDecon.jl demo_seismic [outdir]       # 1-D seismic blind decon demo (SVG)
julia BlindDecon.jl demo_image [bmp] [outdir]   # 2-D image demo, or your own 24-bit BMP
```

---

## The physics / the math behind it

Deconvolution tries to undo a convolution `y = h * x + n`. When the wavelet /
point-spread function `h` is **unknown**, the problem is *blind*: both `h`
and `x` must be estimated from `y` alone. That is ill-posed: `(h, x)` and
`(h', x')` with `h * x = h' * x'` are indistinguishable, so every practical
method breaks the tie with statistical or structural assumptions about `x`
(sparse, white, non-negative, piecewise smooth, natural-image gradients with
heavy tails). The toolkit implements the classical *and* modern
variational state of the art, with the assumptions written out.

## 1-D: signal / seismic

Model: `y[n] = sum_k h[k] x[n-k+1] + noise` (causal, FIR wavelet).

| function | method | assumption that makes blind estimation possible |
|---|---|---|
| `prediction_decon` | gapped Wiener prediction-error (spiking) decon | reflectivity white; wavelet minimum-phase (stationary AR theory) |
| `minphase_from_amp` | Kolmogoroff minimum-phase wavelet from an amplitude spectrum | minimum-phase (Hilbert/log-amplitude relation) |
| `wavelet_from_ensemble` | robust `|H|` estimation by averaging power spectra over traces with a common wavelet | reflectivities white and independent between traces |
| `homomorphic_wavelet` | cepstral (lifter) wavelet estimation | wavelet spectrum smooth, reflectivity spectrum spiky in log domain |
| `med_decon` | Wiggins minimum-entropy deconvolution (varimax/kurtosis maximization, backtracked steepest ascent) | reflectivity sparse (high kurtosis); no Gaussianity, no stationarity needed |
| `sparse_bd1` | alternating minimization: ISTA soft-threshold reflectivity step + ridge-LS wavelet step | reflectivity sparse (l1); wavelet FIR |
| `crossrel_identify` | SIMO cross-relation blind channel identification (smallest-eigenvector of the stacked cross-relation LS) | same input through >= 2 different channels (channel diversity) |
| `wiener_decon1` | non-blind Wiener baseline (known wavelet) | - |

Wavelet generators: `ricker`, `ormsby`, `damped_cosine_wavelet` (a damped
cosine is a *genuinely* minimum-phase broadband wavelet - unlike a Ricker,
which has a spectral null at DC and an infinitely long causal inverse).

## 2-D: images

Model: circular `y = k (x) x + noise`, kernel embedded center-at-origin;
demos pad with replicated borders so the wrap only ever reads replicated
pixels. All x-steps exploit FFT diagonalization; the hyper-Laplacian solver
uses per-pixel IRLS weights + conjugate gradients (the weights make the
system *not* Fourier-diagonal - this is exactly why Krishnan-Fergus solve
the x-subproblem with CG).

| function | method |
|---|---|
| `decon_wiener2` | non-blind Wiener (known kernel) |
| `decon_rl2` | non-blind Richardson-Lucy (Poisson/multiplicative, non-negativity built in) |
| `decon_hq2` | **hyper-Laplacian prior** `lam*sum(|grad x|^alpha)` via IRLS weights `w_i = (|grad_i|^2 + eps^2)^((alpha-2)/2)` + CG. `alpha = 1` is total variation (Chan-Wong), `alpha = 0.5 / 0.8` are the heavy-tailed natural-image statistics of Krishnan & Fergus (2009) |
| `ibd_rl2` | **blind**: Ayers-Dainty iterative blind deconvolution, alternating RL updates for image and kernel, positivity/support/sum-one kernel projections |
| `am_tv2` | **blind**: Chan-Wong alternating minimization (TV image step + constrained LS kernel step, band-limited kernel estimate) |
| `xujia2` | **blind**: Xu-Jia two-stage fast kernel estimation - Osher-Rudin **shock filter** restores sharp salient edges, gradient-domain diagonal-FFT kernel LS, final hyper-Laplacian restoration |
| `shock2` | Osher-Rudin shock filter `u_t = -sign(lap u) |grad u|` (edge sharpening morphological PDE) |
| `kernel_ls2` | constrained LS kernel estimate with radial spectral low-pass (true small-support kernels are band-limited; the mask kills noise-amplified high bands) |
| `prox_lapower` | proximal operator of `tau*|w|^alpha` for any `alpha < 1`, scalar bisection with exact shrinkage threshold |

Kernels: `gaussian_kernel2`, `motion_kernel2`, `disk_kernel2`.
Metrics: `psnr`, `kernel_corr` (center-aligned normalized correlation).

## Selftest (all seeds fixed, PASS lines byte-identical across runs)

```
PASS check  1: fft1 (Bluestein, n=100) vs direct DFT (max err 8.39e-13)
...
PASS check 18: cross-relation blind ID (noiseless) (rel err 7.85e-15 / 1.06e-14)
PASS check 24: decon_hq2 (alpha=0.8) PSNR gain (gain 10.58 dB)
PASS check 25: ibd_rl2 kernel recovery (kcorr 0.983)
PASS check 27: am_tv2 kernel recovery (kcorr 0.985)
PASS check 29: xujia2 kernel recovery (kcorr 0.975)
33 checks, 0 failed, ~1.3 s (plus JIT)
```

Highlights: FFT cross-checked against direct DFTs (pow2 and Bluestein),
`minphase_from_amp` preserves `|H|` to 4.5e-7, spiking decon whitens to
lag-1 autocorr 0.000 and recovers white reflectivity at corr 0.973, MED is
monotone in the varimax and recovers all spike positions exactly, the
sparse blind decon fits the wavelet at corr 1.000, the prox operator is
validated against brute-force grid search, IBD/TV-AM/Xu-Jia recover kernels
at kcorr 0.98/0.99/0.98.

## Demo results

`demo_seismic` (500 samples, 30 Hz damped-cosine minimum-phase wavelet,
30 dB noise): prediction decon corr 0.994, MED 0.996, sparse alternating
1.000, ensemble wavelet estimation 0.996, homomorphic amplitude 0.988.

`demo_image` (96x96 synthetic, Gaussian blur sigma = 1.6, 40 dB noise):

```
method                     PSNR     gain      kcorr     time
blurred (input)           23.17     0.00          -        -
IBD-RL (Ayers-Dainty)     24.32     1.15      0.898     0.29
TV-AM (Chan-Wong)         25.63     2.46      0.978     2.03
Xu-Jia + hyper-Lapl.      25.31     2.14      0.838     0.86
```

Both demos write hand-authored SVG figures (traces, wavelet estimates,
kernel gallery) and BMP images - no plotting package involved.

## Honest limits

- **Trivial fixed point**: initializing alternating blind schemes with a
  delta kernel collapses (x-step is a no-op, kernel refits delta). The
  toolkit initializes with a small Gaussian instead; `kinit` can override.
- **Band limitation is information-theoretic**: spiking decon cannot
  recover reflectivity above the wavelet band (a Ricker wavelet has a
  spectral null at DC; use broadband wavelets for decon tests).
- **Finite-rho half-quadratic bias**: the additive HQ splitting with fixed
  coupling weight does NOT converge to the true prior minimizer (the
  x-subproblem must carry per-pixel weights). This toolkit therefore uses
  IRLS + CG, not diagonal solves, for the hyper-Laplacian x-step.
- Kernel LS without the radial spectral low-pass breaks down at 30 dB
  noise (kcorr 0.46 -> 0.99 with the mask).
- IBD converges slowly and can over-sharpen on noise-free data; RL needs
  hundreds of iterations for sigma >= 1.5 blurs.
- No learned priors (deep-prior/diffusion-based blind decon is out of
  scope for a stdlib-only toolkit), no GPU, no super-resolution claims.
- Blind kernel estimation assumes a single dominant uniform blur; no
  spatially varying kernels.

## Files

```
BlindDecon.jl            the whole toolkit (~1900 lines)
examples/demo_seismic.jl wrapper
examples/demo_image.jl   wrapper (pass your own 24-bit BMP)
```

Author: Dennis Kunde (St0n89), Tarvos. License: MIT.
