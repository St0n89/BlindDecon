# BlindDecon.jl

A **blind deconvolution toolkit** for 1-D signals (seismic/sparse spike) and
2-D images, written in **100% Julia standard library**. No external packages:
hand-written FFT (radix-2 Cooley-Tukey with cached twiddle/bit-reversal
tables + Bluestein for arbitrary lengths), hand-written 24-bit BMP
reader/writer, hand-written SVG figures, all solvers built from scratch.
Julia >= 1.9, single file, zero dependencies.

```
julia BlindDecon.jl selftest                    # 51 deterministic checks, ~5 s (plus JIT)
julia BlindDecon.jl demo_seismic [outdir]       # 1-D seismic blind decon demo (SVG)
julia BlindDecon.jl demo_image [bmp] [outdir]   # 2-D image demo, or your own 24-bit BMP
julia BlindDecon.jl demo_mars [outdir]          # real InSight SEIS marsquake data (SVG)
```

---

## The physics / the math behind it

Deconvolution tries to undo a convolution `y = h * x + n`. When the wavelet /
point-spread function `h` is **unknown**, the problem is *blind*: both `h`
and `x` must be estimated from `y` alone. That is ill-posed: `(h, x)` and
`(h', x')` with `h * x = h' * x'` are indistinguishable, so every practical
method breaks the tie with statistical or structural assumptions about `x`
(sparse, white, non-negative, piecewise smooth, natural-image gradients with
heavy tails). The toolkit implements the classical *and* modern variational
state of the art, with the assumptions written out.

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
| `crossrel_identify_multi` | **v0.2.0**: multichannel cross-relation, all channel pairs, N > 2 channels | channel diversity; pairwise cross-correlations `y_i * y_j = h_i * h_j` |
| `mad_noise1` | **v0.2.0**: robust noise level via median absolute deviation | noise median-unbiased, signal sparse/banded |
| `q_estimate` | **v0.2.0**: constant-Q estimate by pooled STFT log-amplitude regression on `f*t` | dense (white) reflectivity; constant Q |
| `nonstat_decon` | **v0.2.0**: non-stationary short-time spectral whitening (hann windows, temporally smoothed envelope floor) | nothing stationary; only local whiteness |
| `q_attenuated_trace` | **v0.2.0**: constant-Q synthetic generator, STFT-domain `exp(-pi f t/Q)` | - |
| `parse_geocsv`, `bandpass1`, `specflat1` | **v0.2.0**: tolerant PDS GeoCSV reader, zero-padded FFT bandpass, spectral flatness | - |
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
| `decon_hq2` | **hyper-Laplacian prior** `lam*sum(|grad x|^alpha)` via IRLS weights `w_i = (|grad_i|^2 + eps^2)^((alpha-2)/2)` + CG. `alpha = 1` is total variation (Chan-Wong), `alpha = 0.5 / 0.8` are the heavy-tailed natural-image statistics of Krishnan & Fergus (2009). `robust = true` (**v0.2.0**) adds IRLS Huber weights on the data residual (delta auto-set from `mad_noise2`) for salt/outlier-corrupted data |
| `ibd_rl2` | **blind**: Ayers-Dainty iterative blind deconvolution, alternating RL updates for image and kernel, positivity/support/sum-one kernel projections |
| `am_tv2` | **blind**: Chan-Wong alternating minimization (TV image step + constrained LS kernel step, band-limited kernel estimate) |
| `xujia2` | **blind**: Xu-Jia two-stage fast kernel estimation - Osher-Rudin **shock filter** restores sharp salient edges, gradient-domain diagonal-FFT kernel LS, final hyper-Laplacian restoration. `edge_frac` (**v0.2.0**) replaces the FFT pixel mask with exact spatial-domain weighted kernel LS (edge selection by gradient magnitude, per-pixel weights) |
| `blind_pyramid2` | **v0.2.0**: **blind coarse-to-fine pyramid** - preliminary warp-free pyramid down-sampling with kernel transport, alternating BD at each level, kernel broadcast to the next finer level. Fixes the single-scale collapse on large blurs |
| `decon_hq2_multi` | **v0.2.0**: multichannel joint restore - one shared image, differently blurred channels, shared IRLS prior + channel-summed FFT data term |
| `multiframe_bd2` | **v0.2.0**: multiframe blind decon - one kernel per frame, joint image across frames, energy-guarded kernel refit per step |
| `restart_search2` | **v0.2.0**: kernel-space restart search - re-runs blind BD from a pool of init kernel scales, returns the full candidate POOL (joint energy ranking is NOT trustworthy: the delta-near candidate often wins the energy but loses MSE, so selection is left to the caller) |
| `mad_noise2` | **v0.2.0**: robust image noise level via median absolute deviation |
| `shock2` | Osher-Rudin shock filter `u_t = -sign(lap u) |grad u|` (edge sharpening morphological PDE) |
| `kernel_ls2` | constrained LS kernel estimate with radial spectral low-pass (true small-support kernels are band-limited; the mask kills noise-amplified high bands) |
| `prox_lapower` | proximal operator of `tau*|w|^alpha` for any `alpha < 1`, scalar bisection with exact shrinkage threshold |

Kernels: `gaussian_kernel2`, `motion_kernel2`, `disk_kernel2`.
Metrics: `psnr`, `kernel_corr` (center-aligned normalized correlation).

## v0.2.0 performance core

Cached twiddle factor and bit-reversal tables per FFT size, half-spectrum
twiddle generation, in-place radix-2 staging, and an axis-buffered 2-D fast
path (no matrix transposes). Measured on the same machine in one session:
`fft2_` on a 128x128 pow2 grid runs in 0.30 ms vs 1.03 ms in v0.1.0
(**3.5x faster**); on 256x256 it is 1.22 ms vs 2.26 ms (**1.8x**). All FFT
results re-verified against direct DFTs (checks 39-40).

## Selftest (all seeds fixed, PASS lines byte-identical across runs)

```
PASS check  1: fft1 (Bluestein, n=100) vs direct DFT (max err 8.39e-13)
...
PASS check 18: cross-relation blind ID (noiseless) (rel err 7.85e-15 / 1.06e-14)
PASS check 24: decon_hq2 (alpha=0.8) PSNR gain (gain 10.58 dB)
PASS check 25: ibd_rl2 kernel recovery (kcorr 0.983)
PASS check 27: am_tv2 kernel recovery (kcorr 0.985)
PASS check 29: xujia2 kernel recovery (kcorr 0.975)
PASS check 34: crossrel_identify_multi 3-channel kernel recovery (max rel err 5.88e-12)
PASS check 37: q_estimate recovers Q (dense reflectivity, pooled fit) (Qhat 93.5, true 80; ~12% synthesis-blend bias, documented)
PASS check 38: nonstat_decon whitens the attenuated late half better than PEF (late lag1: in 0.984, nonstat 0.446, PEF 0.817)
PASS check 43: robust decon_hq2 survives salt outliers (IRLS Huber) (PSNR gain vs plain 31.4 dB)
PASS check 45: blind_pyramid2 beats single-scale on large blur (kcorr pyr 0.997 vs single 0.470, PSNR gain 1.59 dB)
PASS check 46: decon_hq2_multi gains SNR from redundant channels (3-channel gain vs single 1.39 dB)
PASS check 47: multiframe_bd2 estimates kernels better than single-frame (gaussian-frame kcorr multiframe 0.880 vs single-frame 0.671)
PASS check 48: restart_search2 pool contains the true-scale basin (best pool kcorr 1.000 across 4 init scales)
PASS check 49: xujia2 edge_frac improves kernel LS at moderate noise (kcorr delta 0.077)
PASS check 50: parse_geocsv reads the InSight SEIS S0105a bundle (48000 samples, 20 sps)
PASS check 51: PEF whitens the real S0105a P window (lag1 1.000 -> 0.022, specflat 0.0033 -> 0.9933)
51 checks, 0 failed, ~5 s (plus JIT)
```

Highlights: FFT cross-checked against direct DFTs (pow2, Bluestein, 2-D),
`minphase_from_amp` preserves `|H|` to 4.5e-7, spiking decon whitens to
lag-1 autocorr 0.000, MED recovers all spike positions exactly, the prox
operator is validated against brute-force grid search, IBD/TV-AM/Xu-Jia
recover kernels at kcorr 0.98/0.99/0.98, the 3-channel cross-relation edges
out the 2-channel one, MAD recovers planted noise levels to within 10%
(0.0551 vs 0.05, 0.0101 vs 0.01), the robust Huber variant gains 31.4 dB on
salt outliers, the pyramid rescues the single-scale collapse on a sigma-4
blur (0.997 vs 0.470), and PEF whitening of real Mars data is confirmed by
spectral flatness jumping 0.0033 -> 0.9933.

## Demo results

`demo_seismic` (500 samples, 30 Hz damped-cosine minimum-phase wavelet,
30 dB noise): prediction decon corr 0.994, MED 0.996, sparse alternating
1.000, ensemble wavelet estimation 0.996, homomorphic amplitude 0.988. The
v0.2.0 block adds a constant-Q attenuated trace (`Q = 80`): pooled STFT fit
estimates `Qhat ~ 93` (synthesis-blend bias, see limits), and `nonstat_decon`
whitens the attenuated late half to lag-1 0.446 where stationary PEF stalls
at 0.817.

`demo_image` (96x96 synthetic, Gaussian blur sigma = 1.6, 40 dB noise):

```
method                     PSNR     gain      kcorr     time
blurred (input)           23.17     0.00          -        -
IBD-RL (Ayers-Dainty)     24.32     1.15      0.898     0.29
TV-AM (Chan-Wong)         25.63     2.46      0.978     2.03
Xu-Jia + hyper-Lapl.      25.31     2.14      0.838     0.86
```

The v0.2.0 blocks add: `blind_pyramid2` on a 128x128 image blurred with
sigma 4, radius 12 (40 dB): pyramid kcorr **0.997** where single-scale
collapses to 0.470, plus an extra +1.59 dB PSNR. And `multiframe_bd2` on
three frames (gaussian sigma 1.6 + two motion kernels, 30 dB): kernel
correlations 0.940 / 0.616 where the gaussian-frame-only single-frame run
reaches 0.584.

`demo_mars`: real seismic data. InSight SEIS, event S0105a (sol 105, the
first detected marsquake), at 20 sps from the Mars Seismic
Catalogue v14, read directly from the bundled PDS GeoCSV file (no network
needed). The bundle is a 2400 s crop of the PDS hour (48000 samples):
it contains the event, its full coda, and the lander-activity transient
at t = 1997 s; the full hour remains available from the PDS archive. The demo bandpasses around the P pick, whitens the P window with
PEF (lag-1 autocorr 1.000 -> 0.022, spectral flatness 0.0033 -> 0.9933),
estimates a 128-tap minimum-phase wavelet from the whitened window via the
Kolmogoroff relation, and writes a 5-panel SVG. It is honest about the
noise: the strongest broadband transient in the file (t = 1997 s, well
after the event window) is 3x stronger than the P coda and is almost
certainly lander activity (HP3 mole / robotic arm), not a marsquake.

Data credit: NASA InSight SEIS, PDS Geosciences Node (Washington
University in St. Louis), Mars Seismic Catalogue v14. The bundle file is
included at `data/xb.elyse.02.bhu.S0105a.mqs2019fddj.1.a.csv`.

Both other demos write hand-authored SVG figures and BMP images - no
plotting package involved.

## Honest limits

- **Trivial fixed point**: initializing alternating blind schemes with a
  delta kernel collapses (x-step is a no-op, kernel refits delta). The
  toolkit initializes with a small Gaussian instead; `kinit` can override.
- **Band limitation is information-theoretic**: spiking decon cannot
  recover reflectivity above the wavelet band (a Ricker wavelet has a
  spectral null at DC; use broadband wavelets for decon tests). The same
  trap hides in metrics: lag-1 autocorrelation of smooth band-limited data
  is close to 1 *by construction* and says nothing about decon quality on
  such data.
- **Finite-rho half-quadratic bias**: the additive HQ splitting with fixed
  coupling weight does NOT converge to the true prior minimizer (the
  x-subproblem must carry per-pixel weights). This toolkit therefore uses
  IRLS + CG, not diagonal solves, for the hyper-Laplacian x-step.
- **q_estimate**: the pooled STFT fit needs *dense* (white) reflectivity;
  on sparse-spike traces the log-amplitude regression latches onto event
  envelopes and fails. On dense reflectivity the fit has a systematic
  ~10-15% upward bias from STFT synthesis blending (the synthetic was
  built with the same STFT path, so the measured Qhat 93.5 vs true 80 is
  a blend artifact, not a fit error). Both are documented in the code.
- The single-scale blind schemes (ibd/am/xujia through `blind_pyramid2`)
  collapse on large blurs relative to image size; the pyramid fixes the
  collapse but has its own failure mode: for very small images and extreme
  blur it can alias the coarsest pyramid level into a wrong basin.
- **Multiframe**: joint multiframe BD improves *kernel* quality clearly
  (0.880 vs 0.671 in check 47, 3 real kernels vs 1) but does NOT
  guarantee better PSNR per frame - comparing a wrong-kernel single-frame
  restore (which fails at its own self-limited level) with a
  correct-kernel multiframe restore (which deconvolves harder and pays
  for it) is apples-to-oranges. The demos report the kernel advantage,
  which is the well-posed comparison.
- **restart_search2** returns a candidate pool, not a winner: ranking
  candidates by joint energy provably fails (the near-delta candidate
  often wins the energy but loses the MSE by a wide margin). Callers
  must pick a candidate by their own criterion.
- **xujia2 edge_frac**: exact spatial-domain weighted kernel LS beats the
  FFT-pixel-mask variant (which measurably hurts, documented in the code).
- **demo_mars honesty**: the event-band SNR is modest (rms 315 vs MAD 28
  background); the strongest transient in the bundle is likely lander
  activity, not seismic. The demo prints this rather than hiding it.
- Kernel LS without the radial spectral low-pass breaks down at 30 dB
  noise (kcorr 0.46 -> 0.99 with the mask).
- IBD converges slowly and can over-sharpen on noise-free data; RL needs
  hundreds of iterations for sigma >= 1.5 blurs.
- No learned priors (deep-prior/diffusion-based blind decon is out of
  scope for a stdlib-only toolkit), no GPU, no super-resolution claims.
- Blind kernel estimation assumes a single dominant uniform blur per
  frame; no spatially varying kernels.

## Files

```
BlindDecon.jl            the whole toolkit (~3200 lines)
examples/demo_seismic.jl wrapper
examples/demo_image.jl   wrapper (pass your own 24-bit BMP)
data/xb.elyse.02.bhu.S0105a.mqs2019fddj.1.a.csv  InSight SEIS S0105a bundle (PDS GeoCSV, 1.8 MB)
```

Author: Dennis Kunde (St0n89), Tarvos. License: MIT.
