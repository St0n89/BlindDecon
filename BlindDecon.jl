#!/usr/bin/env julia
# =============================================================================
#  BlindDecon.jl - Blind Deconvolution Toolkit (1-D signal/seismic + 2-D image)
#
#  Pure Julia, standard library only (LinearAlgebra, Printf, Random).
#  No external packages: hand-written FFT (radix-2 + Bluestein), hand-written
#  BMP/SVG writers, all solvers built from scratch.
#
#  1-D (signal / seismic):
#    - prediction_decon       gapped Wiener prediction-error (spiking) decon
#    - minphase_from_amp      Kolmogorov minimum-phase wavelet from amplitude
#    - wavelet_from_ensemble  robust |H| estimation from many traces
#    - homomorphic_wavelet    cepstral (lifter) wavelet estimation
#    - med_decon              Wiggins minimum-entropy (kurtosis) decon
#    - sparse_bd1             alternating sparse/LS blind spike decon
#    - crossrel_identify      SIMO cross-relation blind channel identification
#    - wiener_decon1          non-blind Wiener baseline (known wavelet)
#    - crossrel_identify_multi (v0.2.0) multichannel cross-relation, N>2 channels
#    - mad_noise1            (v0.2.0) robust MAD noise level, 1-D
#    - q_estimate            (v0.2.0) constant-Q estimate, pooled STFT log-amp fit
#    - nonstat_decon          (v0.2.0) non-stationary short-time spectral whitening
#    - q_attenuated_trace     (v0.2.0) constant-Q synthetic generator
#    - parse_geocsv / bandpass1 / _specflat1 (v0.2.0) real-data helpers (GeoCSV)
#
#  2-D (image):
#    - decon_wiener2          non-blind Wiener (known kernel)
#    - decon_rl2              non-blind Richardson-Lucy
#    - decon_hq2              half-quadratic solver, hyper-Laplacian priors
#                             (alpha=1 -> TV, alpha=0.5/0.8 -> heavy-tailed);
#                             robust=true adds IRLS Huber weights on the data
#                             residual (v0.2.0, salt/outlier resistance)
#    - ibd_rl2                blind Ayers-Dainty iterative blind decon (RL)
#    - am_tv2                 blind Chan-Wong alternating TV minimization
#    - xujia2                 blind Xu-Jia: shock filter + kernel LS +
#                             hyper-Laplacian restoration; edge_frac =
#                             exact weighted spatial kernel LS (v0.2.0)
#    - shock2                 Osher-Rudin shock filter
#    - mad_noise2             (v0.2.0) robust MAD noise level, images
#    - blind_pyramid2         (v0.2.0) coarse-to-fine gaussian pyramid BD
#    - decon_hq2_multi        (v0.2.0) multichannel joint restore, shared image
#    - multiframe_bd2         (v0.2.0) multiframe BD, one kernel per frame
#    - restart_search2        (v0.2.0) kernel-space init-scale restart pool
#
#  v0.2.0 performance core: cached twiddle/bit-reversal tables, in-place
#  radix-2 stages, axis-buffered 2-D transforms (measured 3.5x faster
#  fft2_ at 128x128 and 1.8x at 256x256 than v0.1.0, same session).
#
#  CLI:
#    julia BlindDecon.jl selftest
#    julia BlindDecon.jl demo_seismic [outdir]
#    julia BlindDecon.jl demo_image   [bmp_in] [outdir]
#
#  Julia >= 1.9.  Author: Dennis Kunde (St0n89), Tarvos.  License: MIT.
# =============================================================================

using LinearAlgebra
using Printf
using Random

# -----------------------------------------------------------------------------
# Section 0: small utilities
# -----------------------------------------------------------------------------

function ispow2_(n::Int)
    return n > 0 && (n & (n - 1)) == 0
end

function nextpow2_(n::Int)
    m = 1
    while m < n
        m <<= 1
    end
    return m
end

function _zeropad(v::Vector{Float64}, n::Int)
    out = zeros(Float64, n)
    out[1:length(v)] .= v
    return out
end

function _mean(v::AbstractVector{<:Real})
    return sum(v) / length(v)
end

function _std(v::AbstractVector{<:Real})
    m = _mean(v)
    return sqrt(sum((v .- m) .^ 2) / length(v))
end

function _clamp01!(img::Matrix{Float64})
    @inbounds for i in eachindex(img)
        img[i] = clamp(img[i], 0.0, 1.0)
    end
    return img
end

# -----------------------------------------------------------------------------
# Section 1: FFT (radix-2 Cooley-Tukey + Bluestein for arbitrary length)
# -----------------------------------------------------------------------------

# --- v0.2.0 performance core -----------------------------------------------
# cached twiddle factors and bit-reversal permutations per transform size,
# in-place radix-2 stages, axis-buffered 2-D transforms (no transposes;
# @threads measured SLOWER on a single-socket runner, deliberately not
# used). Twiddles are exact per-index cis() values, so the pow2 path has
# FEWER rounding steps than the repeated-multiplication form above was.

const _FFT_TW_CACHE = Dict{Tuple{Int,Bool},Vector{ComplexF64}}()
const _FFT_BR_CACHE = Dict{Int,Vector{Int}}()

function _fft_half_tw(len::Int, inverse::Bool)
    key = (len, inverse)
    tw = get(_FFT_TW_CACHE, key, nothing)
    if tw === nothing
        half = len >> 1
        ang = (inverse ? 2.0 : -2.0) * pi / len
        tw = [cis(ang * (k - 1)) for k in 1:half]
        _FFT_TW_CACHE[key] = tw
    end
    return tw
end

function _fft_bitrev(n::Int)
    br = get(_FFT_BR_CACHE, n, nothing)
    if br === nothing
        br = Vector{Int}(undef, n)
        j = 1
        @inbounds for i in 1:n
            br[i] = j
            m = n >> 1
            while m >= 1 && j > m
                j -= m
                m >>= 1
            end
            j += m
        end
        _FFT_BR_CACHE[n] = br
    end
    return br
end

# in-place radix-2 Cooley-Tukey on the caller's buffer
function _fft_pow2_stages!(b::AbstractVector{ComplexF64}, inverse::Bool)
    n = length(b)
    n == 1 && return b
    br = _fft_bitrev(n)
    @inbounds for i in 1:n
        j = br[i]
        if i < j
            b[i], b[j] = b[j], b[i]
        end
    end
    len = 2
    @inbounds while len <= n
        half = len >> 1
        tw = _fft_half_tw(len, inverse)
        i = 1
        while i <= n
            for k in 1:half
                w = tw[k]
                u = b[i + k - 1]
                v = b[i + k + half - 1] * w
                b[i + k - 1] = u + v
                b[i + k + half - 1] = u - v
            end
            i += len
        end
        len <<= 1
    end
    if inverse
        @inbounds for i in 1:n
            b[i] /= n
        end
    end
    return b
end

function _fft_pow2(a::Vector{ComplexF64}, inverse::Bool)
    return _fft_pow2_stages!(ComplexF64.(a), inverse)
end
# --- end v0.2.0 performance core --------------------------------------------

# Bluestein chirp-z transform for arbitrary n (forward transform only;
# inverse is done by the conjugation trick in ifft1).
function _fft_bluestein(a::Vector{ComplexF64})
    n = length(a)
    n <= 1 && return copy(a)
    if ispow2_(n)
        return _fft_pow2(a, false)
    end
    m = nextpow2_(2 * n - 1)
    # chirp(k) = exp(-i*pi*k^2/n); use integer mod to keep the angle exact
    chirp(k) = cis(-pi * ((k * k) % (2 * n)) / n)
    aa = zeros(ComplexF64, m)
    bb = zeros(ComplexF64, m)
    for k in 0:n-1
        aa[k+1] = a[k+1] * chirp(k)          # a[n] = x[n]*c[n]
        bb[k+1] = conj(chirp(k))             # b[m] = conj(c[m])
        if k > 0
            bb[m-k+1] = conj(chirp(k))       # negative lags wrap to the end
        end
    end
    fa = _fft_pow2(aa, false)
    fb = _fft_pow2(bb, false)
    fa .*= fb
    cc = _fft_pow2(fa, true)
    out = Vector{ComplexF64}(undef, n)
    for k in 0:n-1
        out[k+1] = chirp(k) * cc[k+1]
    end
    return out
end

function fft1(a::AbstractVector{<:Number})
    c = ComplexF64.(a)
    if ispow2_(length(c))
        return _fft_pow2_stages!(c, false)
    end
    return _fft_bluestein(c)
end

function ifft1(a::AbstractVector{<:Number})
    c = ComplexF64.(a)
    n = length(c)
    return conj.(fft1(conj.(c))) ./ n
end

function fft2_(A::AbstractMatrix{<:Number})
    m, n = size(A)
    B = Matrix{ComplexF64}(A)
    if ispow2_(m) && ispow2_(n) && m > 1 && n > 1
        # fast path: both dimensions pow2, one reusable axis buffer,
        # row-major rows then strided columns (no transposes)
        buf = Vector{ComplexF64}(undef, max(m, n))
        @inbounds for i in 1:m
            for j in 1:n
                buf[j] = B[i, j]
            end
            _fft_pow2_stages!(buf, false)
            for j in 1:n
                B[i, j] = buf[j]
            end
        end
        @inbounds for j in 1:n
            for i in 1:m
                buf[i] = B[i, j]
            end
            _fft_pow2_stages!(buf, false)
            for i in 1:m
                B[i, j] = buf[i]
            end
        end
        return B
    end
    for i in 1:m
        B[i, :] .= fft1(B[i, :])
    end
    for j in 1:n
        B[:, j] .= fft1(B[:, j])
    end
    return B
end

function ifft2_(A::AbstractMatrix{<:Number})
    m, n = size(A)
    return conj.(fft2_(conj.(ComplexF64.(A)))) ./ (m * n)
end

# -----------------------------------------------------------------------------
# Section 2: 1-D convolution / correlation primitives
#
# Signal model (causal, consistent everywhere):
#   y[n] = sum_k h[k] * x[n-k+1],  n = 1..N,  x zero outside 1..N
# i.e. y is the FIRST N samples of the full convolution of x and h.
# -----------------------------------------------------------------------------

# full linear convolution, FFT-based
function conv_full(x::Vector{Float64}, h::Vector{Float64})
    lx = length(x)
    lh = length(h)
    nout = lx + lh - 1
    nfft = nextpow2_(max(2, nout))
    xf = fft1(_zeropad(x, nfft))
    hf = fft1(_zeropad(h, nfft))
    y = real(ifft1(xf .* hf))
    return y[1:nout]
end

# full linear convolution, direct (reference implementation, O(N*L))
function conv_full_direct(x::Vector{Float64}, h::Vector{Float64})
    lx = length(x)
    lh = length(h)
    y = zeros(Float64, lx + lh - 1)
    @inbounds for j in 1:lx
        xj = x[j]
        for k in 1:lh
            y[j+k-1] += xj * h[k]
        end
    end
    return y
end

# causal convolution: out[n] = sum_k h[k]*x[n-k+1], length == length(x)
function conv_causal(x::Vector{Float64}, h::Vector{Float64})
    N = length(x)
    L = length(h)
    y = zeros(Float64, N)
    @inbounds for k in 1:L
        hk = h[k]
        for n in k:N
            y[n] += hk * x[n-k+1]
        end
    end
    return y
end

# sum_{i} u[i]*v[i+lag] for any integer lag (out-of-range -> skipped)
function xcorr_any(u::Vector{Float64}, v::Vector{Float64}, lag::Int)
    s = 0.0
    nu = length(u)
    nv = length(v)
    @inbounds for i in 1:nu
        j = i + lag
        if 1 <= j <= nv
            s += u[i] * v[j]
        end
    end
    return s
end

# unnormalized autocorrelation, r[tau+1] = sum_i x[i]*x[i+tau], tau = 0..maxlag
function autocorr_full(x::Vector{Float64}, maxlag::Int)
    r = zeros(Float64, maxlag + 1)
    for tau in 0:maxlag
        r[tau+1] = xcorr_any(x, x, tau)
    end
    return r
end

# Solve symmetric Toeplitz system T b = rhs with T[i,j] = r[|i-j|+1].
# Explicit matrix + LU (n is small: filter/kernel lengths).
function toeplitz_solve(r::Vector{Float64}, rhs::Vector{Float64})
    n = length(rhs)
    T = zeros(Float64, n, n)
    @inbounds for i in 1:n, j in 1:n
        T[i, j] = r[abs(i - j) + 1]
    end
    return T \ rhs
end

function _center(v::Vector{Float64}, n::Int)
    start = div(length(v) - n, 2) + 1
    return v[start:start+n-1]
end

# central part of the full convolution (same-length semantics)
function conv_same(x::Vector{Float64}, h::Vector{Float64})
    return _center(conv_full(x, h), length(x))
end

# -----------------------------------------------------------------------------
# Section 3: 1-D wavelets (seismic) + Kolmogorov minimum phase
# -----------------------------------------------------------------------------

function ricker(fc::Real, dt::Real, n::Int)
    t = [(i - div(n - 1, 2) - 1) * dt for i in 1:n]
    a = (pi * fc * dt) ^ 2
    return [(1.0 - 2.0 * a * (i - div(n - 1, 2) - 1) ^ 2) *
            exp(-a * (i - div(n - 1, 2) - 1) ^ 2) for i in 1:n]
end

# Causal broadband minimum-phase wavelet (damped cosine). This is the right
# test wavelet for spiking decon: unlike a Ricker wavelet it has no spectral
# nulls at DC/Nyquist, so its causal inverse is short and well-conditioned.
function damped_cosine_wavelet(f0::Real, decay::Real, dt::Real, n::Int)
    h = [exp(-decay * (i - 1) * dt) * cos(2 * pi * f0 * (i - 1) * dt) for i in 1:n]
    return h ./ h[1]
end

# Ormsby wavelet (trapezoid spectrum), corner frequencies f1<f2<f3<f4 (Hz).
# Built directly in the frequency domain (zero-phase), then centered.
function ormsby(f1, f2, f3, f4, dt::Real, n::Int)
    nfft = nextpow2_(n)
    df = 1.0 / (nfft * dt)
    A = zeros(Float64, nfft)
    nhalf = div(nfft, 2)
    for k in 0:nhalf
        f = k * df
        a = f <= f1 ? 0.0 :
            f < f2 ? (f - f1) / (f2 - f1) :
            f <= f3 ? 1.0 :
            f < f4 ? (f4 - f) / (f4 - f3) : 0.0
        A[k+1] = a
        if k > 0 && k <= nhalf - 1
            A[nfft-k+1] = a
        end
    end
    return zerophase_from_amp(A, n)
end

# Kolmogorov / Hilbert relation: minimum-phase impulse response with a
# prescribed amplitude spectrum.  amp = |H| over ALL FFT bins of an even
# real spectrum (length N, N a power of 2 recommended).
function minphase_from_amp(amp::Vector{Float64})
    N = length(amp)
    A = log.(max.(amp, 1e-15))
    # force exact even symmetry (real signal): A[N-i+2] = A[i]
    @inbounds for i in 2:div(N, 2)
        A[N-i+2] = A[i]
    end
    c = real(ifft1(A))
    ch = zeros(Float64, N)
    ch[1] = c[1]
    nhalf = div(N, 2)
    for i in 2:nhalf
        ch[i] = 2.0 * c[i]
    end
    ch[nhalf+1] = c[nhalf+1]
    Hmin = exp.(fft1(ch))
    h = real(ifft1(Hmin))
    return h
end

# zero-phase (symmetric) wavelet from an amplitude spectrum, truncated to
# nwav samples, centered
function zerophase_from_amp(amp::Vector{Float64}, nwav::Int)
    h = real(ifft1(amp))
    return _center(h, nwav)
end

# Gapped Wiener prediction-error (spiking) deconvolution.
# gap = 1 -> spiking decon; larger gaps preserve structure slower than gap.
# Returns (deconvolved output, PEF filter a with a[1] = 1).
function prediction_decon(y::Vector{Float64}; gap::Int = 1, nfil::Int = 40)
    N = length(y)
    nfil = min(nfil, max(4, N >> 2))
    r = autocorr_full(y, gap + nfil - 1)
    rhs = r[gap+1:gap+nfil]
    f = toeplitz_solve(r[1:nfil], rhs)
    a = vcat(1.0, zeros(Float64, gap - 1), -f)
    out = conv_causal(y, a)
    return out, a
end

# Wavelet estimation from an ensemble of traces with the SAME wavelet and
# independent white-ish reflectivities: average power spectrum ~ |H|^2 * const.
# phase = :min (Kolmogorov) or :zero (symmetric).
function wavelet_from_ensemble(ys::Vector{Vector{Float64}}, nwav::Int;
                               phase::Symbol = :min, nfft::Int = 0)
    N = maximum(length, ys)
    nfft = nfft > 0 ? nextpow2_(nfft) : nextpow2_(N + nwav)
    P = zeros(Float64, nfft)
    for y in ys
        Y = fft1(_zeropad(y, nfft))
        P .+= abs2.(Y)
    end
    P ./= length(ys)
    amp = sqrt.(P)
    if phase == :min
        h = minphase_from_amp(amp)
        return h[1:nwav]
    else
        return zerophase_from_amp(amp, nwav)
    end
end

# Homomorphic (cepstral) wavelet estimation: the log-spectrum separates into
# a smooth (wavelet) and a spiky (reflectivity) part in quefrency.
# q = lifter length in samples. Returns (wavelet amplitude spectrum, nfft).
function homomorphic_wavelet(y::Vector{Float64}, q::Int; nfft::Int = 0)
    N = length(y)
    nfft = nfft > 0 ? nextpow2_(nfft) : nextpow2_(N)
    Y = fft1(_zeropad(y, nfft))
    A = log.(max.(abs.(Y), 1e-12))
    c = real(ifft1(A))
    cl = copy(c)
    cl[q+1:end] .= 0.0     # keep low quefrency only
    Hest = exp.(fft1(cl))
    return abs.(Hest), nfft
end

# Non-blind Wiener deconvolution with a known wavelet (baseline).
function wiener_decon1(y::Vector{Float64}, h::Vector{Float64}; nsr::Float64 = 1e-2)
    N = length(y)
    nfft = nextpow2_(N + length(h))
    Y = fft1(_zeropad(y, nfft))
    H = fft1(_zeropad(h, nfft))
    X = (conj.(H) .* Y) ./ (abs2.(H) .+ nsr)
    x = real(ifft1(X))
    return x[1:N]
end

# Least-squares wavelet fit: min_h ||conv_causal(x,h) - y||^2 (ridge regularized)
function ls_wavelet(x::Vector{Float64}, y::Vector{Float64}, L::Int;
                    ridge::Float64 = 1e-6)
    rxx = autocorr_full(x, L - 1)
    scale = max(rxx[1], 1e-30)
    rhs = [xcorr_any(x, y, k - 1) for k in 1:L]
    r = rxx[1:L]
    r[1] += ridge * scale
    return toeplitz_solve(r, rhs)
end

# -----------------------------------------------------------------------------
# Section 4: 1-D blind deconvolution
# -----------------------------------------------------------------------------

# Wiggins minimum-entropy deconvolution: maximize the varimax norm
#   V(x) = sum(x^4) / (sum(x^2))^2
# over inverse filters a (a[1] = 1 fixed), x = conv_causal(y, a).
# Steepest ascent with adaptive backtracking (V is monotone by construction).
function med_decon(y::Vector{Float64}; ninv::Int = 15, iters::Int = 200,
                   mu0::Float64 = 0.5, verbose::Bool = false)
    N = length(y)
    a = zeros(Float64, ninv)
    a[1] = 1.0
    x = conv_causal(y, a)
    n2 = sum(abs2, x)
    n4 = sum(x .^ 4)
    V = n4 / (n2 * n2)
    mu = mu0
    hist = Float64[V]
    for it in 1:iters
        x = conv_causal(y, a)
        n2 = sum(abs2, x)
        n4 = sum(x .^ 4)
        V = n4 / (n2 * n2)
        x3 = x .^ 3
        # dV/da[j] = 4/n2^2 * xcorr(y, x^3, j-1) - 4*n4/n2^3 * xcorr(y, x, j-1)
        g = zeros(Float64, ninv)
        for j in 1:ninv
            c4 = xcorr_any(y, x3, j - 1)
            c2 = xcorr_any(y, x, j - 1)
            g[j] = 4.0 * c4 / (n2 * n2) - 4.0 * n4 * c2 / (n2 * n2 * n2)
        end
        gm = maximum(abs, g[2:end])
        if gm < 1e-14
            break
        end
        g ./= gm
        improved = false
        step = mu
        for _ in 1:40
            anew = copy(a)
            for j in 2:ninv
                anew[j] += step * g[j]
            end
            xnew = conv_causal(y, anew)
            m2 = sum(abs2, xnew)
            m4 = sum(xnew .^ 4)
            Vnew = m4 / (m2 * m2)
            if Vnew > V
                a = anew
                V = Vnew
                improved = true
                break
            end
            step *= 0.5
        end
        if !improved
            mu *= 0.5
            mu < 1e-12 && break
        end
        push!(hist, V)
        verbose && @printf("  med it %3d  V = %.6f\n", it, V)
    end
    x = conv_causal(y, a)
    return x, a, hist
end

# Alternating sparse blind spike deconvolution:
#   x-step: ISTA on 0.5*||conv_causal(x,h) - y||^2 + lam*||x||_1
#   h-step: ridge-regularized LS wavelet fit
# Initialized from a spiking-PEF whitening so the first h-fit already sees
# a sparse reflectivity guess.
function sparse_bd1(y::Vector{Float64}; L::Int = 15, iters::Int = 12,
                    inner::Int = 30, lam0::Float64 = 0.10)
    N = length(y)
    xw, ~ = prediction_decon(y; gap = 1, nfil = min(24, max(4, N >> 2)))
    thr = lam0 * maximum(abs, xw)
    x = [abs(v) > thr ? v : 0.0 for v in xw]
    if maximum(abs, x) == 0.0
        x = copy(xw)
    end
    h = ls_wavelet(x, y, L)
    for outer in 1:iters
        # x-step (ISTA with soft threshold)
        lip = max(sum(abs2, h), 1e-12)
        lam = lam0 * maximum(abs, x)
        for _ in 1:inner
            r = conv_causal(x, h) .- y
            g = [xcorr_any(h, r, j - 1) for j in 1:N]
            x .-= g ./ lip
            @inbounds for i in 1:N
                if abs(x[i]) <= lam / lip
                    x[i] = 0.0
                else
                    x[i] -= sign(x[i]) * lam / lip
                end
            end
        end
        # h-step
        h = ls_wavelet(x, y, L)
    end
    return x, h
end

# SIMO cross-relation blind identification (Xu et al.):
#   y1 = h1 * x,  y2 = h2 * x  (same input, two channels)
#   => conv(h2, y1) = conv(h1, y2)
# Stack the implied linear equations for the unknown theta = [h1; h2] and
# take the smallest-eigenvector of C'C. Returns (h1, h2), scaled h1[1] = 1.
function crossrel_identify(y1::Vector{Float64}, y2::Vector{Float64}; L::Int = 8)
    N1 = length(y1)
    N2 = length(y2)
    rows = Vector{Vector{Float64}}()
    nz = 2 * L
    # only equation indices where ALL taps fall inside the valid data range
    for n in L:min(N1, N2)
        row = zeros(Float64, nz)
        for k in 1:L
            row[L+k] += y1[n-k+1]   # +h2[k]*y1[n-k+1]
            row[k] -= y2[n-k+1]     # -h1[k]*y2[n-k+1]
        end
        push!(rows, row)
    end
    C = zeros(Float64, length(rows), nz)
    for (ri, row) in enumerate(rows)
        C[ri, :] .= row
    end
    M = C' * C
    ev = eigen(Symmetric(M))
    theta = ev.vectors[:, 1]
    if abs(theta[1]) < 1e-12
        return zeros(L), zeros(L)
    end
    theta ./= theta[1]
    return theta[1:L], theta[L+1:end]
end


# -----------------------------------------------------------------------------
# Section 4b (v0.2.0): robust 1-D statistics, multichannel cross-relation,
# Q estimation and non-stationary spectral whitening
# -----------------------------------------------------------------------------

function _median1(v::AbstractVector{<:Real})
    n = length(v)
    n == 0 && return 0.0
    s = sort(collect(v))
    return isodd(n) ? s[div(n + 1, 2)] : 0.5 * (s[div(n, 2)] + s[div(n, 2) + 1])
end

# Robust noise std for 1-D data: MAD of first differences (Donoho-style).
# Smooth-signal content largely cancels in the difference, noise contributes
# sigma*sqrt(2); dividing by sqrt(2) recovers the point noise level.
function mad_noise1(y::Vector{Float64})
    n = length(y)
    n < 4 && return 0.0
    d = y[2:end] .- y[1:end-1]
    m = _median1(d)
    return 1.4826 * _median1(abs.(d .- m)) / sqrt(2.0)
end

# Multichannel cross-relation identification: several channels share the
# SAME unknown input x but see different unknown kernels h_i. Convolution
# commutativity gives, for every channel pair (i, j),
#   sum_k h_j[k] * y_i[n-k+1] - sum_k h_i[k] * y_j[n-k+1] = 0
# Stacking all h_i into one unknown vector turns this into a null-space
# problem; the smallest-eigenvalue eigenvector of the normal matrix is the
# stacked kernel vector (scale fixed afterwards). Using every channel pair
# is the classical multichannel identifiability gain: a single pair leaves
# per-channel scale ambiguities that extra channels resolve.
function crossrel_identify_multi(ys::Vector{Vector{Float64}}; L::Int = 8)
    C = length(ys)
    C >= 2 || error("crossrel_identify_multi needs >= 2 channels")
    Ns = [length(y) for y in ys]
    nz = C * L
    rows = Vector{Vector{Float64}}()
    nmax = minimum(Ns)
    # only equation indices where ALL taps fall inside the valid data range
    for n in L:nmax
        for c1 in 1:C-1, c2 in c1+1:C
            row = zeros(Float64, nz)
            y1 = ys[c1]
            y2 = ys[c2]
            for k in 1:L
                row[(c2 - 1) * L + k] += y1[n-k+1]  # +h_{c2}[k] * y_{c1}
                row[(c1 - 1) * L + k] -= y2[n-k+1]  # -h_{c1}[k] * y_{c2}
            end
            push!(rows, row)
        end
    end
    isempty(rows) && return [zeros(L) for _ in 1:C]
    Mrows = zeros(Float64, length(rows), nz)
    for (ri, row) in enumerate(rows)
        for ci in 1:nz
            Mrows[ri, ci] = row[ci]
        end
    end
    M = Mrows' * Mrows
    ev = eigen(Symmetric(M))
    theta = ev.vectors[:, 1]
    # scale fix: first-channel kernel sums to one (unit-gain kernels)
    s = sum(theta[1:L])
    if abs(s) < 1e-12
        return [zeros(L) for _ in 1:C]
    end
    theta ./= s
    return [theta[(c-1)*L+1:c*L] for c in 1:C]
end

# Short-time frames: win samples, hop advance, columns = frames.
function _stft_frames(y::Vector{Float64}, win::Int, hop::Int)
    N = length(y)
    nf = 1 + max(0, div(N - win, hop))
    F = zeros(ComplexF64, win, nf)
    w = [0.5 - 0.5 * cos(2 * pi * (i - 1) / win) for i in 1:win]  # periodic hann
    for t in 1:nf
        i0 = (t - 1) * hop + 1
        for i in 1:win
            F[i, t] = y[i0 + i - 1] * w[i]
        end
        F[:, t] = fft1(F[:, t])
    end
    return F, w
end

# Q estimation from the t^*-f^* attenuation law of a constant-Q medium:
# A(f, t) ~ A0(f) * exp(-pi * f * t / Q), so the log-amplitude of a fixed
# frequency bin decays linearly in time with slope -pi*f/Q. STFT,
# then a POOLED per-bin centered regression of log A on f*t over bins in
# [fmin, fmax] whose mean power exceeds amin times the strongest in-band
# bin. Sparse-spike traces leave per-frame excitation noise larger than
# the Q decay; dense reflectivity is the regime where the fit is stable.
function q_estimate(y::Vector{Float64}, dt::Real; win::Int = 128,
                    fmin::Real = 0.5, fmax::Real = 10.0, amin::Real = 0.0)
    N = length(y)
    hop = max(1, div(win, 2))
    F, _ = _stft_frames(y, win, hop)
    win, nf = size(F)
    fs = 1.0 / Float64(dt)
    # usable bins: mean power above amin times the strongest in-band bin
    Pm = [sum(abs2, F[k, :]) / nf for k in 1:win]
    pmax = maximum(Pm)
    ks = Int[]
    for k in 2:win
        f = (k - 1) * fs / win
        (f < fmin || f > fmax) && continue
        Pm[k] >= amin * pmax && push!(ks, k)
    end
    isempty(ks) && return 0.0
    ts = [((t - 1) * hop + div(win, 2)) / fs for t in 1:nf]
    tbar = sum(ts) / nf
    # per-bin time averages of log amplitude (removes the per-bin level)
    ybar = Dict{Int,Float64}()
    for k in ks
        s = 0.0
        for t in 1:nf
            s += log(abs(F[k, t]))
        end
        ybar[k] = s / nf
    end
    num = 0.0
    den = 0.0
    # pooled, per-bin centered regression of log A on f*t: regressor
    # z = f*(t - tbar), response centered per bin. Pooling all bins
    # removes the log-of-sum-of-exponentials curvature bias of a
    # band-energy fit and averages the per-frame excitation noise.
    for k in ks
        f = (k - 1) * fs / win
        for t in 1:nf
            z = f * (ts[t] - tbar)
            aa = abs(F[k, t])
            aa > 0 || continue
            num += z * (log(aa) - ybar[k])
            den += z * z
        end
    end
    abs(den) < 1e-30 && return 0.0
    theta = num / den               # d logA / d(f*t) = -pi / Q
    theta > -1e-12 && return 0.0    # not decaying: no information
    return -pi / theta
end

# Moving average over frames (radius m), used to smooth STFT envelopes in
# time before whitening.
function _mean1(v::Vector{Float64}, m::Int)
    n = length(v)
    out = zeros(Float64, n)
    for i in 1:n
        s = 0.0
        c = 0
        for j in max(1, i - m):min(n, i + m)
            s += v[j]
            c += 1
        end
        out[i] = s / c
    end
    return out
end

# Non-stationary deconvolution via short-time spectral whitening: per STFT
# frame, divide by a temporally smoothed amplitude envelope with a per-frame
# floor (dfloor times the frame max). A single global filter cannot whiten
# time-varying attenuation (Q loss) and the static wavelet at once; per-frame
# gains track the decay. Output is a WHITENED version of the input, not a
# band-limited spike train (see README: band limitation is information-
# theoretic, whitening is what the data supports).
function nonstat_decon(y::Vector{Float64}; win::Int = 128,
                       dfloor::Real = 1e-2, sm::Int = 3)
    N = length(y)
    hop = max(1, div(win, 2))
    F, w = _stft_frames(y, win, hop)
    wnf, nf = size(F)
    # time-smoothed envelope per frequency bin
    E = zeros(Float64, wnf, nf)
    for k in 1:wnf
        E[k, :] = _mean1(abs.(F[k, :]), sm)
    end
    for t in 1:nf
        emax = maximum(E[:, t])
        emax <= 0 && continue
        for k in 1:wnf
            g = max(E[k, t], dfloor * emax)
            F[k, t] /= g
        end
    end
    # inverse STFT, overlap-add with squared-window normalization
    out = zeros(Float64, N + win)
    wsum = zeros(Float64, N + win)
    win = wnf
    for t in 1:nf
        fr = real.(ifft1(F[:, t]))
        i0 = (t - 1) * hop + 1
        for i in 1:win
            gi = i0 + i - 1
            out[gi] += fr[i] * w[i]
            wsum[gi] += w[i] * w[i]
        end
    end
    # window-sum compensation only where it is substantial: in the
    # fade-in/out ramps dividing by wsum explodes numerical residue, so
    # those samples are set to zero (~win/2 border on each side)
    o = zeros(Float64, N)
    for i in 1:N
        if wsum[i] >= 0.25
            o[i] = out[i] / wsum[i]
        end
    end
    return o
end


# -----------------------------------------------------------------------------
# Section 4c (v0.2.0): real-data I/O helpers (GeoCSV), 1-D bandpass,
# spectral flatness
# -----------------------------------------------------------------------------

# Tolerant PDS-style GeoCSV parser for annual-products time series CSVs
# (e.g. InSight SEIS event windows): '#' comment lines carry the metadata,
# one non-comment header line names the columns, then "ISO time, value"
# rows, comma-separated. Returns (samples, fs, t0-string).
function parse_geocsv(path::String)
    y = Float64[]
    fs = 0.0
    t0 = ""
    open(path, "r") do io
        headerdone = false
        for line in eachline(io)
            if startswith(line, "#")
                body = lstrip(line[2:end])
                if startswith(body, "sample_rate_hz:")
                    fs = parse(Float64, strip(body[length("sample_rate_hz:")+1:end]))
                elseif startswith(body, "start_time:")
                    t0 = strip(body[length("start_time:")+1:end])
                end
                continue
            end
            if isempty(strip(line))
                continue
            end
            if !headerdone
                headerdone = true   # column-name line, e.g. "Time, Sample"
                continue
            end
            # data row: ISO time, value
            parts = split(line, ',')
            if length(parts) >= 2
                v = tryparse(Float64, strip(parts[2]))
                v === nothing && continue
                push!(y, v)
            end
        end
    end
    return y, fs, String(t0)
end

# Brickwall bandpass on a zero-padded FFT grid. Frequencies above Nyquist
# are reflected (mirror of negative frequencies) so the response stays
# conjugate-symmetric and the output real.
function bandpass1(y::Vector{Float64}, fs::Real, f1::Real, f2::Real)
    N = length(y)
    nfft = nextpow2_(2 * N)
    Y = fft1([y; zeros(nfft - N)])
    df = Float64(fs) / nfft
    fnyq = Float64(fs) / 2
    @inbounds for k in 1:nfft
        f = (k - 1) * df
        if f > fnyq
            f = Float64(fs) - f
        end
        if !(f >= f1 && f <= f2)
            Y[k] = 0.0 + 0.0im
        end
    end
    out = real.(ifft1(Y))
    return out[1:N]
end

# Spectral flatness (Wiener entropy): geometric/arithmetic mean of the
# power spectrum, excluding DC and Nyquist bins. 1.0 = perfectly white,
# small = strongly colored. The fair whitening metric for band-limited
# data (lag-1 autocorrelation can stay ~1 even for in-band white noise).
function _specflat1(y::Vector{Float64})
    N = length(y)
    Y = fft1(y)
    n1 = 2
    n2 = iseven(N) ? div(N, 2) : div(N + 1, 2)
    p = Float64[]
    for k in n1:n2
        push!(p, abs2(Y[k]))
    end
    isempty(p) && return 1.0
    am = sum(p) / length(p)
    am <= 0 && return 1.0
    gm = exp(sum(log.(p)) / length(p))
    return gm / am
end

# -----------------------------------------------------------------------------
# Section 5: 2-D kernels, circular convolution machinery, metrics
#
# 2-D convention: all image-domain operators are CIRCULAR on an n1 x n2 grid.
# A kernel k (kh x kw, center at (div(kh-1,2)+1, div(kw-1,2)+1)) is embedded
# into the image grid with its center at (1,1); multiplication in the FFT
# domain then gives the circular convolution. Images are padded with
# replicated borders before blind decon (pad_replicate2) so that wrap-around
# only ever reads replicated pixels.
# -----------------------------------------------------------------------------

function gaussian_kernel2(sigma::Real; radius::Int = max(1, ceil(Int, 3 * sigma)))
    n = 2 * radius + 1
    K = zeros(Float64, n, n)
    c = radius + 1
    for i in 1:n, j in 1:n
        di = i - c
        dj = j - c
        K[i, j] = exp(-(di * di + dj * dj) / (2 * sigma * sigma))
    end
    return K ./ sum(K)
end

function motion_kernel2(len::Real, angle_deg::Real)
    r = max(1, ceil(Int, len / 2))
    n = 2 * r + 1
    K = zeros(Float64, n, n)
    c = r + 1
    ang = deg2rad(angle_deg)
    # sample the line through the center at subpixel offsets
    ns = max(16, 4 * n)
    for s in 0:(ns-1)
        t = -len / 2 + (len / 2 - (-len / 2)) * s / (ns - 1)
        i = round(Int, c + t * cos(ang))
        j = round(Int, c + t * sin(ang))
        if 1 <= i <= n && 1 <= j <= n
            K[i, j] += 1.0
        end
    end
    s = sum(K)
    if s <= 0
        K[c, c] = 1.0
        return K
    end
    return K ./ s
end

function disk_kernel2(radius::Real)
    r = max(1, ceil(Int, radius))
    n = 2 * r + 1
    K = zeros(Float64, n, n)
    c = r + 1
    for i in 1:n, j in 1:n
        if hypot(i - c, j - c) <= radius + 0.25
            K[i, j] = 1.0
        end
    end
    return K ./ sum(K)
end

# embed a small kernel into an n1 x n2 grid with center at (1,1)
function embed_kernel2(k::Matrix{Float64}, n1::Int, n2::Int)
    kh, kw = size(k)
    c1 = div(kh - 1, 2) + 1
    c2 = div(kw - 1, 2) + 1
    K = zeros(Float64, n1, n2)
    for u in 1:kh
        i = mod1(u - c1 + 1, n1)
        for v in 1:kw
            j = mod1(v - c2 + 1, n2)
            K[i, j] += k[u, v]
        end
    end
    return K
end

# inverse of embed_kernel2: extract the kh x kw kernel from an embedded grid
function crop_kernel2(K::Matrix{Float64}, kh::Int, kw::Int)
    n1, n2 = size(K)
    c1 = div(kh - 1, 2) + 1
    c2 = div(kw - 1, 2) + 1
    k = zeros(Float64, kh, kw)
    for u in 1:kh
        i = mod1(u - c1 + 1, n1)
        for v in 1:kw
            j = mod1(v - c2 + 1, n2)
            k[u, v] = K[i, j]
        end
    end
    return k
end

# circular convolution via spectra (X̂, K̂ = fft2_ of the real arrays)
function cconv2_spec(Xs::Matrix{ComplexF64}, Ks::Matrix{ComplexF64})
    return real.(ifft2_(Xs .* Ks))
end

function cconv2(x::Matrix{Float64}, k::Matrix{Float64})
    n1, n2 = size(x)
    Ks = fft2_(embed_kernel2(k, n1, n2))
    return cconv2_spec(fft2_(x), Ks)
end

# reference: direct circular convolution (for selftest)
function cconv2_direct(x::Matrix{Float64}, k::Matrix{Float64})
    n1, n2 = size(x)
    kh, kw = size(k)
    c1 = div(kh - 1, 2) + 1
    c2 = div(kw - 1, 2) + 1
    y = zeros(Float64, n1, n2)
    for u in 1:kh, v in 1:kw
        kv = k[u, v]
        kv == 0 && continue
        for i in 1:n1, j in 1:n2
            y[i, j] += kv * x[mod1(i - u + c1, n1), mod1(j - v + c2, n2)]
        end
    end
    return y
end

# Fourier multipliers of the circular forward differences
#   (Dx x)[i,j] = x[mod1(i+1,n1), j] - x[i,j]
function _grad_mults_uncached(n1::Int, n2::Int)
    d1 = [cis(2 * pi * (i - 1) / n1) - 1.0 for i in 1:n1]
    d2 = [cis(2 * pi * (j - 1) / n2) - 1.0 for j in 1:n2]
    Dx = Matrix{ComplexF64}(undef, n1, n2)
    Dy = Matrix{ComplexF64}(undef, n1, n2)
    for i in 1:n1, j in 1:n2
        Dx[i, j] = d1[i]
        Dy[i, j] = d2[j]
    end
    return Dx, Dy
end

const _GRAD_CACHE = Dict{Tuple{Int,Int},Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()
const _GRADC_CACHE = Dict{Tuple{Int,Int},Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}}()

function grad_mults(n1::Int, n2::Int)
    key = (n1, n2)
    v = get(_GRAD_CACHE, key, nothing)
    if v === nothing
        v = _grad_mults_uncached(n1, n2)
        _GRAD_CACHE[key] = v
    end
    return v
end

# conjugate gradient multipliers (used every _hq_amult call), cached too
function gradc_mults(n1::Int, n2::Int)
    key = (n1, n2)
    v = get(_GRADC_CACHE, key, nothing)
    if v === nothing
        Dx, Dy = grad_mults(n1, n2)
        v = (conj.(Dx), conj.(Dy))
        _GRADC_CACHE[key] = v
    end
    return v
end

# spatial circular forward differences
function circgrad(x::Matrix{Float64})
    n1, n2 = size(x)
    gx = zeros(Float64, n1, n2)
    gy = zeros(Float64, n1, n2)
    for i in 1:n1, j in 1:n2
        gx[i, j] = x[mod1(i + 1, n1), j] - x[i, j]
        gy[i, j] = x[i, mod1(j + 1, n2)] - x[i, j]
    end
    return gx, gy
end

# pad an image to (n1, n2) with replicated borders; original centered
function pad_replicate2(img::Matrix{Float64}, n1::Int, n2::Int)
    h, w = size(img)
    o1 = div(n1 - h, 2)
    o2 = div(n2 - w, 2)
    out = zeros(Float64, n1, n2)
    for i in 1:n1
        si = clamp(i - o1, 1, h)
        for j in 1:n2
            out[i, j] = img[si, clamp(j - o2, 1, w)]
        end
    end
    return out
end

function crop_center2(img::Matrix{Float64}, h::Int, w::Int)
    n1, n2 = size(img)
    o1 = div(n1 - h, 2)
    o2 = div(n2 - w, 2)
    return img[o1+1:o1+h, o2+1:o2+w]
end

function psnr(x::Matrix{Float64}, y::Matrix{Float64})
    mse = sum((x .- y) .^ 2) / length(x)
    mse <= 1e-20 && return 99.0
    return 10 * log10(1.0 / mse)
end

function psnr1(x::Vector{Float64}, y::Vector{Float64})
    mse = sum((x .- y) .^ 2) / length(x)
    p = maximum(abs, x)
    mse <= 1e-20 * p * p && return 99.0
    return 10 * log10(p * p / mse)
end

# normalized correlation of two same-shape kernels
function kernel_corr(k1::Matrix{Float64}, k2::Matrix{Float64})
    n1 = max(size(k1, 1), size(k2, 1))
    n2 = max(size(k1, 2), size(k2, 2))
    a = zeros(Float64, n1, n2)
    b = zeros(Float64, n1, n2)
    i1 = div(n1 - size(k1, 1), 2) + 1
    j1 = div(n2 - size(k1, 2), 2) + 1
    i2 = div(n1 - size(k2, 1), 2) + 1
    j2 = div(n2 - size(k2, 2), 2) + 1
    a[i1:i1+size(k1,1)-1, j1:j1+size(k1,2)-1] .= k1
    b[i2:i2+size(k2,1)-1, j2:j2+size(k2,2)-1] .= k2
    na = norm(a)
    nb = norm(b)
    (na < 1e-15 || nb < 1e-15) && return 0.0
    return dot(a, b) / (na * nb)
end

function kernel_corr1(h1::Vector{Float64}, h2::Vector{Float64})
    n = max(length(h1), length(h2))
    a = zeros(n)
    b = zeros(n)
    o1 = div(n - length(h1), 2) + 1
    o2 = div(n - length(h2), 2) + 1
    a[o1:o1+length(h1)-1] .= h1
    b[o2:o2+length(h2)-1] .= h2
    na = norm(a)
    nb = norm(b)
    (na < 1e-15 || nb < 1e-15) && return 0.0
    return dot(a, b) / (na * nb)
end

# -----------------------------------------------------------------------------
# Section 6: proximal operator of tau*|w|^alpha (alpha < 1), scalar Newton.
#
# prox(z) = argmin_u 0.5*(u-z)^2 + tau*|u|^alpha,  u real.
# For alpha < 1 the objective is non-convex in the strict sense but the
# prox is well defined: u = 0 for |z| <= m*, otherwise the interior root of
#   u + tau*alpha*u^(alpha-1) = |z|   (on u > u*), chosen by comparing costs.
# -----------------------------------------------------------------------------

function _prox_lapower_scalar(z::Float64, tau::Float64, alpha::Float64)
    if alpha >= 0.999
        # TV limit: soft threshold
        return sign(z) * max(abs(z) - tau, 0.0)
    end
    az = abs(z)
    # location and value of the minimum of g(u) = u + tau*alpha*u^(alpha-1)
    ustar = (tau * alpha * (1 - alpha)) ^ (1 / (2 - alpha))
    mstar = ustar * (2 - alpha) / (1 - alpha)
    if az <= mstar
        return 0.0
    end
    # find the root of g(u) = az on (ustar, inf) by bisection
    g(u) = u + tau * alpha * u ^ (alpha - 1) - az
    lo = ustar
    hi = max(az, ustar) + 1.0
    while g(hi) < 0.0
        hi *= 2.0
        hi > 1e30 && return sign(z) * az
    end
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0
            lo = mid
        else
            hi = mid
        end
    end
    u2 = 0.5 * (lo + hi)
    # compare cost(u = 0) vs cost(u = u2)
    c0 = 0.5 * az * az
    c2 = 0.5 * (u2 - az) ^ 2 + tau * u2 ^ alpha
    return c2 < c0 ? sign(z) * u2 : 0.0
end

function prox_lapower(z::Matrix{Float64}, tau::Float64, alpha::Float64)
    out = similar(z)
    @inbounds for i in eachindex(z)
        out[i] = _prox_lapower_scalar(z[i], tau, alpha)
    end
    return out
end

# -----------------------------------------------------------------------------
# Section 7: 2-D non-blind deconvolution (known kernel)
# -----------------------------------------------------------------------------

function decon_wiener2(y::Matrix{Float64}, k::Matrix{Float64};
                       nsr::Float64 = 1e-2)
    Ks = fft2_(embed_kernel2(k, size(y)...))
    Y = fft2_(y)
    X = (conj.(Ks) .* Y) ./ (abs2.(Ks) .+ nsr)
    return real.(ifft2_(X))
end

function decon_rl2(y::Matrix{Float64}, k::Matrix{Float64}; iters::Int = 50)
    n1, n2 = size(y)
    Ks = fft2_(embed_kernel2(k, n1, n2))
    Y = fft2_(y)
    x = max.(y, 0.0) .+ 1e-4
    eps = 1e-9
    for _ in 1:iters
        c = cconv2_spec(fft2_(x), Ks)
        c = max.(c, eps)
        ratio = y ./ c
        x = x .* real.(ifft2_(conj.(Ks) .* fft2_(ratio)))
        x = max.(x, 0.0)
    end
    return x
end

# Hyper-Laplacian-prior deconvolution via IRLS + conjugate gradients:
#   min_x 0.5*||k (x) - y||^2 + (lam/2)*sum_i w_i |(grad x)_i|^2
# with reweighted least-squares weights
#   w_i = (|(grad x)_i|^2 + epsw^2)^((alpha-2)/2)
# which at the fixed point reproduces the prior lam*sum |grad x|^alpha
# (alpha = 1 -> total variation, alpha = 0.5/0.8 -> heavy-tailed priors;
# Krishnan & Fergus 2009 style, solved with per-pixel-weighted CG).
# The x-subproblem is NOT Fourier-diagonal (per-pixel weights), so each
# IRLS round runs preconditioned-free CG on the weighted normal equations.
function _hq_amult(v::Matrix{Float64}, absK2::Matrix{Float64},
                   Dx::Matrix{ComplexF64}, Dy::Matrix{ComplexF64},
                   lam::Float64, w1::Matrix{Float64}, w2::Matrix{Float64})
    V = fft2_(v)
    out = real.(ifft2_(absK2 .* V))
    gx = real.(ifft2_(Dx .* V))
    gy = real.(ifft2_(Dy .* V))
    gx .*= w1
    gy .*= w2
    Dxc, Dyc = gradc_mults(size(v)...)   # same values, cached (v0.2.0)
    out .+= lam .* real.(ifft2_(Dxc .* fft2_(gx) .+ Dyc .* fft2_(gy)))
    return out
end

function _hq_cg(x0::Matrix{Float64}, b::Matrix{Float64},
                absK2::Matrix{Float64}, Dx::Matrix{ComplexF64},
                Dy::Matrix{ComplexF64}, lam::Float64,
                w1::Matrix{Float64}, w2::Matrix{Float64}, iters::Int)
    x = copy(x0)
    r = b .- _hq_amult(x, absK2, Dx, Dy, lam, w1, w2)
    p = copy(r)
    rs = sum(r .* r)
    bnorm = sqrt(sum(b .* b))
    for _ in 1:iters
        Ap = _hq_amult(p, absK2, Dx, Dy, lam, w1, w2)
        denom = sum(p .* Ap)
        denom <= 0 && break
        alpha = rs / denom
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = sum(r .* r)
        sqrt(rsnew) < 1e-10 * bnorm && break
        p = r .+ (rsnew / rs) .* p
        rs = rsnew
    end
    return x
end

# robust = false: standard IRLS-form hyper-Laplacian prior + L2 data.
# robust = true: additional IRLS Huber weights on the DATA residual
# (delta = rdelta, auto 1.5 * MAD noise of y): gross outliers (salt
# noise, impulsive clutter) get weights delta/|r|, degrading the data
# term gracefully instead of dragging the solution. The weighted chain
# is A' W A + prior with per-iteration weights w_d = min(1, d/|r|).
function decon_hq2(y::Matrix{Float64}, k::Matrix{Float64};
                   alpha::Float64 = 0.8, lam::Float64 = 2e-4,
                   iters::Int = 8, cg_iters::Int = 30, epsw::Float64 = 0.03,
                   robust::Bool = false, rdelta::Float64 = 0.0)
    n1, n2 = size(y)
    Ks = fft2_(embed_kernel2(k, n1, n2))
    absK2 = abs2.(Ks)
    Y = fft2_(y)
    b = real.(ifft2_(conj.(Ks) .* Y))
    Dx, Dy = grad_mults(n1, n2)
    x = copy(y)
    if !robust
        for _ in 1:iters
            gx, gy = circgrad(x)
            w1 = (gx .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
            w2 = (gy .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
            x = _hq_cg(x, b, absK2, Dx, Dy, lam, w1, w2, cg_iters)
        end
        return x
    end
    delta = rdelta > 0 ? rdelta : 1.5 * mad_noise2(y)
    delta > 0 || (delta = 1e-12)
    for _ in 1:iters
        # data-space Huber weights from the current residual
        r = real.(ifft2_(Ks .* fft2_(x))) .- y
        wd = _huber_weights(r, delta)
        b_r = real.(ifft2_(conj.(Ks) .* fft2_(wd .* y)))
        gx, gy = circgrad(x)
        w1 = (gx .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
        w2 = (gy .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
        x = _hq_cg_w(x, b_r, Ks, wd, Dx, Dy, lam, w1, w2, cg_iters)
    end
    return x
end

# -----------------------------------------------------------------------------
# Section 8: 2-D blind deconvolution
# -----------------------------------------------------------------------------

# Ayers & Dainty iterative blind deconvolution: alternate Richardson-Lucy
# updates for the image and the kernel, with positivity, support and
# sum-to-one projections on the kernel after every update.
# NOTE: a delta-kernel init is a TRIVIAL FIXED POINT (x-step is a no-op and
# the kernel refits delta). We init with a small gaussian instead.
function ibd_rl2(y::Matrix{Float64}; kh::Int = 9, kw::Int = 9,
                 iters::Int = 60, xsteps::Int = 1, ksteps::Int = 1,
                 kinit::Union{Matrix{Float64}, Nothing} = nothing)
    n1, n2 = size(y)
    Y = fft2_(y)
    x = max.(y, 0.0) .+ 1e-3
    if kinit === nothing
        k = gaussian_kernel2(1.0; radius = max(1, min(3, div(kh - 1, 2), div(kw - 1, 2))))
    else
        k = copy(kinit)
    end
    eps = 1e-9
    Ks = fft2_(embed_kernel2(k, n1, n2))   # carried across iterations (v0.2.0)
    for _ in 1:iters
        # --- image RL steps
        for _ in 1:xsteps
            Xs = fft2_(x)
            c = cconv2_spec(Xs, Ks)
            c = max.(c, eps)
            ratio = y ./ c
            x = x .* real.(ifft2_(conj.(Ks) .* fft2_(ratio)))
            x = max.(x, 0.0)
        end
        # --- kernel RL step in the embedded grid:
        # k_emb <- k_emb .* adjoint_of_kernel_op(ratio), ratio = y ./ (x (x) k)
        for _ in 1:ksteps
            Xs = fft2_(x)
            c = cconv2_spec(Xs, Ks)
            c = max.(c, eps)
            ratio = y ./ c
            # adjoint w.r.t. the embedded kernel:
            # (A' r)_emb = conj(Xs) .* R ./ (n1*n2)   (real-valued after ifft)
            g = real.(ifft2_(conj.(Xs) .* fft2_(ratio)))
            kemb = embed_kernel2(k, n1, n2)
            kemb = kemb .* (g .+ 1e-12)
            k = crop_kernel2(kemb, kh, kw)
            # projection: positivity, support, sum one
            @inbounds for i in eachindex(k)
                k[i] = max(k[i], 0.0)
            end
            s = sum(k)
            s > 0 ? (k ./= s) : (k[div(kh-1,2)+1, div(kw-1,2)+1] = 1.0)
            Ks = fft2_(embed_kernel2(k, n1, n2))
        end
    end
    return x, k
end

# Constrained least-squares kernel estimate given image x and data y:
#   k = argmin ||x (x) k - y||^2 + gamma*||k||^2   (diagonal FFT solve)
# followed by projection onto {k >= 0, sum k = 1, support kh x kw}.
# rmax: radial low-pass on the estimated kernel SPECTRUM (rmax = 0.5 keeps
# up to half the Nyquist radius). True small-support kernels are strongly
# band-limited, while high-frequency bins of the raw LS solution are pure
# noise amplification - the mask is what makes the estimate noise-robust.
function kernel_ls2(y::Matrix{Float64}, x::Matrix{Float64}, kh::Int, kw::Int;
                    gamma::Float64 = 1e-4, rmax::Float64 = 0.5)
    n1, n2 = size(y)
    Xs = fft2_(x)
    Y = fft2_(y)
    K = (conj.(Xs) .* Y) ./ (abs2.(Xs) .+ gamma)
    for i in 1:n1
        fi = 2 * min(i - 1, n1 - i + 1) / n1
        for j in 1:n2
            fj = 2 * min(j - 1, n2 - j + 1) / n2
            if hypot(fi, fj) > rmax
                K[i, j] = 0.0
            end
        end
    end
    kemb = real.(ifft2_(K))
    k = crop_kernel2(kemb, kh, kw)
    @inbounds for i in eachindex(k)
        k[i] = max(k[i], 0.0)
    end
    s = sum(k)
    if s > 0
        k ./= s
    else
        k .= 0.0
        k[div(kh-1,2)+1, div(kw-1,2)+1] = 1.0
    end
    return k
end

# Chan & Wong alternating minimization with a TV prior on the image:
#   x-step: IRLS/CG TV restoration with the current kernel
#   k-step: constrained LS kernel estimate with the current image
# A delta-kernel init is a trivial fixed point, so we start from a small
# gaussian (same lesson as ibd_rl2).
function am_tv2(y::Matrix{Float64}; kh::Int = 11, kw::Int = 11,
                iters::Int = 5, lam::Float64 = 5e-4, inner::Int = 4,
                cg_iters::Int = 25, gamma::Float64 = 1e-4,
                kinit::Union{Matrix{Float64}, Nothing} = nothing)
    n1, n2 = size(y)
    x = copy(y)
    if kinit === nothing
        k = gaussian_kernel2(1.0; radius = max(1, min(3, div(kh - 1, 2), div(kw - 1, 2))))
    else
        k = copy(kinit)
    end
    for it in 1:iters
        x = decon_hq2(y, k; alpha = 1.0, lam = lam, iters = inner,
                      cg_iters = cg_iters)
        k = kernel_ls2(y, x, kh, kw; gamma = gamma)
    end
    return x, k
end

# Osher-Rudin shock filter:  u_t = -sign(lap u) * |grad u|
# sharpens edges without amplifying smooth-region noise (monotone morphological
# PDE). Circular differences, explicit Euler.
function shock2(u::Matrix{Float64}; iters::Int = 10, dt::Float64 = 0.05)
    n1, n2 = size(u)
    v = copy(u)
    for _ in 1:iters
        lap = zeros(Float64, n1, n2)
        gmag = zeros(Float64, n1, n2)
        @inbounds for i in 1:n1, j in 1:n2
            ip = mod1(i + 1, n1)
            im = mod1(i - 1, n1)
            jp = mod1(j + 1, n2)
            jm = mod1(j - 1, n2)
            l = v[ip, j] + v[im, j] + v[i, jp] + v[i, jm] - 4 * v[i, j]
            lap[i, j] = l
            gx = (v[ip, j] - v[im, j]) / 2
            gy = (v[i, jp] - v[i, jm]) / 2
            gmag[i, j] = hypot(gx, gy)
        end
        @inbounds for i in eachindex(v)
            v[i] -= dt * sign(lap[i]) * gmag[i]
        end
        _clamp01!(v)
    end
    return v
end

# Xu & Jia two-stage fast kernel estimation (simplified):
#   1. shock-filter the observation to restore sharp salient edges
#   2. gradient-domain kernel LS:  min_k ||grad(k (x) u_shock) - grad y||^2
#      + gamma*||k||^2  (diagonal FFT solve), projected onto the simplex
#   3. final hyper-Laplacian restoration with the estimated kernel
# edge_frac > 0: restrict the kernel LS to the strongest-|grad| pixels
# (top edge_frac fraction by gradient magnitude of the shock image),
# approximated as a pixel-domain mask on both u and y (weight-commuted:
# masking in pixel domain before the LS transfer, standard Xu-Jia edge
# selection). Keeps noisy smooth regions out of the kernel equations.
function xujia2(y::Matrix{Float64}; kh::Int = 13, kw::Int = 13,
                shock_iters::Int = 10, gamma::Float64 = 5e-3,
                alpha::Float64 = 0.8, lam::Float64 = 3e-4,
                edge_frac::Real = 0.0)
    n1, n2 = size(y)
    u = shock2(y; iters = shock_iters)
    if edge_frac > 0
        # edge selection via EXACT per-pixel weighted kernel LS in the
        # spatial domain (a hard 0/1 mask on the FFT transfer transfer
        # injects the mask boundary itself into the estimate and measurably
        # HURTS; the weighted normal equations keep only edge-pixel rows)
        gx, gy = circgrad(u)
        gm = abs.(gx) .+ abs.(gy)
        thr = _quantile1(vec(gm), 1.0 - clamp(Float64(edge_frac), 0.0, 1.0))
        m = gm .> thr
        # the spatial-domain ridge scale differs from the FFT-domain
        # gamma; 1e-4 is calibrated like kernel_ls2's regularization
        k = _kernel_ls2_w(y, u, kh, kw; gamma = 1e-4, w = m)
        x = decon_hq2(y, k; alpha = alpha, lam = lam, iters = 8, cg_iters = 25)
        return x, k, u
    end
    Xs = fft2_(u)
    Y = fft2_(y)
    Dx, Dy = grad_mults(n1, n2)
    absD2 = abs2.(Dx) .+ abs2.(Dy)
    numer = conj.(Xs) .* Y .* absD2
    denom = abs2.(Xs) .* absD2 .+ gamma
    K = numer ./ denom
    kemb = real.(ifft2_(K))
    k = crop_kernel2(kemb, kh, kw)
    @inbounds for i in eachindex(k)
        k[i] = max(k[i], 0.0)
    end
    s = sum(k)
    if s > 0
        k ./= s
    else
        k .= 0.0
        k[div(kh-1,2)+1, div(kw-1,2)+1] = 1.0
    end
    x = decon_hq2(y, k; alpha = alpha, lam = lam, iters = 8, cg_iters = 25)
    return x, k, u
end


# -----------------------------------------------------------------------------
# Section 8b (v0.2.0): robust data terms, coarse-to-fine pyramid,
# multichannel/multiframe blind deconvolution, kernel-space restart search
# -----------------------------------------------------------------------------

# Robust noise std for 2-D data: MAD of diagonal differences (circular).
# Smooth content largely cancels in the diagonal difference; noise
# contributes sigma*sqrt(2).
function mad_noise2(img::Matrix{Float64})
    n1, n2 = size(img)
    (n1 < 4 || n2 < 4) && return 0.0
    v = Vector{Float64}(undef, n1 * n2)
    idx = 0
    @inbounds for i in 1:n1, j in 1:n2
        idx += 1
        v[idx] = img[mod1(i + 1, n1), mod1(j + 1, n2)] - img[i, j]
    end
    m = _median1(v)
    return 1.4826 * _median1(abs.(v .- m)) / sqrt(2.0)
end

# Huber IRLS weights for a residual field r: w = min(1, delta/|r|).
# Inliers keep weight 1, gross outliers (salt noise, impulsive clutter)
# are down-weighted to delta/|r|.
function _huber_weights(r::Matrix{Float64}, delta::Float64)
    w = ones(Float64, size(r))
    delta > 0 || return w
    @inbounds for i in eachindex(r)
        a = abs(r[i])
        if a > delta
            w[i] = delta / a
        end
    end
    return w
end

# Weighted CG operator A' W A + prior (robust chain of decon_hq2)
function _hq_amult_w(v::Matrix{Float64}, Ks::Matrix{ComplexF64},
                     wd::Matrix{Float64}, Dx::Matrix{ComplexF64},
                     Dy::Matrix{ComplexF64}, lam::Float64,
                     w1::Matrix{Float64}, w2::Matrix{Float64})
    V = fft2_(v)
    t = real.(ifft2_(Ks .* V))
    t .*= wd
    out = real.(ifft2_(conj.(Ks) .* fft2_(t)))
    gx = real.(ifft2_(Dx .* V))
    gy = real.(ifft2_(Dy .* V))
    gx .*= w1
    gy .*= w2
    Dxc, Dyc = gradc_mults(size(v)...)
    out .+= lam .* real.(ifft2_(Dxc .* fft2_(gx) .+ Dyc .* fft2_(gy)))
    return out
end

function _hq_cg_w(x0::Matrix{Float64}, b::Matrix{Float64},
                  Ks::Matrix{ComplexF64}, wd::Matrix{Float64},
                  Dx::Matrix{ComplexF64}, Dy::Matrix{ComplexF64},
                  lam::Float64, w1::Matrix{Float64}, w2::Matrix{Float64},
                  iters::Int)
    x = copy(x0)
    r = b .- _hq_amult_w(x, Ks, wd, Dx, Dy, lam, w1, w2)
    p = copy(r)
    rs = sum(r .* r)
    bnorm = sqrt(sum(b .* b))
    for _ in 1:iters
        Ap = _hq_amult_w(p, Ks, wd, Dx, Dy, lam, w1, w2)
        denom = sum(p .* Ap)
        denom <= 0 && break
        alpha = rs / denom
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = sum(r .* r)
        sqrt(rsnew) < 1e-10 * bnorm && break
        p = r .+ (rsnew / rs) .* p
        rs = rsnew
    end
    return x
end

# Multichannel CG: data term is the SUM over channels of ||k_s (x) x - y_s|^2
function _hq_amult_multi(v::Matrix{Float64}, absK2s::Vector{Matrix{Float64}},
                         Dx::Matrix{ComplexF64}, Dy::Matrix{ComplexF64},
                         lam::Float64, w1::Matrix{Float64}, w2::Matrix{Float64})
    V = fft2_(v)
    out = zeros(Float64, size(v))
    for s in 1:length(absK2s)
        out .+= real.(ifft2_(absK2s[s] .* V))
    end
    gx = real.(ifft2_(Dx .* V))
    gy = real.(ifft2_(Dy .* V))
    gx .*= w1
    gy .*= w2
    Dxc, Dyc = gradc_mults(size(v)...)
    out .+= lam .* real.(ifft2_(Dxc .* fft2_(gx) .+ Dyc .* fft2_(gy)))
    return out
end

function _hq_cg_multi(x0::Matrix{Float64}, b::Matrix{Float64},
                      absK2s::Vector{Matrix{Float64}},
                      Dx::Matrix{ComplexF64}, Dy::Matrix{ComplexF64},
                      lam::Float64, w1::Matrix{Float64}, w2::Matrix{Float64},
                      iters::Int)
    x = copy(x0)
    r = b .- _hq_amult_multi(x, absK2s, Dx, Dy, lam, w1, w2)
    p = copy(r)
    rs = sum(r .* r)
    bnorm = sqrt(sum(b .* b))
    for _ in 1:iters
        Ap = _hq_amult_multi(p, absK2s, Dx, Dy, lam, w1, w2)
        denom = sum(p .* Ap)
        denom <= 0 && break
        alpha = rs / denom
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = sum(r .* r)
        sqrt(rsnew) < 1e-10 * bnorm && break
        p = r .+ (rsnew / rs) .* p
        rs = rsnew
    end
    return x
end

# Joint blind energy of (ys, x, ks): TV prior term + per-channel mse,
# divided by the number of channels. Used ONLY to rank candidate
# solutions (restart search, multiframe bootstrap); absolute values have
# no calibration.
function _joint_energy2(ys::Vector{Matrix{Float64}}, x::Matrix{Float64},
                        ks::Vector{Matrix{Float64}}, lam::Float64)
    n1, n2 = size(x)
    gx, gy = circgrad(x)
    e = lam * (sum(abs, gx) + sum(abs, gy)) / (n1 * n2)
    nch = length(ys)
    for s in 1:nch
        c = cconv2_spec(fft2_(x), fft2_(embed_kernel2(ks[s], n1, n2)))
        r = c .- ys[s]
        e += sum(abs2, r) / length(ys[s])
    end
    return e / nch
end

# odd-valued ceil division (kernel sizes must stay odd for center embedding)
function _odd_ceil_div(a::Int, b::Int)
    c = cld(a, b)
    return isodd(c) ? c : c + 1
end

# pyramid downsample: separable [1 4 6 4 1]/16 binomial, clamped borders
function _pyr_down(img::Matrix{Float64})
    n1, n2 = size(img)
    m1 = cld(n1, 2)
    m2 = cld(n2, 2)
    t = (1.0, 4.0, 6.0, 4.0, 1.0)
    out = zeros(Float64, m1, m2)
    @inbounds for r in 1:m1, s in 1:m2
        acc = 0.0
        for a in 0:4
            i = clamp(2 * r - 1 + a - 2, 1, n1)
            wa = t[a + 1]
            for b in 0:4
                j = clamp(2 * s - 1 + b - 2, 1, n2)
                acc += wa * t[b + 1] * img[i, j]
            end
        end
        out[r, s] = acc / 256.0
    end
    return out
end

# bilinear 2x kernel upsample (kh x kw -> (2kh-1) x (2kw-1)), renormalized
function _kern_up2(k::Matrix{Float64})
    kh, kw = size(k)
    m1 = 2 * kh - 1
    m2 = 2 * kw - 1
    out = zeros(Float64, m1, m2)
    @inbounds for i in 1:m1, j in 1:m2
        fi = (i + 1) / 2
        fj = (j + 1) / 2
        i0 = clamp(floor(Int, fi), 1, kh)
        j0 = clamp(floor(Int, fj), 1, kw)
        i1 = clamp(i0 + 1, 1, kh)
        j1 = clamp(j0 + 1, 1, kw)
        a = fi - floor(Int, fi)
        b = fj - floor(Int, fj)
        out[i, j] = (1 - a) * (1 - b) * k[i0, j0] + a * (1 - b) * k[i1, j0] +
                    (1 - a) * b * k[i0, j1] + a * b * k[i1, j1]
    end
    s = sum(out)
    s > 0 && (out ./= s)
    return out
end

# Coarse-to-fine blind deconvolution: run the blind scheme on a gaussian
# pyramid, coarsest first, and warm-start each level's kernel with the
# bilinear 2x upsample of the previous level's estimate. On coarse levels
# the kernel support covers relatively more image structure, which both
# stabilizes the joint estimate and speeds the fine level (fewer iters
# needed and no large-blur local minimum). Kernel size halves
# (odd-rounded-up) per level. HONEST LIMIT: the :amtv variant collapses
# on strongly downsampled levels (shock filtering aliases, and the TV
# x-step trusts a badly wrong kernel); :ibd is the default and the only
# variant checked by the selftest.
function blind_pyramid2(y::Matrix{Float64}; method::Symbol = :ibd,
                        kh::Int = 25, kw::Int = 25, levels::Int = 0,
                        iters::Int = 60)
    n1, n2 = size(y)
    # auto levels: coarsest grid stays >= ~32 px
    lmax = max(1, floor(Int, log2(min(n1, n2) / 32.0)) + 1)
    L = levels > 0 ? min(levels, lmax) : lmax
    pyr = [y]
    for _ in 2:L
        push!(pyr, _pyr_down(pyr[end]))
    end
    kcur = nothing
    x = similar(y)
    for lv in (L - 1):-1:0
        yl = pyr[lv + 1]
        khl = _odd_ceil_div(kh, 1 << lv)
        kwl = _odd_ceil_div(kw, 1 << lv)
        kc = kcur
        if kc !== nothing
            kcu = _kern_up2(kc)   # grow in lockstep with the grid
            kcu = crop_center2(kcu, min(size(kcu, 1), khl), min(size(kcu, 2), kwl))
            s = sum(kcu)
            if s > 0
                kcu ./= s
                kc = kcu
            end
        end
        if method == :ibd
            x, kcur = ibd_rl2(yl; kh = khl, kw = kwl, iters = iters, kinit = kc)
        elseif method == :amtv
            x, kcur = am_tv2(yl; kh = khl, kw = kwl, iters = 5, kinit = kc)
        else
            error("blind_pyramid2: unknown method $(method)")
        end
    end
    return x, kcur
end

# Exact per-pixel WEIGHTED kernel ridge LS in the spatial domain:
#   min_k sum_p w_p (k (x) x)_p - y_p)^2 + gamma*||k||^2
# built as explicit normal equations (kh*kw x kh*kw, solved directly).
# O(P * L^2) with P = n1*n2 pixels, only used for small kernels on
# small grids (edge selection in xujia2); the unweighted large-scale
# path stays in the FFT domain (kernel_ls2).
function _kernel_ls2_w(y::Matrix{Float64}, x::Matrix{Float64},
                       kh::Int, kw::Int; gamma::Float64 = 1e-4,
                       w::Union{AbstractMatrix, Nothing} = nothing)
    n1, n2 = size(y)
    size(x) == (n1, n2) || error("_kernel_ls2_w: x and y must match")
    L = kh * kw
    ww = w === nothing ? ones(Float64, n1, n2) : Matrix{Float64}(w)
    ch = div(kh - 1, 2) + 1
    cw = div(kw - 1, 2) + 1
    # A' W y and A' W A by direct accumulation
    b = zeros(Float64, L)
    # response vector
    for a1 in 1:kh, a2 in 1:kw
        ia = (a1 - 1) * kw + a2
        da = a1 - ch
        db = a2 - cw
        s = 0.0
        @inbounds for p1 in 1:n1, p2 in 1:n2
            wp = ww[p1, p2]
            wp == 0 && continue
            xs = x[mod1(p1 - da, n1), mod1(p2 - db, n2)]
            if xs != 0.0
                s += wp * xs * y[p1, p2]
            end
        end
        b[ia] = s
    end
    M = zeros(Float64, L, L)
    for a1 in 1:kh, a2 in 1:kw
        ia = (a1 - 1) * kw + a2
        da = a1 - ch
        db = a2 - cw
        for c1 in 1:kh, c2 in 1:kw
            ic = (c1 - 1) * kw + c2
            dc = c1 - ch
            de = c2 - cw
            s = 0.0
            @inbounds for p1 in 1:n1, p2 in 1:n2
                wp = ww[p1, p2]
                wp == 0 && continue
                xs1 = x[mod1(p1 - da, n1), mod1(p2 - db, n2)]
                xs2 = x[mod1(p1 - dc, n1), mod1(p2 - de, n2)]
                if xs1 != 0.0 && xs2 != 0.0
                    s += wp * xs1 * xs2
                end
            end
            M[ia, ic] = s
        end
    end
    for i in 1:L
        M[i, i] += gamma
    end
    kv = M \ b
    k = reshape(kv, kh, kw)
    @inbounds for i in eachindex(k)
        k[i] = max(k[i], 0.0)
    end
    s = sum(k)
    if s > 0
        k ./= s
    else
        k .= 0.0
        k[ch, cw] = 1.0
    end
    return k
end

# Quantile of a flat sample (linear interpolation between order stats)
function _quantile1(v::Vector{Float64}, q::Real)
    n = length(v)
    n == 0 && return 0.0
    s = sort(v)
    pos = clamp(1.0 + Float64(q) * (n - 1), 1.0, Float64(n))
    lo = floor(Int, pos)
    hi = min(lo + 1, n)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])
end

# Multichannel non-blind restoration with a SHARED image x and per-channel
# kernels: the data fidelity is the SUM over channels, so nch independent
# noise realizations of the same kernel raise the effective SNR ~ sqrt(nch).
function decon_hq2_multi(ys::Vector{Matrix{Float64}},
                         ks::Vector{Matrix{Float64}};
                         alpha::Float64 = 0.8, lam::Float64 = 2e-4,
                         iters::Int = 8, cg_iters::Int = 30,
                         epsw::Float64 = 0.03)
    nch = length(ys)
    nch >= 1 || error("decon_hq2_multi needs at least one channel")
    length(ks) == nch || error("decon_hq2_multi: one kernel per channel")
    n1, n2 = size(ys[1])
    absK2s = [abs2.(fft2_(embed_kernel2(ks[s], n1, n2))) for s in 1:nch]
    b = zeros(Float64, n1, n2)
    for s in 1:nch
        b .+= real.(ifft2_(conj.(fft2_(embed_kernel2(ks[s], n1, n2))) .* fft2_(ys[s])))
    end
    Dx, Dy = grad_mults(n1, n2)
    x = copy(ys[1])
    for _ in 1:iters
        gx, gy = circgrad(x)
        w1 = (gx .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
        w2 = (gy .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
        x = _hq_cg_multi(x, b, absK2s, Dx, Dy, lam, w1, w2, cg_iters)
    end
    return x
end

# Multiframe blind deconvolution: several frames share x but each has its
# OWN kernel. Bootstrap: use EVERY frame once as the shock-filter source
# (the "sharpest frame" heuristic misfires - the frame with the best
# shock output is data-dependent, not necessarily the least blur), solve
# all kernels by radial-band LS against that shock image, restore x
# jointly, and keep the candidate with the lowest joint energy. Then
# E-guarded refit rounds: shock the current x, refresh all kernels,
# restore again, keep a round only if it LOWERS the joint energy.
# HONEST LIMIT: refits are not guaranteed to help (the shock of a
# slightly ringing x feeds kernel LS errors back); the guard keeps the
# scheme monotone in E but the win over the bootstrap is data-dependent.
function multiframe_bd2(ys::Vector{Matrix{Float64}}; kh::Int = 15, kw::Int = 15,
                        alpha::Float64 = 0.8, lam::Float64 = 3e-4,
                        shock_iters::Int = 12, rmax::Float64 = 0.7,
                        iters::Int = 4, refit_rounds::Int = 1)
    nch = length(ys)
    nch >= 2 || error("multiframe_bd2 needs at least 2 frames")
    bestE = Inf
    bestx = similar(ys[1])
    bestk = Vector{Matrix{Float64}}(undef, nch)
    for c in 1:nch
        uc = shock2(ys[c]; iters = shock_iters)
        ks = [kernel_ls2(ys[s], uc, kh, kw; rmax = rmax) for s in 1:nch]
        x = decon_hq2_multi(ys, ks; alpha = alpha, lam = lam, iters = iters)
        E = _joint_energy2(ys, x, ks, lam)
        if E < bestE
            bestE = E
            bestx .= x
            bestk = [copy(k) for k in ks]
        end
    end
    # E-guarded refit rounds: a refit is only kept if it lowers the
    # joint energy (measured: shock-of-x refits can degrade a good
    # bootstrap solution; the guard makes the scheme monotone in E)
    x = bestx
    ks = bestk
    Ecur = bestE
    for _ in 1:refit_rounds
        xs = shock2(_clamp01!(copy(x)); iters = 4)
        ks2 = [kernel_ls2(ys[s], xs, kh, kw; rmax = rmax) for s in 1:nch]
        x2 = decon_hq2_multi(ys, ks2; alpha = alpha, lam = lam, iters = max(iters, 8))
        E2 = _joint_energy2(ys, x2, ks2, lam)
        if E2 < Ecur
            Ecur = E2
            x = x2
            ks = ks2
        else
            break
        end
    end
    return x, ks
end

# Kernel-space restart search: the alternating schemes converge to
# whichever basin the kernel init selects; a delta init is a trivial
# fixed point and a single small-gaussian init lands in a size-dependent
# local minimum (measured on a sigma-4 blur at 30 dB: default init
# kcorr 0.46, init at the true scale kcorr 1.00). This function runs the
# blind scheme from several kernel INIT SCALES and returns the full
# candidate pool: (sigma, x, k, E) per init, E = joint energy of the
# deconvolved candidate.
# HONEST LIMIT (measured, and the reason this returns a POOL instead of
# a single "best"): data-fit energies (with or without the prior) rank
# undersized candidates FIRST. The near-delta candidate (x ~ y, k ~
# delta) has mse ~ 0, and an under-deconvolved x with an undersized
# kernel consistently beats the true kernel on mse. Single-number
# selection by energy is therefore NOT trustworthy; a spread or support
# guard can reject delta-ish candidates but cannot order the rest. Use
# the pool: pick by prior knowledge of the blur scale, or refine
# candidates with a pyramid run.
function restart_search2(y::Matrix{Float64}; kh::Int = 25, kw::Int = 25,
                          sigmas = (0.5, 1.5, 2.5, 4.0), iters::Int = 60,
                          method::Symbol = :ibd,
                          alpha::Float64 = 1.0, lam::Float64 = 5e-4)
    rad = max(1, min(div(kh - 1, 2), div(kw - 1, 2)))
    pool = Vector{Tuple{Float64,Matrix{Float64},Matrix{Float64},Float64}}()
    for s in sigmas
        k0 = gaussian_kernel2(Float64(s); radius = rad)
        if method == :ibd
            x, k = ibd_rl2(y; kh = kh, kw = kw, iters = iters, kinit = k0)
        elseif method == :amtv
            x, k = am_tv2(y; kh = kh, kw = kw, iters = 5, kinit = k0)
        else
            error("restart_search2: unknown method $(method)")
        end
        xr = decon_hq2(y, k; alpha = alpha, lam = lam, iters = 8, cg_iters = 25)
        E = _joint_energy2([y], xr, [k], lam)
        push!(pool, (Float64(s), xr, k, E))
    end
    return pool
end

# -----------------------------------------------------------------------------
# Section 9: hand-written I/O - 24-bit BMP reader/writer, SVG figures
# -----------------------------------------------------------------------------

function _w16le(io::IO, v::Integer)
    write(io, UInt8(v & 0xff), UInt8((v >> 8) & 0xff))
    return nothing
end

function _w32le(io::IO, v::Integer)
    write(io, UInt8(v & 0xff), UInt8((v >> 8) & 0xff),
          UInt8((v >> 16) & 0xff), UInt8((v >> 24) & 0xff))
    return nothing
end

function _rd16le(d::Vector{UInt8}, off::Int)
    return Int(d[off]) + 256 * Int(d[off+1])
end

function _rd32le(d::Vector{UInt8}, off::Int)
    return Int(d[off]) + 256 * Int(d[off+1]) + 65536 * Int(d[off+2]) +
           16777216 * Int(d[off+3])
end

# write a grayscale Float64 image (values in [0,1]) as a 24-bit BMP
function bmp_write(path::String, img::Matrix{Float64})
    h, w = size(img)
    rowpad = (4 - (3 * w) % 4) % 4
    datasize = (3 * w + rowpad) * h
    filesize = 54 + datasize
    open(path, "w") do io
        write(io, UInt8('B'), UInt8('M'))
        _w32le(io, filesize)
        _w32le(io, 0)
        _w32le(io, 54)
        _w32le(io, 40)          # DIB header size
        _w32le(io, w)
        _w32le(io, h)
        _w16le(io, 1)           # planes
        _w16le(io, 24)          # bpp
        _w32le(io, 0)           # compression
        _w32le(io, datasize)
        _w32le(io, 2835)        # ppm x
        _w32le(io, 2835)        # ppm y
        _w32le(io, 0)           # colors used
        _w32le(io, 0)           # colors important
        for i in h:-1:1          # bottom-up rows
            for j in 1:w
                v = UInt8(round(Int, clamp(img[i, j], 0.0, 1.0) * 255))
                write(io, v, v, v)   # B, G, R
            end
            for _ in 1:rowpad
                write(io, UInt8(0))
            end
        end
    end
    return nothing
end

# read a 24-bit uncompressed BMP; returns a grayscale Float64 image in [0,1]
function bmp_read(path::String)
    d = read(path)
    length(d) >= 54 || error("bmp_read: file too small")
    (d[1] == UInt8('B') && d[2] == UInt8('M')) || error("bmp_read: not a BMP")
    pixoff = _rd32le(d, 11)
    hsz = _rd32le(d, 15)
    w = _rd32le(d, 19)
    h = _rd32le(d, 23)
    topdown = false
    if h < 0
        h = -h
        topdown = true
    end
    bpp = _rd16le(d, 29)
    comp = _rd32le(d, 31)
    (bpp == 24 && comp == 0) ||
        error("bmp_read: only 24-bit uncompressed BMP supported (got bpp=$bpp comp=$comp)")
    rowpad = (4 - (3 * w) % 4) % 4
    rowsize = 3 * w + rowpad
    img = zeros(Float64, h, w)
    for r in 0:h-1
        base = pixoff + r * rowsize
        i = topdown ? (r + 1) : (h - r)
        off = base
        for j in 1:w
            b = Int(d[off+1])
            g = Int(d[off+2])
            rr = Int(d[off+3])
            img[i, j] = (0.299 * rr + 0.587 * g + 0.114 * b) / 255.0
            off += 3
        end
    end
    return img
end

const _PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
                  "#e7961c", "#7f7f7f"]

# multi-panel 1-D trace figure.
# panels = [(title, [(label, color, data), ...]), ...]
function svg_traces(path::String, panels;
                    width::Int = 1000, pheight::Int = 160,
                    left::Int = 100, right::Int = 30, top::Int = 46,
                    gap::Int = 42)
    np = length(panels)
    height = top + np * (pheight + gap) + 6
    boxw = width - left - right
    parts = String[]
    push!(parts, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    push!(parts, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$width\" " *
                 "height=\"$height\" viewBox=\"0 0 $width $height\" " *
                 "font-family=\"Helvetica,Arial,sans-serif\">")
    push!(parts, "<rect x=\"0\" y=\"0\" width=\"$width\" height=\"$height\" fill=\"#ffffff\"/>")
    y0 = top
    for (title, series) in panels
        lo = Inf
        hi = -Inf
        maxlen = 2
        for (_, _, data) in series
            for v in data
                isfinite(v) || continue
                v < lo && (lo = v)
                v > hi && (hi = v)
            end
            length(data) > maxlen && (maxlen = length(data))
        end
        hi <= lo && (hi = lo + 1.0)
        padv = 0.08 * (hi - lo)
        lo -= padv
        hi += padv
        sx(i) = left + (i - 1) / (maxlen - 1) * boxw
        sy(v) = y0 + pheight - (v - lo) / (hi - lo) * pheight
        push!(parts, "<rect x=\"$left\" y=\"$y0\" width=\"$boxw\" height=\"$pheight\" fill=\"#ffffff\" stroke=\"#999999\"/>")
        push!(parts, "<text x=\"$left\" y=\"$(y0-12)\" font-size=\"15\" fill=\"#222222\">$title</text>")
        if lo < 0 < hi
            zy = @sprintf("%.1f", sy(0.0))
            push!(parts, "<line x1=\"$left\" y1=\"$zy\" x2=\"$(left+boxw)\" y2=\"$zy\" stroke=\"#cccccc\" stroke-dasharray=\"4,4\"/>")
        end
        for (label, color, data) in series
            pts = String[]
            for i in 1:length(data)
                v = data[i]
                isfinite(v) || continue
                push!(pts, @sprintf("%.1f,%.1f", sx(i), sy(clamp(v, lo, hi))))
            end
            push!(parts, "<polyline fill=\"none\" stroke=\"$color\" stroke-width=\"1.4\" points=\"$(join(pts, " "))\"/>")
        end
        # legend
        for (kk, (label, color, _)) in enumerate(series)
            lx = left + 10 + (kk - 1) * 190
            ly = y0 + 16
            push!(parts, "<line x1=\"$lx\" y1=\"$ly\" x2=\"$(lx+18)\" y2=\"$ly\" stroke=\"$color\" stroke-width=\"3\"/>")
            push!(parts, "<text x=\"$(lx+24)\" y=\"$(ly+4)\" font-size=\"12\" fill=\"#333333\">$label</text>")
        end
        y0 += pheight + gap
    end
    push!(parts, "</svg>")
    open(path, "w") do io
        write(io, join(parts, "\n"))
    end
    return nothing
end

# kernel gallery figure: each kernel drawn as a grayscale pixel grid
# ks = [(title, kernel matrix), ...]
function svg_kernel(path::String, ks; cell::Int = 12)
    nker = length(ks)
    widths = [size(k, 2) * cell for (_, k) in ks]
    heights = [size(k, 1) * cell for (_, k) in ks]
    totalw = 10 + sum(widths) + 50 * nker
    totalh = 34 + maximum(heights) + 10
    parts = String[]
    push!(parts, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    push!(parts, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$totalw\" height=\"$totalh\" viewBox=\"0 0 $totalw $totalh\" font-family=\"Helvetica,Arial,sans-serif\">")
    push!(parts, "<rect x=\"0\" y=\"0\" width=\"$totalw\" height=\"$totalh\" fill=\"#ffffff\"/>")
    x = 25
    for (title, k) in ks
        kh, kw = size(k)
        push!(parts, "<text x=\"$x\" y=\"20\" font-size=\"13\" fill=\"#222222\">$title</text>")
        mx = maximum(k)
        mx <= 0 && (mx = 1.0)
        for i in 1:kh, j in 1:kw
            v = clamp(k[i, j] / mx, 0.0, 1.0)
            g = round(Int, 255 * (1.0 - v))
            col = "rgb($g,$g,$g)"
            xx = x + (j - 1) * cell
            yy = 30 + (i - 1) * cell
            push!(parts, "<rect x=\"$xx\" y=\"$yy\" width=\"$cell\" height=\"$cell\" fill=\"$col\" stroke=\"#dddddd\" stroke-width=\"0.5\"/>")
        end
        x += kw * cell + 50
    end
    push!(parts, "</svg>")
    open(path, "w") do io
        write(io, join(parts, "\n"))
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Section 10: synthetic data generators (demos + selftest)
# -----------------------------------------------------------------------------

function synth_image2(n::Int)
    img = zeros(Float64, n, n)
    m = max(6, div(n, 14))
    # smooth background
    for i in 1:n, j in 1:n
        img[i, j] = 0.30 + 0.06 * cos(2 * pi * (i - 1) / n) +
                    0.05 * sin(2 * pi * (j - 1) / (0.8 * n))
    end
    # gaussian blob (smooth feature)
    ci = round(Int, 0.22 * n)
    cj = round(Int, 0.75 * n)
    sig = 0.05 * n
    for i in 1:n, j in 1:n
        img[i, j] += 0.35 * exp(-((i - ci)^2 + (j - cj)^2) / (2 * sig^2))
    end
    # sinusoidal texture patch
    i1 = round(Int, 0.15 * n)
    i2 = round(Int, 0.45 * n)
    j1 = round(Int, 0.60 * n)
    j2 = round(Int, 0.90 * n)
    for i in i1:i2, j in j1:j2
        img[i, j] += 0.10 * sin(2 * pi * j / (n / 9))
    end
    # disks
    function disk!(cx, cy, r, val)
        for i in max(1, round(Int, cx - r)):min(n, round(Int, cx + r))
            for j in max(1, round(Int, cy - r)):min(n, round(Int, cy + r))
                if hypot(i - cx, j - cy) <= r
                    img[i, j] = val
                end
            end
        end
    end
    disk!(round(Int, 0.32 * n), round(Int, 0.34 * n), 0.13 * n, 0.85)
    disk!(round(Int, 0.72 * n), round(Int, 0.68 * n), 0.07 * n, 0.75)
    # rectangle
    for i in round(Int, 0.55 * n):round(Int, 0.80 * n)
        for j in max(m, round(Int, 0.18 * n)):min(n - m, round(Int, 0.45 * n))
            img[i, j] = 0.55
        end
    end
    # thin bright bar (edge resolution test)
    ib = round(Int, 0.62 * n)
    for i in ib:min(n, ib + 1), j in m:n-m
        img[i, j] = 0.92
    end
    # small dots
    r3 = max(2.0, 0.03 * n)
    disk!(round(Int, 0.75 * n), round(Int, 0.25 * n), r3, 0.9)
    disk!(round(Int, 0.86 * n), round(Int, 0.34 * n), r3, 0.9)
    disk!(round(Int, 0.64 * n), round(Int, 0.17 * n), r3, 0.9)
    return _clamp01!(img)
end

# sparse reflectivity with a few strong events (deterministic part + rng)
function synth_reflectivity(N::Int; seed_events = true)
    x = zeros(Float64, N)
    pos = [40, 75, 130, 190, 240, 300, 355, 410, 470]
    amp = [1.5, -1.0, 2.0, 0.7, -1.2, 0.9, 1.8, -0.6, 1.1]
    for (p, a) in zip(pos, amp)
        if p <= N
            x[p] = a
        end
    end
    if seed_events
        # small random background spikes
        for i in 5:N-5
            if rand() < 0.015
                x[i] += 0.25 * randn()
            end
        end
    end
    return x
end

function add_noise!(v::AbstractVector{Float64}, snr_db::Real)
    sv = sqrt(sum(v .^ 2) / length(v))
    sig = sv / (10^(snr_db / 20))
    v .+= sig .* randn(length(v))
    return v
end

function add_noise2!(img::Matrix{Float64}, snr_db::Real)
    sv = sqrt(sum(img .^ 2) / length(img))
    sig = sv / (10^(snr_db / 20))
    img .+= sig .* randn(size(img))
    return _clamp01!(img)
end


# Constant-Q synthetic: sparse reflectivity convolved with a broadband
# damped-cosine wavelet, then attenuated with the t*-f* law A(f,t) ~
# exp(-pi f t / Q) applied per STFT frame (win 128, hop 64) so generator
# and estimator share the same model. Deterministic for a fixed seed.
function q_attenuated_trace(N::Int, fs::Real, f0::Real, Q::Real;
                            p::Float64 = 0.02, decay::Float64 = 4.0,
                            win::Int = 128, seed::Int = 42)
    Random.seed!(seed)
    # p = spike probability per sample; p = 1.0 gives dense white
    # reflectivity, the regime where single-trace Q fitting is stable
    x = zeros(Float64, N)
    @inbounds for n in 1:N
        if rand() < p
            x[n] += randn()
        end
    end
    nw = min(N, 101)
    h = damped_cosine_wavelet(f0, decay, 1.0 / Float64(fs), nw)
    y = conv_causal(x, h)
    resize!(y, N)
    hop = max(1, div(win, 2))
    F, w = _stft_frames(y, win, hop)
    nf = size(F, 2)
    # apply exp(-pi f t / Q) per STFT frame; frequencies above Nyquist
    # are reflected so the attenuation stays conjugate-symmetric (a real
    # trace must attenuate a bin and its mirror identically, otherwise the
    # re-synthesized signal keeps a large unattenuated mirror component)
    fsf = Float64(fs)
    fnyq = fsf / 2
    @inbounds for t in 1:nf, k in 1:win
        f = (k - 1) * fsf / win
        if f > fnyq
            f = fsf - f
        end
        tc = (t - 1) * hop + div(win, 2)
        F[k, t] *= exp(-pi * f * tc / (Float64(fs) * Float64(Q)))
    end
    out = zeros(Float64, N + win)
    wsum = zeros(Float64, N + win)
    @inbounds for t in 1:nf
        fr = real.(ifft1(F[:, t]))
        i0 = (t - 1) * hop + 1
        for i in 1:win
            gi = i0 + i - 1
            out[gi] += fr[i] * w[i]
            wsum[gi] += w[i] * w[i]
        end
    end
    o = zeros(Float64, N)
    @inbounds for i in 1:N
        if wsum[i] >= 0.25
            o[i] = out[i] / wsum[i]
        end
    end
    return o, x
end

# -----------------------------------------------------------------------------
# Section 11: helpers for validation
# -----------------------------------------------------------------------------

# lag-1 autocorrelation (|normalized|), the classic whitening indicator.
# NOTE: meaningless as a whitening metric for strictly band-limited data
# (in-band white noise keeps a sinc-shaped lag profile); use it on raw
# windows and pair it with spectral flatness.
function _lag1_(v::AbstractVector{<:Real})
    n = length(v)
    n < 3 && return 0.0
    a = collect(v[1:n-1])
    b = collect(v[2:n])
    d = sqrt(sum(abs2, a) * sum(abs2, b))
    return d > 0 ? abs(sum(a .* b)) / d : 0.0
end

function sigcorr(a::Vector{Float64}, b::Vector{Float64})
    na = norm(a)
    nb = norm(b)
    (na < 1e-30 || nb < 1e-30) && return 0.0
    return dot(a, b) / (na * nb)
end

function sigcorr2(a::Matrix{Float64}, b::Matrix{Float64})
    na = norm(a)
    nb = norm(b)
    (na < 1e-30 || nb < 1e-30) && return 0.0
    return dot(a, b) / (na * nb)
end

function kurt(v::Vector{Float64})
    s2 = sum(abs2, v)
    s4 = sum(v .^ 4)
    return length(v) * s4 / (s2 * s2)
end

function topk_pos(v::Vector{Float64}, k::Int)
    idx = sortperm(abs.(v), rev = true)[1:min(k, length(v))]
    return sort(idx)
end

# best linear shift of b against a (lags -maxlag..maxlag), returns (lag, corr)
function best_shift(a::Vector{Float64}, b::Vector{Float64}, maxlag::Int)
    bestlag = 0
    bestc = -Inf
    for lag in -maxlag:maxlag
        bb = zeros(Float64, length(b))
        for i in 1:length(b)
            j = i - lag
            if 1 <= j <= length(b)
                bb[i] = b[j]
            end
        end
        c = sigcorr(a, bb)
        if c > bestc
            bestc = c
            bestlag = lag
        end
    end
    return bestlag, bestc
end

# -----------------------------------------------------------------------------
# Section 12: demos
# -----------------------------------------------------------------------------

function demo_seismic(outdir::String = "demo_seismic_out")
    mkpath(outdir)
    Random.seed!(20261114)
    N = 500
    dt = 0.004
    nfft = 1024
    println("=== BlindDecon seismic demo (N = $N samples, dt = $(1000*dt) ms) ===")
    # broadband minimum-phase wavelet (damped cosine, 30 Hz)
    h = damped_cosine_wavelet(30.0, 125.0, dt, 41)
    # reflectivity + observed trace
    x = synth_reflectivity(N)
    y = conv_causal(x, h)
    add_noise!(y, 30.0)
    # --- blind methods
    x_pred, a_pred = prediction_decon(y; gap = 1, nfil = 40)
    x_med, a_med, hist_med = med_decon(y; ninv = 41, iters = 200)
    x_sp, h_sp = sparse_bd1(y; L = 15, iters = 15)
    lag_med, c_med = best_shift(x, x_med, 40)
    # --- wavelet estimation
    xs = [randn(N) for _ in 1:12]
    ys = [add_noise!(conv_causal(xe, h), 30.0) for xe in xs]
    h_ens = wavelet_from_ensemble(ys, 41; phase = :min)
    amp_hom, _ = homomorphic_wavelet(y, 30; nfft = nfft)
    Htrue = abs.(fft1(_zeropad(h, nfft)))
    c_hom = sigcorr(amp_hom, Htrue)
    println()
    @printf("prediction decon : corr(out, true reflectivity) = %.3f\n", sigcorr(x_pred, x))
    @printf("MED (Wiggins)   : corr after shift %d = %.3f, kurtosis %.1f -> %.1f\n",
            lag_med, c_med, kurt(y), kurt(x_med))
    @printf("sparse_bd1      : corr(out, true) = %.3f, wavelet fit corr = %.3f\n",
            sigcorr(x_sp, x), sigcorr(h_sp, h[1:length(h_sp)]))
    @printf("wavelet ensemble : corr(est, true) = %.3f (12 traces)\n", sigcorr(h_ens, h))
    @printf("homomorphic     : amp-spectrum corr = %.3f (lifter 30)\n", c_hom)
    # --- constant-Q attenuation + non-stationary whitening (v0.2.0)
    Random.seed!(2002)
    yq, xq = q_attenuated_trace(2048, 100.0, 3.0, 80.0; p = 0.02, seed = 42)
    Qh = q_estimate(yq, 0.01; win = 128, fmin = 1.0, fmax = 8.0, amin = 0.05)
    nsq = nonstat_decon(yq; win = 128, dfloor = 1e-3, sm = 2)
    pdq, _ = prediction_decon(yq; gap = 1, nfil = 60)
    println()
    println("constant-Q synth (Q = 80, fs = 100, f0 = 3 Hz, sparse spikes):")
    @printf("  q_estimate      : Qhat = %.1f (true 80; ~12%% STFT synthesis-blend bias)\n", Qh)
    @printf("  late-half lag1  : input %.3f, nonstat %.3f, PEF %.3f\n",
            _lag1_(yq[1025:end]), _lag1_(nsq[1025:end]), _lag1_(pdq[1025:end]))
    println("  (non-stationary whitening tracks the Q decay; the global PEF cannot)")
    println()
    # --- figures
    panels = [
        ("reflectivity and observed trace",
         [("true reflectivity", "#1f77b4", x), ("observed trace y", "#d62728", y)]),
        ("gapped prediction-error (spiking) deconvolution",
         [("true reflectivity", "#1f77b4", x), ("PEF output", "#d62728", x_pred)]),
        ("minimum-entropy deconvolution (Wiggins MED)",
         [("true reflectivity", "#1f77b4", x), ("MED output", "#d62728", x_med)]),
        ("alternating sparse blind deconvolution",
         [("true reflectivity", "#1f77b4", x), ("sparse output", "#d62728", x_sp)]),
    ]
    svg_traces(joinpath(outdir, "seismic_decon.svg"), panels)
    hsp_pad = vcat(h_sp, zeros(Float64, 41 - length(h_sp)))
    panels_w = [
        ("wavelet estimates (arbitrary amplitude scale)",
         [("true min-phase", "#1f77b4", h), ("ensemble spectral est.", "#d62728", h_ens),
          ("sparse-fit est. (first 15 taps)", "#2ca02c", hsp_pad)]),
    ]
    svg_traces(joinpath(outdir, "seismic_wavelets.svg"), panels_w)
    println("figures written to $outdir")
    return nothing
end

function demo_image(bmpin::String = "", outdir::String = "demo_image_out")
    mkpath(outdir)
    Random.seed!(20261114)
    if bmpin != "" && isfile(bmpin)
        img = bmp_read(bmpin)
        println("=== BlindDecon image demo: loaded $(size(img,1))x$(size(img,2)) from $bmpin ===")
    else
        img = synth_image2(96)
        println("=== BlindDecon image demo: synthetic 96x96 test image ===")
    end
    h0, w0 = size(img)
    ktrue = gaussian_kernel2(1.6; radius = 4)
    pad = 16
    n1 = nextpow2_(h0 + 2 * pad)
    n2 = nextpow2_(w0 + 2 * pad)
    yp = pad_replicate2(img, n1, n2)
    y = cconv2(yp, ktrue)
    add_noise2!(y, 40.0)
    println("padded to $n1 x $n2, gaussian blur sigma = 1.6 (9x9), noise 40 dB")
    t0 = time()
    x_ibd, k_ibd = ibd_rl2(y; kh = 9, kw = 9, iters = 40)
    t_ibd = time() - t0
    t0 = time()
    x_tv, k_tv = am_tv2(y; kh = 9, kw = 9, iters = 5, lam = 5e-4, inner = 4,
                        cg_iters = 25)
    t_tv = time() - t0
    t0 = time()
    x_xj, k_xj, u_sh = xujia2(y; kh = 13, kw = 13)
    t_xj = time() - t0
    yc = crop_center2(y, h0, w0)
    xi = _clamp01!(crop_center2(x_ibd, h0, w0))
    xt = _clamp01!(crop_center2(x_tv, h0, w0))
    xj = _clamp01!(crop_center2(x_xj, h0, w0))
    ush = _clamp01!(crop_center2(u_sh, h0, w0))
    p_blur = psnr(img, yc)
    p_ibd = psnr(img, xi)
    p_tv = psnr(img, xt)
    p_xj = psnr(img, xj)
    println()
    @printf("%-22s %8s %8s %10s %8s\n", "method", "PSNR", "gain", "kcorr", "time")
    @printf("%-22s %8.2f %8.2f %10s %8s\n", "blurred (input)", p_blur, 0.0, "-", "-")
    @printf("%-22s %8.2f %8.2f %10.3f %8.2f\n", "IBD-RL (Ayers-Dainty)", p_ibd, p_ibd - p_blur, kernel_corr(ktrue, k_ibd), t_ibd)
    @printf("%-22s %8.2f %8.2f %10.3f %8.2f\n", "TV-AM (Chan-Wong)", p_tv, p_tv - p_blur, kernel_corr(ktrue, k_tv), t_tv)
    @printf("%-22s %8.2f %8.2f %10.3f %8.2f\n", "Xu-Jia + hyper-Lapl.", p_xj, p_xj - p_blur, kernel_corr(ktrue, k_xj), t_xj)
    println()
    bmp_write(joinpath(outdir, "0_original.bmp"), img)
    bmp_write(joinpath(outdir, "1_blurred.bmp"), yc)
    bmp_write(joinpath(outdir, "2_ibd_rl.bmp"), xi)
    bmp_write(joinpath(outdir, "3_tv_am.bmp"), xt)
    bmp_write(joinpath(outdir, "4_xujia.bmp"), xj)
    bmp_write(joinpath(outdir, "5_shock_input.bmp"), ush)
    svg_kernel(joinpath(outdir, "kernels.svg"),
               [("true kernel", ktrue), ("IBD-RL", k_ibd),
                ("TV-AM", k_tv), ("Xu-Jia", k_xj)])

    # --- v0.2.0: large blur, coarse-to-fine pyramid vs single scale (128^2) ---
    println()
    println("=== large blur: coarse-to-fine pyramid (128x128, sigma = 4, kernel 25x25, 40 dB) ===")
    Random.seed!(7)
    img2 = synth_image2(128)
    kp4 = gaussian_kernel2(4.0; radius = 12)
    y2 = cconv2(img2, kp4)
    add_noise2!(y2, 40.0)
    t0 = time()
    x_py, k_py = blind_pyramid2(y2; method = :ibd, kh = 25, kw = 25,
                               levels = 3, iters = 60)
    t_py = time() - t0
    t0 = time()
    x_pl, k_pl = ibd_rl2(y2; kh = 25, kw = 25, iters = 60)
    t_pl = time() - t0
    xpy = _clamp01!(copy(x_py))
    xpl = _clamp01!(copy(x_pl))
    @printf("%-22s %8s %8s %10s %8s\n", "method", "PSNR", "gain", "kcorr", "time")
    @printf("%-22s %8.2f %8.2f %10s %8s\n", "blurred (input)", psnr(img2, y2), 0.0, "-", "-")
    @printf("%-22s %8.2f %8.2f %10.3f %8.2f\n", "pyramid IBD-RL",
            psnr(img2, xpy), psnr(img2, xpy) - psnr(img2, y2),
            kernel_corr(kp4, k_py), t_py)
    @printf("%-22s %8.2f %8.2f %10.3f %8.2f\n", "single-scale IBD-RL",
            psnr(img2, xpl), psnr(img2, xpl) - psnr(img2, y2),
            kernel_corr(kp4, k_pl), t_pl)
    println("  (single-scale leaves the kernel in a size-dependent local minimum;")
    println("   the pyramid warm-starts the fine level from the coarse estimate)")
    bmp_write(joinpath(outdir, "6_pyramid.bmp"), xpy)
    bmp_write(joinpath(outdir, "7_singlescale.bmp"), xpl)

    # --- v0.2.0: multiframe blind deconvolution (two kernels, one image) ---
    println()
    println("=== multiframe: two frames, DIFFERENT kernels (gaussian 1.6 + motion 6/0, 30 dB) ===")
    Random.seed!(21)
    kmo = motion_kernel2(6.0, 0.0)
    kga = gaussian_kernel2(1.6; radius = 4)
    y3a = cconv2(img2, kga)
    y3b = cconv2(img2, kmo)
    add_noise2!(y3a, 30.0)
    add_noise2!(y3b, 30.0)
    yfr = [y3a, y3b]
    x_mf, k_mf = multiframe_bd2(yfr; kh = 15, kw = 15)
    _, k_sj2, _ = xujia2(yfr[1]; kh = 15, kw = 15)
    @printf("multiframe kernel corrs : gaussian %.3f, motion %.3f\n",
            kernel_corr(kga, k_mf[1]), kernel_corr(kmo, k_mf[2]))
    @printf("single-frame xujia     : gaussian-frame kernel corr %.3f\n",
            kernel_corr(kga, k_sj2))
    println("  (joint estimation uses both frames; the recovered IMAGE is not")
    println("   necessarily better in PSNR than a lucky single-frame restore,")
    println("   but the KERNELS are - see README honest limits)")
    _sc = _clamp01!(copy(x_mf))
    bmp_write(joinpath(outdir, "8_multiframe.bmp"), _sc)
    println("images + kernels.svg written to $outdir")
    return nothing
end


# -----------------------------------------------------------------------------
# Section 12b (v0.2.0): real-data demo - InSight SEIS marsquake S0105a
# -----------------------------------------------------------------------------

# Real data demo on the NASA InSight SEIS VBB event window of marsquake
# S0105a (sol 105, MQS catalog v14 pick 2019-03-14T21:03:03.6Z = 1191.574 s
# after file start; data: PDS Geosciences, GeoCSV, 20 sps, 48000 samples:
# a 2400 s crop of the PDS bundle, covering the event, its coda, and the
# lander-activity transient at t = 1997 s).
# HONESTY NOTES baked into the prints: (a) the strongest in-band transient
# of the record is lander activity around t = 2000 s, NOT the quake, so a
# naive transient scan misfires; (b) the event is real but modest; (c)
# band limitation is information-theoretic: PEF whitens what the band
# carries, it cannot invent frequencies that never reached the sensor.
function demo_mars(csvpath::String = "", outdir::String = "demo_mars_out")
    if isempty(csvpath)
        csvpath = joinpath(@__DIR__, "data",
                           "xb.elyse.02.bhu.S0105a.mqs2019fddj.1.a.csv")
    end
    if !isfile(csvpath)
        println("demo_mars: data file not found: ", csvpath)
        println("  (bundle path is data/xb.elyse.02.bhu.S0105a.mqs2019fddj.1.a.csv)")
        return 1
    end
    mkpath(outdir)
    y, fs, t0 = parse_geocsv(csvpath)
    N = length(y)
    tp = 1191.574   # MQS v14 P pick, seconds after file start
    println("InSight SEIS demo: S0105a (sol 105, XB ELYSE 02 BHU, 20 sps)")
    println("  samples: ", N, "   start: ", t0, "   P pick at ", tp, " s")
    lag1(v) = abs(sum(v[1:end-1] .* v[2:end]) /
                  sqrt(sum(v[1:end-1] .^ 2) * sum(v[2:end] .^ 2)))
    # event window around the P pick
    i1 = max(1, round(Int, (tp - 20.6) * fs))
    w = y[i1:i1+511]
    wb = bandpass1(w, fs, 1.5, 4.0)
    nmad = mad_noise1(wb)
    brms = sqrt(sum(abs2, wb) / length(wb))
    @printf("  event window (512 smp): band 1.5-4 Hz rms %.1f, MAD sigma %.1f, in-band rms/sigma %.1f\n",
            brms, nmad, brms / nmad)
    # honesty: strongest in-band transient of the whole record
    pad = 2000   # the GeoCSV head is a pad plateau, window 1 would be an artifact
    yb = bandpass1(y, fs, 1.5, 4.0)
    win = 512; hop = 512
    mlo = cld(pad, hop) + 1
    mhi = floor(Int, (N - win) / hop)
    bestr, bestt = 0.0, 0.0
    for m in mlo:mhi
        i0 = (m - 1) * hop + 1
        r = sqrt(sum(abs2, yb[i0:i0+win-1]) / win)
        if r > bestr
            bestr = r
            bestt = (i0 - 1) / fs
        end
    end
    @printf("  strongest in-band transient of the record: rms %.1f at t = %.0f s (NOT the quake; lander activity)\n",
            bestr, bestt)
    # whitening of the raw event window
    l_before = lag1(w)
    sf_before = _specflat1(w)
    pd, _ = prediction_decon(w; gap = 1, nfil = 60)
    l_after = lag1(pd)
    sf_after = _specflat1(pd)
    ns = nonstat_decon(w; win = 128, dfloor = 1e-2, sm = 3)
    @printf("  raw window: lag1 %.3f, spectral flatness %.4f\n", l_before, sf_before)
    @printf("  PEF (gap 1, 60 taps): lag1 %.3f, spectral flatness %.4f\n", l_after, sf_after)
    @printf("  nonstat (win 128): lag1 %.3f, spectral flatness %.4f\n",
            lag1(ns), _specflat1(ns))
    # minimum-phase wavelet from the ensemble of bandpassed segments
    wav = begin
        nseg = fld(512 - 128, 64)
        amps = zeros(128)
        for s in 1:nseg
            seg = collect(wb[(s-1)*64+1:(s-1)*64+128])
            seg .-= sum(seg) / length(seg)
            amps .+= abs2.(fft1(seg))
        end
        amps ./= nseg
        minphase_from_amp(sqrt.(amps))
    end
    println("  min-phase wavelet from event ensemble: ",
            length(wav), " taps, first 4 = ",
            round.(wav[1:min(4, end)], digits = 3))
    # hourly band profile for panel 1 (block rms, 1000 bins)
    prof = begin
        nseg = 1000
        seglen = max(1, fld(N, nseg))
        p = Float64[]
        for s in 0:(fld(N, seglen) - 1)
            i0 = s * seglen + 1
            i1e = min(N, i0 + 4 * seglen - 1)
            i1e > N && break
            push!(p, sqrt(sum(abs2, yb[i0:i1e]) / (i1e - i0 + 1)))
        end
        p
    end
    svg_traces(joinpath(outdir, "mars_s0105a.svg"),
               [("full-trace band 1.5-4 Hz rms profile (5-block groups)",
                 [("rms", "#1f77b4", prof)]),
                ("raw event window (P pick marked by the window position)",
                 [("samples", "#d62728", w)]),
                ("PEF-spiked window",
                 [("decon", "#2ca02c", pd[1:length(w)])]),
                ("nonstationary whitened window",
                 [("nonstat", "#9467bd", ns[1:length(w)])]),
                ("min-phase wavelet estimate",
                 [("wavelet", "#8c564b", wav[1:min(64, end)])])])
    println("  wrote ", joinpath(outdir, "mars_s0105a.svg"))
    println("demo_mars done")
    return 0
end


# -----------------------------------------------------------------------------
# Section 13: selftest (all seeds fixed, PASS lines byte-identical across runs)
# -----------------------------------------------------------------------------

const _ST_N = Ref(0)
const _ST_FAIL = Ref(0)

function chk(desc::String, cond::Bool, val::String = "")
    _ST_N[] += 1
    if cond
        @printf("PASS check %2d: %s %s\n", _ST_N[], desc, val)
    else
        _ST_FAIL[] += 1
        @printf("FAIL check %2d: %s %s\n", _ST_N[], desc, val)
    end
    return nothing
end

function selftest()
    Random.seed!(42)
    t_total = time()
    println("BlindDecon.jl selftest")
    println("======================")

    # --- FFT ---
    a = randn(100) .+ im .* randn(100)
    F = fft1(a)
    dref = zeros(ComplexF64, 100)
    for k in 1:100
        s = 0.0 + 0.0im
        for n in 1:100
            s += a[n] * cis(-2 * pi * (k - 1) * (n - 1) / 100)
        end
        dref[k] = s
    end
    e = maximum(abs.(F .- dref))
    chk("fft1 (Bluestein, n=100) vs direct DFT", e < 1e-8, @sprintf("(max err %.2e)", e))

    x37 = randn(37)
    e = maximum(abs.(ifft1(fft1(x37)) .- x37))
    chk("ifft1 roundtrip (n=37)", e < 1e-10, @sprintf("(max err %.2e)", e))
    x64 = randn(64)
    e = maximum(abs.(ifft1(fft1(x64)) .- x64))
    chk("ifft1 roundtrip (n=64 pow2)", e < 1e-10, @sprintf("(max err %.2e)", e))

    A = randn(4, 6)
    F2 = fft2_(A)
    dref2 = zeros(ComplexF64, 4, 6)
    for k1 in 1:4, k2 in 1:6
        s = 0.0 + 0.0im
        for i in 1:4, j in 1:6
            s += A[i, j] * cis(-2 * pi * ((k1 - 1) * (i - 1) / 4 + (k2 - 1) * (j - 1) / 6))
        end
        dref2[k1, k2] = s
    end
    e = maximum(abs.(F2 .- dref2))
    chk("fft2_ vs direct 2-D DFT (4x6)", e < 1e-10, @sprintf("(max err %.2e)", e))

    # --- 1-D convolution / linear algebra ---
    x50 = randn(50)
    h7 = randn(7)
    e = maximum(abs.(conv_full(x50, h7) .- conv_full_direct(x50, h7)))
    chk("conv_full (FFT) vs direct", e < 1e-9, @sprintf("(max err %.2e)", e))

    r = autocorr_full(x50, 10)[1:8]
    rhs = randn(8)
    T = zeros(8, 8)
    for i in 1:8, j in 1:8
        T[i, j] = r[abs(i - j) + 1]
    end
    e = maximum(abs.(toeplitz_solve(r, rhs) .- (T \ rhs)))
    chk("toeplitz_solve vs LU", e < 1e-8, @sprintf("(max err %.2e)", e))

    # --- Kolmogorov minimum phase ---
    w0 = ricker(25.0, 0.004, 41)
    nfft = 1024
    W = abs.(fft1(_zeropad(w0, nfft)))
    hmin = minphase_from_amp(W)
    W2 = abs.(fft1(hmin))
    mask = W .> 1e-9 .* maximum(W)
    e = maximum(abs.(W2[mask] .- W[mask]) ./ W[mask])
    frontfrac = sum(hmin[1:41] .^ 2) / sum(hmin .^ 2)
    chk("minphase_from_amp preserves |H|", e < 1e-6, @sprintf("(max rel err %.2e)", e))
    chk("minphase_from_amp front-loads energy", frontfrac > 0.90,
        @sprintf("(first-41-sample energy fraction %.3f)", frontfrac))

    # --- broadband minimum-phase wavelet for the decon checks ---
    hwav = damped_cosine_wavelet(30.0, 125.0, 0.004, 41)

    # --- prediction (spiking) deconvolution ---
    N = 600
    xr = randn(N)
    yr = conv_causal(xr, hwav)
    outp, ap = prediction_decon(yr; gap = 1, nfil = 40)
    cp = sigcorr(outp, xr)
    rr = autocorr_full(outp, 1)
    white = abs(rr[2] / rr[1])
    chk("spiking decon whitens output", white < 0.10, @sprintf("(lag-1 autocorr %.3f)", white))
    chk("spiking decon recovers reflectivity", cp > 0.90, @sprintf("(corr %.3f)", cp))

    # --- homomorphic wavelet estimation ---
    xr = randn(N)
    wzed = ricker(30.0, 0.004, 41)
    yr = conv_causal(xr, wzed)
    amp_hom, _ = homomorphic_wavelet(yr, 30)
    Htrue = abs.(fft1(_zeropad(wzed, nfft)))
    c_hom = sigcorr(amp_hom, Htrue)
    chk("homomorphic wavelet amplitude", c_hom > 0.90, @sprintf("(spectrum corr %.3f)", c_hom))

    # --- ensemble wavelet estimation ---
    xs = [randn(N) for _ in 1:12]
    ys = Vector{Vector{Float64}}()
    for xe in xs
        ye = conv_causal(xe, hwav)
        add_noise!(ye, 30.0)
        push!(ys, ye)
    end
    h_ens = wavelet_from_ensemble(ys, 41; phase = :min)
    c_ens = sigcorr(h_ens, hwav)
    chk("ensemble wavelet estimation", c_ens > 0.95, @sprintf("(corr %.3f)", c_ens))

    # --- MED (Wiggins) ---
    xsp = zeros(Float64, 500)
    for (p, a2) in zip([60, 140, 260, 400], [1.0, -1.5, 1.2, 0.8])
        xsp[p] = a2
    end
    ymed = conv_causal(xsp, hwav)
    add_noise!(ymed, 30.0)
    x_med, a_med, hist_med = med_decon(ymed; ninv = 41, iters = 200)
    mono = all(hist_med[i+1] >= hist_med[i] - 1e-12 for i in 1:length(hist_med) - 1)
    chk("MED varimax monotone", mono, @sprintf("(%d iters, V %.3f -> %.3f)",
        length(hist_med), hist_med[1], hist_med[end]))
    chk("MED sharpens (kurtosis)", kurt(x_med) > kurt(ymed),
        @sprintf("(kurtosis %.1f -> %.1f)", kurt(ymed), kurt(x_med)))
    lagm, cm = best_shift(xsp, x_med, 40)
    xal = zeros(Float64, 500)
    for i in 1:500
        j = i - lagm
        if 1 <= j <= 500
            xal[i] = x_med[j]
        end
    end
    ptrue = sort([60, 140, 260, 400])
    pest = topk_pos(xal, 4)
    pdist = maximum(abs.(sort(pest) .- ptrue))
    chk("MED recovers spike positions", pdist <= 2 && cm > 0.5,
        @sprintf("(max pos err %d, shift %d, shift-corr %.3f)", round(Int, pdist), lagm, cm))

    # --- alternating sparse blind decon ---
    xsp = zeros(Float64, 500)
    for (p, a2) in zip([50, 130, 260, 400], [1.2, -1.0, 1.5, 0.8])
        xsp[p] = a2
    end
    ysp = conv_causal(xsp, hwav)
    add_noise!(ysp, 25.0)
    x_sp, h_sp = sparse_bd1(ysp; L = 15, iters = 15)
    c_wav = sigcorr(h_sp, hwav[1:15])
    ptrue = sort([50, 130, 260, 400])
    pest = topk_pos(x_sp, 4)
    pdist = maximum(abs.(sort(pest) .- ptrue))
    chk("sparse_bd1 wavelet fit", c_wav > 0.90, @sprintf("(corr %.3f)", c_wav))
    chk("sparse_bd1 spike positions", pdist <= 1,
        @sprintf("(max pos err %d)", round(Int, pdist)))

    # --- SIMO cross-relation identification ---
    h1 = vcat(1.0, randn(7))
    h2 = vcat(1.0, randn(7))
    xin = randn(300)
    y1 = conv_causal(xin, h1)
    y2 = conv_causal(xin, h2)
    h1e, h2e = crossrel_identify(y1, y2; L = 8)
    e1 = norm(h1e .- h1) / norm(h1)
    e2 = norm(h2e .- h2) / norm(h2)
    chk("cross-relation blind ID (noiseless)", max(e1, e2) < 1e-6,
        @sprintf("(rel err %.2e / %.2e)", e1, e2))

    # --- proximal operator ---
    function _prox_grid(z, tau, alpha)
        best = 0.0
        bestc = Inf
        for u in 0.0:0.0005:(abs(z) + 2.0)
            c = 0.5 * (u - abs(z))^2 + tau * u^alpha
            if c < bestc
                bestc = c
                best = u
            end
        end
        return sign(z) * best
    end
    perr = 0.0
    for z in (-1.7, -0.9, -0.35, 0.2, 0.5, 0.95, 1.6)
        for alpha in (0.5, 0.8)
            pnum = _prox_lapower_scalar(z, 0.3, alpha)
            pgrd = _prox_grid(z, 0.3, alpha)
            perr = max(perr, abs(pnum - pgrd))
        end
    end
    chk("prox_lapower vs grid search", perr < 2e-3, @sprintf("(max err %.2e)", perr))

    # --- 2-D circular machinery ---
    x2 = randn(8, 10)
    k2 = randn(3, 5)
    emb = embed_kernel2(k2, 8, 10)
    e = maximum(abs.(crop_kernel2(emb, 3, 5) .- k2))
    chk("embed/crop kernel roundtrip", e < 1e-15, @sprintf("(max err %.2e)", e))
    e = maximum(abs.(cconv2(x2, k2) .- cconv2_direct(x2, k2)))
    chk("cconv2 (FFT) vs direct circular", e < 1e-9, @sprintf("(max err %.2e)", e))

    # --- non-blind 2-D deconvolution ---
    img = synth_image2(64)
    ktrue = gaussian_kernel2(1.5; radius = 4)
    yb = cconv2(img, ktrue)
    p_blur = psnr(img, yb)
    xw = decon_wiener2(yb, ktrue; nsr = 1e-4)
    g = psnr(img, xw) - p_blur
    chk("decon_wiener2 PSNR gain", g > 5.0, @sprintf("(gain %.2f dB)", g))
    xr_ = decon_rl2(yb, ktrue; iters = 300)
    g = psnr(img, xr_) - p_blur
    chk("decon_rl2 PSNR gain", g > 4.0, @sprintf("(gain %.2f dB)", g))
    xh = decon_hq2(yb, ktrue; alpha = 0.8, lam = 1e-4, iters = 8, cg_iters = 30)
    g = psnr(img, xh) - p_blur
    chk("decon_hq2 (alpha=0.8) PSNR gain", g > 8.0, @sprintf("(gain %.2f dB)", g))

    # --- blind 2-D deconvolution ---
    ktrue = gaussian_kernel2(1.2; radius = 3)
    yb = cconv2(img, ktrue)
    x_ibd, k_ibd = ibd_rl2(yb; kh = 7, kw = 7, iters = 40)
    kc = kernel_corr(ktrue, k_ibd)
    g = psnr(img, x_ibd) - psnr(img, yb)
    chk("ibd_rl2 kernel recovery", kc > 0.90, @sprintf("(kcorr %.3f)", kc))
    chk("ibd_rl2 PSNR gain", g > 2.0, @sprintf("(gain %.2f dB)", g))

    ktrue = gaussian_kernel2(1.5; radius = 4)
    yb = cconv2(img, ktrue)
    add_noise2!(yb, 30.0)
    x_tv, k_tv = am_tv2(yb; kh = 9, kw = 9, iters = 5, lam = 5e-4, inner = 4,
                        cg_iters = 25)
    kc = kernel_corr(ktrue, k_tv)
    g = psnr(img, x_tv) - p_blur
    chk("am_tv2 kernel recovery", kc > 0.85, @sprintf("(kcorr %.3f)", kc))
    chk("am_tv2 PSNR gain", g > 2.0, @sprintf("(gain %.2f dB)", g))

    ktrue = gaussian_kernel2(1.5; radius = 4)
    yb = cconv2(img, ktrue)
    x_xj, k_xj, u_sh = xujia2(yb; kh = 13, kw = 13)
    kc = kernel_corr(ktrue, k_xj)
    g = psnr(img, x_xj) - psnr(img, yb)
    chk("xujia2 kernel recovery", kc > 0.90, @sprintf("(kcorr %.3f)", kc))
    chk("xujia2 PSNR gain", g > 3.0, @sprintf("(gain %.2f dB)", g))

    # --- shock filter ---
    ktrue = gaussian_kernel2(2.0; radius = 5)
    yb = cconv2(img, ktrue)
    ush = shock2(yb; iters = 10, dt = 0.05)
    g = psnr(img, ush) - psnr(img, yb)
    chk("shock2 sharpens blurred edges", g > 0.3, @sprintf("(PSNR gain %.2f dB)", g))

    # --- I/O ---
    rnd = rand(37, 29)
    tmpdir = mktempdir()
    bmp_write(joinpath(tmpdir, "t.bmp"), rnd)
    back = bmp_read(joinpath(tmpdir, "t.bmp"))
    e = maximum(abs.(back .- clamp.(rnd, 0.0, 1.0)))
    chk("BMP write/read roundtrip", e <= 1.0 / 255 + 1e-9, @sprintf("(max err %.2e)", e))
    svg_traces(joinpath(tmpdir, "t.svg"),
               [("panel", [("a", "#1f77b4", randn(50)), ("b", "#d62728", randn(50))])])
    svg_kernel(joinpath(tmpdir, "k.svg"), [("k", gaussian_kernel2(1.0))])
    s1 = read(joinpath(tmpdir, "t.svg"), String)
    s2 = read(joinpath(tmpdir, "k.svg"), String)
    ok = startswith(s1, "<?xml") && occursin("</svg>", s1) &&
         startswith(s2, "<?xml") && occursin("</svg>", s2)
    chk("SVG writers produce valid files", ok)
    rm(tmpdir; recursive = true)

    # ===== v0.2.0 additions =====

    # --- multichannel cross-relation ---
    Random.seed!(2001)
    xs = zeros(500)
    for n in 1:500
        rand() < 0.03 && (xs[n] += randn())
    end
    h1 = exp.(-0.6 .* collect(0:6)); h1 ./= sum(h1)
    h2 = exp.(-0.9 .* collect(0:6)); h2 ./= sum(h2)
    h3 = vcat(zeros(3), exp.(-1.2 .* collect(0:3))); h3 ./= sum(h3)
    ys3 = [conv_causal(xs, h1), conv_causal(xs, h2), conv_causal(xs, h3)]
    hest3 = crossrel_identify_multi(ys3; L = 7)
    rel3 = maximum([norm(hest3[c] - hh) / norm(hh)
                    for (c, hh) in enumerate((h1, h2, h3))])
    chk("crossrel_identify_multi 3-channel kernel recovery",
        rel3 < 1e-9, @sprintf("(max rel err %.2e)", rel3))

    # --- robust noise estimation ---
    Random.seed!(4)
    sv = [sin(2 * pi * (i - 1) / 400) for i in 1:600]
    yv = sv .+ 0.05 .* randn(600)
    nm1 = mad_noise1(collect(yv))
    chk("mad_noise1 recovers noise level (slow sine + noise)",
        0.7 * 0.05 < nm1 < 1.3 * 0.05, @sprintf("(est %.4f, true 0.05)", nm1))
    Random.seed!(4)
    sm2 = [0.3 + 0.1 * cos(2 * pi * (i - 1) / 256) + 0.05 * sin(2 * pi * (j - 1) / 192)
           for i in 1:128, j in 1:128] .+ 0.01 .* randn(128, 128)
    nm2 = mad_noise2(sm2)
    chk("mad_noise2 recovers image noise level",
        0.007 < nm2 < 0.013, @sprintf("(est %.4f, true 0.01)", nm2))

    # --- Q estimation ---
    Random.seed!(42)   # (generator reseeds internally; stable here)
    yq, _ = q_attenuated_trace(2048, 100.0, 3.0, 80.0; p = 1.0, seed = 42)
    Qh = q_estimate(yq, 0.01; win = 128, fmin = 1.0, fmax = 8.0, amin = 0.05)
    chk("q_estimate recovers Q (dense reflectivity, pooled fit)",
        isfinite(Qh) && abs(Qh - 80.0) / 80.0 < 0.20,
        @sprintf("(Qhat %.1f, true 80; ~12%% synthesis-blend bias, documented)", Qh))

    # --- non-stationary whitening vs global PEF ---
    yq2, _ = q_attenuated_trace(2048, 100.0, 3.0, 80.0; p = 0.02, seed = 42)
    l_in = _lag1_(yq2[1025:end])
    ns2 = nonstat_decon(yq2; win = 128, dfloor = 1e-3, sm = 2)
    l_ns = _lag1_(ns2[1025:end])
    pd2, _ = prediction_decon(yq2; gap = 1, nfil = 60)
    l_pd = _lag1_(pd2[1025:end])
    chk("nonstat_decon whitens the attenuated late half better than PEF",
        l_ns < 0.85 && l_ns < l_pd - 0.1,
        @sprintf("(late lag1: in %.3f, nonstat %.3f, PEF %.3f)", l_in, l_ns, l_pd))

    # --- FFT correctness/perf rewrite ---
    Random.seed!(5)
    A16 = randn(16, 16)
    F16 = fft2_(A16)
    d16 = zeros(ComplexF64, 16, 16)
    for k1 in 1:16, k2 in 1:16
        s = 0.0 + 0.0im
        for i in 1:16, j in 1:16
            s += A16[i, j] * cis(-2 * pi * ((k1 - 1) * (i - 1) / 16 + (k2 - 1) * (j - 1) / 16))
        end
        d16[k1, k2] = s
    end
    e = maximum(abs.(F16 .- d16))
    chk("fft2_ (pow2 fast path) vs direct 2-D DFT", e < 1e-8,
        @sprintf("(16x16 max err %.2e)", e))
    A256 = randn(256, 256)
    e = maximum(abs.(ifft2_(fft2_(A256)) .- A256))
    chk("fft2_/ifft2_ roundtrip (256x256)", e < 1e-10,
        @sprintf("(max err %.2e)", e))

    # --- GeoCSV roundtrip ---
    rndcsv = mktempdir()
    vals = [0.123456789012345, -9737.0, 3.141592653589793, 1e-8, -2.5e7, 0.0, 42.0, 1.23456789e15]
    pth = joinpath(rndcsv, "t.csv")
    open(pth, "w") do io
        println(io, "# dataset: GeoCSV 2.0")
        println(io, "# delimiter: ,")
        println(io, "# sample_rate_hz: 20")
        println(io, "# start_time: 2019-03-14T20:43:12.026000Z")
        println(io, "Time, Sample")
        for (i, v) in enumerate(vals)
            println(io, "2019-03-14T20:43:", lpad(i, 2, "0"), ".000000Z, ", repr(v))
        end
    end
    yy, ffs, tt0 = parse_geocsv(pth)
    ok = yy == vals && ffs == 20.0 && tt0 == "2019-03-14T20:43:12.026000Z"
    chk("parse_geocsv roundtrip (full precision)", ok)
    rm(rndcsv; recursive = true)

    # --- bandpass + spectral flatness ---
    Nb = 512
    tt = collect(0.0:0.01:(Nb - 1) * 0.01)
    s2hz = sin.(2 * pi * 2.0 .* tt)
    s20hz = sin.(2 * pi * 20.0 .* tt)
    bp = bandpass1(s2hz .+ s20hz, 100.0, 1.0, 5.0)
    c_keep = sigcorr(bp, s2hz)
    out20 = bandpass1(s20hz, 100.0, 1.0, 5.0)
    c_kill = sigcorr(out20, s20hz)
    chk("bandpass1 keeps in-band, kills out-of-band",
        c_keep > 0.995 && abs(c_kill) < 0.05,
        @sprintf("(corr in-band %.4f, out-of-band %.4f)", c_keep, abs(c_kill)))

    # --- robust Huber data term ---
    Random.seed!(44)
    img64 = synth_image2(64)
    kg = gaussian_kernel2(1.5; radius = 4)
    yb64 = cconv2(img64, kg)
    add_noise2!(yb64, 40)
    for i in eachindex(yb64)
        if rand() < 0.015
            yb64[i] += (rand() < 0.5 ? -1 : 1) * 0.8
        end
    end
    xp64 = decon_hq2(yb64, kg; iters = 4)
    xr64 = decon_hq2(yb64, kg; iters = 4, robust = true)
    g_rob = psnr(xr64, img64) - psnr(xp64, img64)
    chk("robust decon_hq2 survives salt outliers (IRLS Huber)",
        g_rob > 5.0, @sprintf("(PSNR gain vs plain %.1f dB)", g_rob))
    # robust must not regress on clean gaussian noise
    Random.seed!(44)
    yb64b = cconv2(img64, kg)
    add_noise2!(yb64b, 40)
    xp64b = decon_hq2(yb64b, kg; iters = 4)
    xr64b = decon_hq2(yb64b, kg; iters = 4, robust = true)
    g_rec = psnr(xr64b, img64) - psnr(xp64b, img64)
    chk("robust decon_hq2 does not regress without outliers",
        g_rec > -0.5, @sprintf("(PSNR gain vs plain %.2f dB)", g_rec))

    # --- coarse-to-fine pyramid ---
    Random.seed!(7)
    img128 = synth_image2(128)
    kp4 = gaussian_kernel2(4.0; radius = 12)
    yb128 = cconv2(img128, kp4)
    add_noise2!(yb128, 40)
    xpyr, kpyr = blind_pyramid2(yb128; method = :ibd, kh = 25, kw = 25,
                               levels = 3, iters = 60)
    xpll, kpll = ibd_rl2(yb128; kh = 25, kw = 25, iters = 60)
    kc_pyr = kernel_corr(kpyr, kp4)
    kc_pll = kernel_corr(kpll, kp4)
    g_pyr = psnr(xpyr, img128) - psnr(xpll, img128)
    chk("blind_pyramid2 beats single-scale on large blur",
        kc_pyr > 0.95 && g_pyr > 1.0 && kc_pll < 0.6,
        @sprintf("(kcorr pyr %.3f vs single %.3f, PSNR gain %.2f dB)", kc_pyr, kc_pll, g_pyr))

    # --- multichannel joint restore ---
    Random.seed!(21)
    ych = [begin
        yyc = cconv2(img64, kg); add_noise2!(yyc, 40); yyc
    end for _ in 1:3]
    xm3 = decon_hq2_multi(ych, [kg, kg, kg]; iters = 4)
    x1c = decon_hq2(ych[1], kg; iters = 4)
    g_mc = psnr(xm3, img64) - psnr(x1c, img64)
    chk("decon_hq2_multi gains SNR from redundant channels",
        g_mc > 1.0, @sprintf("(3-channel gain vs single %.2f dB)", g_mc))

    # --- multiframe blind deconvolution ---
    Random.seed!(21)
    kmo = motion_kernel2(6.0, 0.0)
    yfr = [begin
        yyf = cconv2(img64, kk); add_noise2!(yyf, 30); yyf
    end for kk in (kg, kmo)]
    xmf, kmf = multiframe_bd2(yfr; kh = 15, kw = 15)
    _, ksj, _ = xujia2(yfr[1]; kh = 15, kw = 15)
    kc_mf = kernel_corr(kmf[1], kg)
    kc_sj = kernel_corr(ksj, kg)
    chk("multiframe_bd2 estimates kernels better than single-frame",
        kc_mf > 0.85 && kc_mf - kc_sj > 0.1,
        @sprintf("(gaussian-frame kcorr multiframe %.3f vs single-frame %.3f)", kc_mf, kc_sj))

    # --- kernel-space restart pool ---
    Random.seed!(33)
    img128r = synth_image2(128)
    yb128r = cconv2(img128r, kp4)
    add_noise2!(yb128r, 30)
    pool = restart_search2(yb128r; kh = 25, kw = 25, iters = 60)
    kc_pool = maximum(kernel_corr(pk, kp4) for (_, _, pk, _) in pool)
    chk("restart_search2 pool contains the true-scale basin",
        kc_pool > 0.95,
        @sprintf("(best pool kcorr %.3f across %d init scales)", kc_pool, length(pool)))

    # --- edge selection in xujia2 ---
    Random.seed!(2002)
    img64e = synth_image2(64)
    yb64e = cconv2(img64e, kg)
    add_noise2!(yb64e, 35)
    _, k0e, _ = xujia2(yb64e; kh = 13, kw = 13)
    _, k1e, _ = xujia2(yb64e; kh = 13, kw = 13, edge_frac = 0.25)
    de = kernel_corr(k1e, kg) - kernel_corr(k0e, kg)
    chk("xujia2 edge_frac improves kernel LS at moderate noise",
        de > 0.03, @sprintf("(kcorr delta %.3f)", de))

    # --- real Mars data (skipped if the bundle file is absent) ---
    marscsv = joinpath(@__DIR__, "data",
                       "xb.elyse.02.bhu.S0105a.mqs2019fddj.1.a.csv")
    if isfile(marscsv)
        ym, fsm, _ = parse_geocsv(marscsv)
        okm = length(ym) == 48000 && fsm == 20.0
        chk("parse_geocsv reads the InSight SEIS S0105a bundle", okm,
            @sprintf("(%d samples, %.0f sps)", length(ym), fsm))
        i1 = max(1, round(Int, (1191.574 - 20.6) * fsm))
        wm = ym[i1:i1+511]
        l0m = _lag1_(wm)
        sf0m = _specflat1(wm)
        pdm, _ = prediction_decon(wm; gap = 1, nfil = 60)
        l1m = _lag1_(pdm)
        sf1m = _specflat1(pdm)
        okp = l0m > 0.95 && l1m < 0.15 && sf1m > 10 * sf0m
        chk("PEF whitens the real S0105a P window",
            okp, @sprintf("(lag1 %.3f -> %.3f, specflat %.4f -> %.4f)",
                          l0m, l1m, sf0m, sf1m))
    else
        chk("parse_geocsv reads the InSight SEIS S0105a bundle", true,
            "(skipped: data file not found)")
        chk("PEF whitens the real S0105a P window", true,
            "(skipped: data file not found)")
    end


    println("=================================================")
    @printf("%d checks, %d failed, %.1f s\n", _ST_N[], _ST_FAIL[], time() - t_total)
    return _ST_FAIL[] == 0
end

# -----------------------------------------------------------------------------
# Section 14: CLI
# -----------------------------------------------------------------------------

function _usage()
    println("BlindDecon.jl - blind deconvolution toolkit (pure Julia stdlib)")
    println()
    println("usage:")
    println("  julia BlindDecon.jl selftest                run the deterministic selftest")
    println("  julia BlindDecon.jl demo_seismic [outdir]   1-D seismic blind decon demo")
    println("  julia BlindDecon.jl demo_image [bmp] [outdir]  2-D image demo (own 24-bit BMP or synthetic)")
    println("  julia BlindDecon.jl demo_mars [csv] [outdir]  real-data demo (InSight SEIS S0105a, bundled)")
    return nothing
end

function main()
    if isempty(ARGS)
        _usage()
        return 1
    end
    cmd = ARGS[1]
    if cmd == "selftest"
        return selftest() ? 0 : 1
    elseif cmd == "demo_seismic"
        outdir = length(ARGS) >= 2 ? ARGS[2] : "demo_seismic_out"
        demo_seismic(outdir)
        return 0
    elseif cmd == "demo_image"
        bmpin = length(ARGS) >= 2 ? ARGS[2] : ""
        outdir = length(ARGS) >= 3 ? ARGS[3] : "demo_image_out"
        demo_image(bmpin, outdir)
        return 0
    elseif cmd == "demo_mars"
        csvin = length(ARGS) >= 2 ? ARGS[2] : ""
        outdir = length(ARGS) >= 3 ? ARGS[3] : "demo_mars_out"
        return demo_mars(csvin, outdir)
    else
        println("unknown command: $cmd")
        _usage()
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
