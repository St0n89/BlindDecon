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
#
#  2-D (image):
#    - decon_wiener2          non-blind Wiener (known kernel)
#    - decon_rl2              non-blind Richardson-Lucy
#    - decon_hq2              half-quadratic solver, hyper-Laplacian priors
#                             (alpha=1 -> TV, alpha=0.5/0.8 -> heavy-tailed)
#    - ibd_rl2                blind Ayers-Dainty iterative blind decon (RL)
#    - am_tv2                 blind Chan-Wong alternating TV minimization
#    - xujia2                 blind Xu-Jia: shock filter + gradient-domain
#                             kernel LS + hyper-Laplacian restoration
#    - shock2                 Osher-Rudin shock filter
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

function _fft_pow2(a::Vector{ComplexF64}, inverse::Bool)
    n = length(a)
    n == 1 && return copy(a)
    b = ComplexF64.(a)
    # bit-reversal permutation
    j = 1
    @inbounds for i in 1:n-1
        if i < j
            b[i], b[j] = b[j], b[i]
        end
        m = n >> 1
        while m >= 1 && j > m
            j -= m
            m >>= 1
        end
        j += m
    end
    len = 2
    @inbounds while len <= n
        half = len >> 1
        ang = (inverse ? 2.0 : -2.0) * pi / len
        wl = cis(ang)
        i = 1
        while i <= n
            w = 1.0 + 0.0im
            for k in 1:half
                u = b[i + k - 1]
                v = b[i + k + half - 1] * w
                b[i + k - 1] = u + v
                b[i + k + half - 1] = u - v
                w *= wl
            end
            i += len
        end
        len <<= 1
    end
    if inverse
        b ./= n
    end
    return b
end

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
        return _fft_pow2(c, false)
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
function grad_mults(n1::Int, n2::Int)
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
    out .+= lam .* real.(ifft2_(conj.(Dx) .* fft2_(gx) .+ conj.(Dy) .* fft2_(gy)))
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

function decon_hq2(y::Matrix{Float64}, k::Matrix{Float64};
                   alpha::Float64 = 0.8, lam::Float64 = 2e-4,
                   iters::Int = 8, cg_iters::Int = 30, epsw::Float64 = 0.03)
    n1, n2 = size(y)
    Ks = fft2_(embed_kernel2(k, n1, n2))
    absK2 = abs2.(Ks)
    Y = fft2_(y)
    b = real.(ifft2_(conj.(Ks) .* Y))
    Dx, Dy = grad_mults(n1, n2)
    x = copy(y)
    for _ in 1:iters
        gx, gy = circgrad(x)
        w1 = (gx .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
        w2 = (gy .^ 2 .+ epsw * epsw) .^ ((alpha - 2.0) / 2.0)
        x = _hq_cg(x, b, absK2, Dx, Dy, lam, w1, w2, cg_iters)
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
    for _ in 1:iters
        Ks = fft2_(embed_kernel2(k, n1, n2))
        # --- image RL steps
        for _ in 1:xsteps
            c = cconv2_spec(fft2_(x), Ks)
            c = max.(c, eps)
            ratio = y ./ c
            x = x .* real.(ifft2_(conj.(Ks) .* fft2_(ratio)))
            x = max.(x, 0.0)
        end
        # --- kernel RL step in the embedded grid:
        # k_emb <- k_emb .* adjoint_of_kernel_op(ratio), ratio = y ./ (x (x) k)
        for _ in 1:ksteps
            c = cconv2_spec(fft2_(x), Ks)
            c = max.(c, eps)
            ratio = y ./ c
            # adjoint w.r.t. the embedded kernel:
            # (A' r)_emb = conj(Xs) .* R ./ (n1*n2)   (real-valued after ifft)
            g = real.(ifft2_(conj.(fft2_(x)) .* fft2_(ratio)))
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
function xujia2(y::Matrix{Float64}; kh::Int = 13, kw::Int = 13,
                shock_iters::Int = 10, gamma::Float64 = 5e-3,
                alpha::Float64 = 0.8, lam::Float64 = 3e-4)
    n1, n2 = size(y)
    u = shock2(y; iters = shock_iters)
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

# -----------------------------------------------------------------------------
# Section 11: helpers for validation
# -----------------------------------------------------------------------------

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
    println("images + kernels.svg written to $outdir")
    return nothing
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
    else
        println("unknown command: $cmd")
        _usage()
        return 1
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
