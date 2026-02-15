# ------------------------------------------------------------
# Gamma idiosyncrasy for individual-level random effects
#
# This file implements the Gamma frailty model:
#   exp(α_i) ~ Gamma(k, θ)
#
# Uses Gauss-Legendre (GL-16) quadrature for numerical integration.
# ------------------------------------------------------------

using SpecialFunctions: gamma

"""
    GammaIdiosyncrasy(k=3.97, θ=2.05)

Individual idiosyncrasy with exp(α_i) ~ Gamma(k, θ).
Requires GL quadrature for integration.
"""
struct GammaIdiosyncrasy <: AbstractIdiosyncrasy
    k::Float64
    θ::Float64
    name::Symbol
    GammaIdiosyncrasy(k::Float64=3.97, θ::Float64=2.05) = new(k, θ, :gamma)
end

function create_Fφ(idio::GammaIdiosyncrasy, η::Real, n::Int)
    k, θ = idio.k, idio.θ
    # Gamma mixture: integrate exp(α) ~ Gamma(k, θ)
    # Result: φ = 1 - (θ / (θ + η*exp(h)))^k
    __Fφ(h) = 1 - (θ / (θ + η * exp(h)))^k

    Fφ(h; p=ones(n)) = begin
        ϕ = __Fφ.(h)
        return Dict(0 => ϕ[1:n], 1 => ϕ[n+1:2n])
    end
    return Fφ
end

function compute_return_rates(idio::GammaIdiosyncrasy, data::Union{SortedDict,Dict}, μ; p=ones(data[:n]))
    return compute_return_rates_gamma(idio, data, μ; p=p)
end

# ------------------------------------------------------------
# Gamma return rate computation
# ------------------------------------------------------------

@doc raw"""
    compute_return_rates_gamma(idio, data, μ; p=ones(n)) -> Dict(0=>r0, 1=>r1)
    
Compute cohort-wise return rates r = ϕ_f / (1 - s_f) using GL-16 quadrature
for the Gamma mixture integral. Does not modify `data`.

Uses assumption: exp(α_i) ~ Gamma(k, θ)
"""
function compute_return_rates_gamma(idio::GammaIdiosyncrasy, data::Union{SortedDict,Dict}, μ; p=ones(data[:n]))
    n = data[:n]
    η = data[:η]
    k, θ = idio.k, idio.θ
    Ωf = @. -log(max(data[:σ], 1e-9))
    h = data[:Fh](μ; p=p)
    h0 = h[1:n]
    h1 = h[n+1:2n]
    A0 = η .* exp.(h0)
    A1 = η .* exp.(h1)
    ϕ0f, _χ0f, s0f = _gl16_cr_split_vec(A0, Ωf, k, θ)
    ϕ1f, _χ1f, s1f = _gl16_cr_split_vec(A1, Ωf, k, θ)
    r0 = ϕ0f ./ max.(1 .- s0f, 1e-12)
    r1 = ϕ1f ./ max.(1 .- s1f, 1e-12)
    return Dict(0 => r0, 1 => r1)
end

# ------------------------------------------------------------
# Gauss-Legendre (GL-16) quadrature nodes and weights
# ------------------------------------------------------------

const _GL16_x = [
    0.9894009349916499325961542,
    0.9445750230732325760779884,
    0.8656312023878317438804679,
    0.7554044083550030338951012,
    0.6178762444026437484466718,
    0.4580167776572273863424194,
    0.2816035507792589132304605,
    0.0950125098376374401853193,
]

const _GL16_w = [
    0.0271524594117540948517806,
    0.0622535239386478928628438,
    0.0951585116824927848099251,
    0.1246289712555338720524763,
    0.1495959888165767320815017,
    0.1691565193950025381893121,
    0.1826034150449235888667637,
    0.1894506104550684962853967,
]

# ------------------------------------------------------------
# GL-16 integration routines
# ------------------------------------------------------------

@doc """
    _gl16_integral(f) -> Real

Compute ∫₀¹ f(t) dt using 16-point Gauss-Legendre quadrature.
"""
_gl16_integral(f) = begin
    s = 0.0
    @inbounds for i in 1:8
        t1 = 0.5 * (1 + _GL16_x[i])
        t2 = 0.5 * (1 - _GL16_x[i])
        s += _GL16_w[i] * (f(t1) + f(t2))
    end
    0.5 * s
end

@doc """
    _gl16_cr_split_scalar(A, Ω, k, θ) -> φ, χ, s

Compute the Gamma mixture integral for competing risks (scalar version):
    s = e^{-Ω} (θ/(θ + A))^k           (survival)
    I = ∫_0^1 e^{-t Ω} (θ/(θ + t A))^k dt  (GL-16 quadrature)
    φ = 1 - s - Ω * I                   (recidivism)
    χ = 1 - s - φ                       (other exit)
"""
_gl16_cr_split_scalar(A, Ω, k, θ) = begin
    s = exp(-Ω) * (θ / (θ + A))^k
    I = _gl16_integral(t -> exp(-t * Ω) * (θ / (θ + t * A))^k)
    φ = 1 - s - Ω * I
    χ = 1 - s - φ
    return φ, χ, s
end

@doc """
    _gl16_cr_split_vec(A, Ω, k, θ) -> φ, χ, s

Vectorized version of `_gl16_cr_split_scalar`.
"""
_gl16_cr_split_vec(A::AbstractVector, Ω::AbstractVector, k::Real, θ::Real) = begin
    n = length(A)
    φ = similar(A, n)
    χ = similar(A, n)
    s = similar(A, n)
    @inbounds for i in 1:n
        φ[i], χ[i], s[i] = _gl16_cr_split_scalar(A[i], Ω[i], k, θ)
    end
    return φ, χ, s
end

# ------------------------------------------------------------
# Gamma distribution utilities
# ------------------------------------------------------------

@doc """
    _gamma_pdf(g, k, θ) -> Real
    
Gamma PDF: f(g; k, θ) = θ^k / Γ(k) * g^{k-1} * exp(-θ*g)
Using rate parameterization (θ is rate, not scale).
"""
_gamma_pdf(g::Real, k::Real, θ::Real) = g > 0 ? (θ^k / gamma(k)) * g^(k - 1) * exp(-θ * g) : 0.0

@doc """
    _gamma_quantile_approx(p, k, θ) -> Real

Approximate Gamma quantile (inverse CDF) using Wilson-Hilferty approximation.
"""
function _gamma_quantile_approx(p::Real, k::Real, θ::Real)
    if p <= 0
        return 0.0
    elseif p >= 1
        return 10 * k / θ  # large value
    end
    # Wilson-Hilferty approximation
    z = _norminv(p)
    cube = 1 + z * sqrt(2 / (9 * k)) - 1 / (9 * k)
    cube = max(cube, 0.01)
    return (k / θ) * cube^3
end

@doc """
    _norminv(p) -> Real

Approximate standard normal inverse CDF (probit function).
"""
function _norminv(p::Real)
    if p <= 0
        return -6.0
    elseif p >= 1
        return 6.0
    elseif p == 0.5
        return 0.0
    end
    if p > 0.5
        return -_norminv(1 - p)
    end
    t = sqrt(-2 * log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t^2) / (1 + d1 * t + d2 * t^2 + d3 * t^3))
end

# ------------------------------------------------------------
# GL-16 averaged competing-risk probabilities for CT model
# (continuous-time with Gamma idiosyncrasy)
# ------------------------------------------------------------

@doc """
    _gl16_ct_probs_scalar(A, ωp, ωf, Δ, k, θ; δ_inc=0.0) -> (s_p, m_pf, y_p, s_f, x_f, y_f, inc_p, inc_f)

Compute Gamma-averaged CT competing-risk probabilities for a single cohort:
- s_p   = E[exp(-(λ + ωp)Δ)]                          (stay in probation)
- m_pf  = E[ωp/(λ+ωp) (1 - exp(-(λ+ωp)Δ))]            (probation → follow-up)
- y_p   = E[(1-δ_inc)λ/(λ+ωp) (1 - exp(-(λ+ωp)Δ))]    (recidivism in probation, not incarcerated)
- s_f   = E[exp(-(λ + ωf)Δ)]                          (stay in follow-up)
- x_f   = E[ωf/(λ+ωf) (1 - exp(-(λ+ωf)Δ))]            (exit: complete follow-up)
- y_f   = E[(1-δ_inc)λ/(λ+ωf) (1 - exp(-(λ+ωf)Δ))]    (recidivism in follow-up, not incarcerated)
- inc_p = E[δ_inc*λ/(λ+ωp) (1 - exp(-(λ+ωp)Δ))]       (incarcerated from probation)
- inc_f = E[δ_inc*λ/(λ+ωf) (1 - exp(-(λ+ωf)Δ))]       (incarcerated from follow-up)

where λ = A * g and g ~ Gamma(k, θ), δ_inc is incarceration probability upon reoffense.
"""
function _gl16_ct_probs_scalar(A::Real, ωp::Real, ωf::Real, Δ::Real, k::Real, θ::Real; δ_inc::Real=0.0)
    g_max = _gamma_quantile_approx(0.9999, k, θ)

    s_p = _gl16_integral(t -> begin
        g = t * g_max
        λ = A * g
        val = exp(-(λ + ωp) * Δ)
        pdf_g = _gamma_pdf(g, k, θ)
        val * pdf_g * g_max
    end)

    m_pf = _gl16_integral(t -> begin
        g = t * g_max
        λ = A * g
        denom = max(λ + ωp, 1e-12)
        val = (ωp / denom) * (1 - exp(-denom * Δ))
        pdf_g = _gamma_pdf(g, k, θ)
        val * pdf_g * g_max
    end)

    y_p_total = _gl16_integral(t -> begin
        g = t * g_max
        λ = A * g
        denom = max(λ + ωp, 1e-12)
        val = (λ / denom) * (1 - exp(-denom * Δ))
        pdf_g = _gamma_pdf(g, k, θ)
        val * pdf_g * g_max
    end)
    y_p = (1 - δ_inc) * y_p_total
    inc_p = δ_inc * y_p_total

    s_f = _gl16_integral(t -> begin
        g = t * g_max
        λ = A * g
        val = exp(-(λ + ωf) * Δ)
        pdf_g = _gamma_pdf(g, k, θ)
        val * pdf_g * g_max
    end)

    x_f = _gl16_integral(t -> begin
        g = t * g_max
        λ = A * g
        denom = max(λ + ωf, 1e-12)
        val = (ωf / denom) * (1 - exp(-denom * Δ))
        pdf_g = _gamma_pdf(g, k, θ)
        val * pdf_g * g_max
    end)

    y_f_total = _gl16_integral(t -> begin
        g = t * g_max
        λ = A * g
        denom = max(λ + ωf, 1e-12)
        val = (λ / denom) * (1 - exp(-denom * Δ))
        pdf_g = _gamma_pdf(g, k, θ)
        val * pdf_g * g_max
    end)
    y_f = (1 - δ_inc) * y_f_total
    inc_f = δ_inc * y_f_total

    return s_p, m_pf, y_p, s_f, x_f, y_f, inc_p, inc_f
end

@doc """
    _gl16_ct_probs_vec(A, ωp, ωf, Δ, k, θ; δ_inc=0.0) -> (s_p, m_pf, y_p, s_f, x_f, y_f, inc_p, inc_f)

Vectorized version of _gl16_ct_probs_scalar.
"""
function _gl16_ct_probs_vec(A::AbstractVector, ωp::AbstractVector, ωf::AbstractVector,
    Δ::Real, k::Real, θ::Real; δ_inc::Real=0.0)
    n = length(A)
    s_p = similar(A, n)
    m_pf = similar(A, n)
    y_p = similar(A, n)
    s_f = similar(A, n)
    x_f = similar(A, n)
    y_f = similar(A, n)
    inc_p = similar(A, n)
    inc_f = similar(A, n)
    @inbounds for i in 1:n
        s_p[i], m_pf[i], y_p[i], s_f[i], x_f[i], y_f[i], inc_p[i], inc_f[i] = _gl16_ct_probs_scalar(A[i], ωp[i], ωf[i], Δ, k, θ; δ_inc=δ_inc)
    end
    return s_p, m_pf, y_p, s_f, x_f, y_f, inc_p, inc_f
end
