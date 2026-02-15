# ------------------------------------------------------------
# Idiosyncrasy module for individual-level random effects
#
# This file provides the base interface for handling idiosyncrasy 
# in individual recidivism rates through random intercept α_i.
#
# Model: λ_i = λ_0 * exp(h_i), where h_i = α_i + κμ + score_v
#
# Supported idiosyncrasy types:
#   ConstantIdiosyncrasy - assume α_i = α₀ (no randomness)
#   GammaIdiosyncrasy    - exp(α_i) ~ Gamma(k, θ) (frailty model)
#
# Future extensions: LogNormalIdiosyncrasy, DiscreteMixtureIdiosyncrasy, etc.
# ------------------------------------------------------------

using DataStructures: SortedDict

# ------------------------------------------------------------
# Idiosyncrasy type definitions
# ------------------------------------------------------------

"""
Abstract type for idiosyncrasy specifications.
"""
abstract type AbstractIdiosyncrasy end

"""
    ConstantIdiosyncrasy(α₀=0.0)

No individual idiosyncrasy - all individuals have the same intercept α_i = α₀.
"""
struct ConstantIdiosyncrasy <: AbstractIdiosyncrasy
    α₀::Float64
    name::Symbol
    ConstantIdiosyncrasy(α₀::Float64=3.97 / 2.05) = new(α₀, :constant)
end

# ------------------------------------------------------------
# Recidivism probability functions
# ------------------------------------------------------------

@doc """
    create_Fφ(idio::AbstractIdiosyncrasy, η, n) -> Fφ

Create the recidivism probability function based on idiosyncrasy type.

- `ConstantIdiosyncrasy`: Exponential φ = 1 - exp(-η * exp(h + α₀))
- `GammaIdiosyncrasy`: Gamma mixture φ = 1 - (θ/(θ + η*exp(h)))^k
"""
function create_Fφ(idio::ConstantIdiosyncrasy, η::Real, n::Int)
    α₀ = idio.α₀
    # Constant α_i = α₀: no idiosyncrasy
    # φ = 1 - exp(-η * exp(h + α₀))
    __Fφ(h) = 1 - exp(-η * exp(h + α₀))

    Fφ(h; p=ones(n)) = begin
        ϕ = __Fφ.(h)
        return Dict(0 => ϕ[1:n], 1 => ϕ[n+1:2n])
    end
    return Fφ
end

# ------------------------------------------------------------
# Return rate computation dispatcher
# ------------------------------------------------------------

@doc raw"""
    compute_return_rates(data, μ; p=ones(n)) -> Dict(0=>r0, 1=>r1)
    
Dispatcher for computing cohort-wise return rates r = ϕ_f / (1 - s_f).
Dispatches based on `data[:idiosyncrasy]` type.
"""
function compute_return_rates(data::Union{SortedDict,Dict}, μ; p=ones(data[:n]))
    idio = data[:idiosyncrasy]
    return compute_return_rates(idio, data, μ; p=p)
end

# Dispatch for ConstantIdiosyncrasy
function compute_return_rates(idio::ConstantIdiosyncrasy, data::Union{SortedDict,Dict}, μ; p=ones(data[:n]))
    return compute_return_rates_constant(idio, data, μ; p=p)
end

# ------------------------------------------------------------
# Constant idiosyncrasy (α_i = α₀)
# ------------------------------------------------------------

@doc raw"""
    compute_return_rates_constant(idio, data, μ; p=ones(n)) -> Dict(0=>r0, 1=>r1)
    
Compute cohort-wise return rates r = ϕ_f / (1 - s_f) with constant α_i = α₀.
No integration over any distribution needed.

Uses: λ_i = λ_0 * exp(h + α₀), with h = score + κμ
      r = λ / (λ + ω) for competing risks in off-probation
"""
function compute_return_rates_constant(idio::ConstantIdiosyncrasy, data::Union{SortedDict,Dict}, μ; p=ones(data[:n]))
    n = data[:n]
    η = data[:η]
    α₀ = idio.α₀
    Ωf = @. -log(max(data[:σ], 1e-9))  # follow-up period length (in hazard units)
    h = data[:Fh](μ; p=p)
    h0 = h[1:n]
    h1 = h[n+1:2n]

    # λ * T for follow-up period (include α₀)
    λT0 = η .* exp.(h0 .+ α₀) .* Ωf
    λT1 = η .* exp.(h1 .+ α₀) .* Ωf

    # Competing risks: r = λ / (λ + ω) = λT / (λT + ωT)
    # where Ωf = ω_{c,tf} * T_e
    r0 = λT0 ./ (λT0 .+ Ωf)
    r1 = λT1 ./ (λT1 .+ Ωf)
    return Dict(0 => r0, 1 => r1)
end
