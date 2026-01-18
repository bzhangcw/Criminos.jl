# ------------------------------------------------------------
# Continuous-time data utilities (reuse discrete data processing)
# ------------------------------------------------------------
include(joinpath(@__DIR__, "..", "discrete", "datarec.jl"))

"""
    generate_random_data_ct(dims::Tuple{Int,Int}, episode::Float64=240.0; λ0::Float64=1/240)

Generate cohort structure and parameters for the continuous-time model.
Reuses the discrete data generator for cohort metadata and arrivals, and
derives CT hazards:
- Term hazards: ωᵖ = -log(γ) / episode, ωᶠ = -log(σ) / episode
- Recidivism baseline hazard scaling λ0 (per day); individual hazards use exp(h)

Returns a Dict with discrete fields (for reuse) plus CT helpers.
"""
function generate_random_data_ct(
    dims::Tuple{Int,Int},
    episode::Float64=240.0;
    λ0::Float64=1 / 240
)
    data_d = generate_random_data(dims, episode)  # from discrete/datarec.jl
    n = data_d[:n]
    γ = data_d[:γ]
    σ = data_d[:σ]

    # Continuous-time term hazards per cohort (piecewise-constant on an episode)
    ωp = @. -log(max(γ, 1e-9)) / episode
    ωf = @. -log(max(σ, 1e-9)) / episode

    # Reuse the discrete score/h mapping: untreated (first n), treated (next n)
    Fh = data_d[:Fh]

    # Helper to turn h into CT recidivism hazards by arm
    λ_from_h = function (μ; p = ones(n))
        h = Fh(μ; p=p)
        h0 = h[1:n]
        h1 = h[n+1:2n]
        λ0_vec = λ0 .* exp.(h0)
        λ1_vec = λ0 .* exp.(h1)
        return Dict(0 => λ0_vec, 1 => λ1_vec)
    end

    data = copy(data_d)
    data[:episode] = episode
    data[:Δ] = episode
    data[:ωp] = ωp
    data[:ωf] = ωf
    data[:λ0_ct] = λ0
    data[:λ_from_h] = λ_from_h
    # arrivals: mirror discrete but under beta key
    if haskey(data_d, :β)
        data[:β] = data_d[:β]
    elseif haskey(data_d, :λ)
        data[:β] = data_d[:λ]
    else
        data[:β] = zeros(n)
    end
    return data
end

"""
    varphi_ct(λ::AbstractVector, ω::AbstractVector, Δ::Real)
Per-interval recidivism probability in CT with competing risk:
ϕ = λ/(λ+ω) * (1 - exp(-(λ+ω) Δ)).
"""
varphi_ct(λ::AbstractVector, ω::AbstractVector, Δ::Real) = begin
    s = λ .+ ω
    # avoid division by zero
    z = similar(λ)
    @inbounds for i in eachindex(s)
        if s[i] <= 0
            z[i] = 0.0
        else
            z[i] = λ[i] / s[i] * (1 - exp(-s[i] * Δ))
        end
    end
    z
end

"""
    compute_phi_ct(data::Dict, μ; p=ones(n))
Return a Dict:
  0 => Dict(:p => ϕ0p, :f => ϕ0f), 1 => Dict(:p => ϕ1p, :f => ϕ1f)
where ϕ are per-interval recidivism probabilities for untreated/treated and term p/f.
"""
function compute_phi_ct(data::Dict, μ; p=ones(data[:n]))
    n = data[:n]
    Δ = data[:Δ]
    ωp = data[:ωp]
    ωf = data[:ωf]
    λ = data[:λ_from_h](μ; p=p)
    # Use Gamma-mixture-consistent split via A=η*exp(h), Ω=ω*Δ
    k = get(data, :hat_k, 3.97)
    θ = get(data, :hat_θ, 2.05)
    η = data[:η]
    # under Fh: h = [h0; h1]; reconstruct h0/h1 by inverting lambda map if needed:
    # we can derive h0,h1 from λ by h = log.(λ/λ0_ct)
    h0 = log.(max.(λ[0] ./ max(get(data, :λ0_ct, 1 / Δ), 1e-12), 1e-12))
    h1 = log.(max.(λ[1] ./ max(get(data, :λ0_ct, 1 / Δ), 1e-12), 1e-12))
    A0 = η .* exp.(h0)
    A1 = η .* exp.(h1)
    Ωp = ωp .* Δ
    Ωf = ωf .* Δ
    ϕ0p, _, _ = cr_split_vec(A0, Ωp, k, θ)
    ϕ0f, _, _ = cr_split_vec(A0, Ωf, k, θ)
    ϕ1p, _, _ = cr_split_vec(A1, Ωp, k, θ)
    ϕ1f, _, _ = cr_split_vec(A1, Ωf, k, θ)
    return Dict(0 => Dict(:p => ϕ0p, :f => ϕ0f),
        1 => Dict(:p => ϕ1p, :f => ϕ1f))
end

# ------------------------------------------------------------
# Gamma-mixture competing-risk split using 1D Gauss–Legendre
# For each element: A = η e^{h}, Ω = ω Δ, G ~ Gamma(k, θ)
# s = e^{-Ω} (θ/(θ + A))^k
# I = ∫_0^1 e^{-t Ω} (θ/(θ + t A))^k dt  (GL-16 quadrature)
# φ = 1 - s - Ω I ; χ = 1 - s - φ
# ------------------------------------------------------------
const GL16_x = [
    0.9894009349916499325961542,
    0.9445750230732325760779884,
    0.8656312023878317438804679,
    0.7554044083550030338951012,
    0.6178762444026437484466718,
    0.4580167776572273863424194,
    0.2816035507792589132304605,
    0.0950125098376374401853193,
]
const GL16_w = [
    0.0271524594117540948517806,
    0.0622535239386478928628438,
    0.0951585116824927848099251,
    0.1246289712555338720524763,
    0.1495959888165767320815017,
    0.1691565193950025381893121,
    0.1826034150449235888667637,
    0.1894506104550684962853967,
]

gl16_integral(f) = begin
    s = 0.0
    @inbounds for i in 1:8
        t1 = 0.5 * (1 + GL16_x[i])
        t2 = 0.5 * (1 - GL16_x[i])
        s += GL16_w[i] * (f(t1) + f(t2))
    end
    0.5 * s
end

function cr_split_scalar(A::Float64, Ω::Float64, k::Float64, θ::Float64)
    s = exp(-Ω) * (θ / (θ + A))^k
    I = gl16_integral(t -> exp(-t * Ω) * (θ / (θ + t * A))^k)
    φ = 1 - s - Ω * I
    χ = 1 - s - φ
    return φ, χ, s
end

function cr_split_vec(A::AbstractVector, Ω::AbstractVector, k::Real, θ::Real)
    n = length(A)
    φ = similar(A, Float64, n)
    χ = similar(A, Float64, n)
    s = similar(A, Float64, n)
    @inbounds for i in 1:n
        φ[i], χ[i], s[i] = cr_split_scalar(float(A[i]), float(Ω[i]), float(k), float(θ))
    end
    return φ, χ, s
end


