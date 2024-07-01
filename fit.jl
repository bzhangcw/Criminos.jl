using ForwardDiff
using LinearAlgebra
using Random
using Printf
using LaTeXStrings
using JuMP
using Criminos
using Plots
using ProgressMeter
using ColorSchemes
using SCS, COPT

using CSV, Tables, DataFrames

@doc raw"""
old style randomized approach
"""
function generate_random_Ω(N, n, ℜ; kwargs...)
    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec_Ψ]...)
    D, _ = eigs(G)
    D = real(D)
    _c = vcat([_Ψ.Q * _Ψ.λ for _Ψ in vec_Ψ]...)
    if cc.style_correlation == :uppertriangular
        ∇₀ = Matrix(G)
        ∇ₜ = UpperTriangular(cc.style_correlation_seed(N, N))
        Hₜ = Symmetric(cc.style_correlation_seed(N, N))
        Hₜ = Hₜ' * Hₜ
        # this is the lower bound to
        #   guarantee the uniqueness of NE
        H₀ = (
            cc.style_correlation_psd ? G + 1e-3 * I(N) : zeros(N, N)
        )
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    else
        throw(ErrorException("not implemented"))
    end
    if !cc.style_correlation_subp
        Hₜ = blockdiag([sparse(Hₜ[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    end
    ∇₀, H₀, ∇ₜ, Hₜ, _... = Ω
    ω∇ω(y, τ) = begin
        _t = τ .+ 1
        _T = diagm(τ)
        # Cₜ = diagm(τ) * Hₜ * diagm(τ) + H₀
        # cₜ = (∇ₜ*τ)[:] - (∇₀*_c)[:]
        Cₜ = _T * (Hₜ) * _T + H₀
        cₜ = -(∇ₜ*_t)[:] - (Cₜ*∇₀*_c)[:]
        _w = (
            y' * Cₜ * y / 2 +    # second-order
            y' * cₜ             # first-order
        )
        ∇ = Cₜ * y + cₜ
        return _w, ∇
    end
    return ω∇ω, G, cc.ι
end

@doc raw"""
fitting data to a desired distribution
ρₛ - wanted stationary distribution
τₛ - wanted treatment that achieves ρₛ
"""
function generate_fitting_Ω(N, n, ℜ;
    ρₛ=rand(N),
    τₛ=rand(N),
    Σ₁=nothing,
    Σ₂=nothing,
    style_mixin_monotonicity=3
)
    @info "" style_mixin_monotonicity ρₛ τₛ
    # shifting parameter
    ϵ = 5e-2
    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec_Ψ]...)
    expt(τ) = 1 .- exp.(-τ .^ 2) .+ ϵ
    T = diagm(expt(τₛ))
    invT = inv(T)

    md = Model(optimizer_with_attributes(
        () -> COPT.Optimizer(),
    ))

    # ι must exceed maximum ρₛ
    M₁, M₂, b₁, θ, ι = [], [], [], [], []
    for (id, _Ψ) in enumerate(vec_Ψ)
        _ρ = ρₛ[(id-1)*n+1:id*n]
        push!(ι, max(cc.ι, _ρ...))
        push!(M₁, -_Ψ.Γ + I + diagm(_ρ) * _Ψ.Γₕ)
        push!(M₂, _Ψ.Γ + I + cc.ι .* _Ψ.Γₕ)
        push!(b₁, _ρ .* _Ψ.λ)
        push!(θ, _Ψ.Q * _Ψ.λ)
    end

    @variable(md, yv[1:N] .>= 0)
    @variable(md, _bv[1:N] .>= 0)
    @variable(md, _Bv[1:N, 1:N] .>= 0)
    set_upper_bound.(_bv, 1e4)
    @constraint(md, _Bv[1:N, 1:N] - diagm(_bv) .== 0)
    if style_mixin_monotonicity != 2
        # decreasing
        @error("not implemented")
    else
        style_mixin_monotonicity == 2
        # U-shaped
        @constraint(md, (_Bv * Σ₁ * T * ones(N) + _Bv * invT * Σ₂ * vcat(θ...)) .== yv)
    end
    for (id, _Ψ) in enumerate(vec_Ψ)
        @constraint(md, M₁[id] * yv[(id-1)*n+1:id*n] .== b₁[id])
    end
    for (id, _Ψ) in enumerate(vec_Ψ)
        # @constraint(md, M₂[id] * yv[(id-1)*n+1:id*n] .<= ι[id] .* _Ψ.λ)
        # @constraint(md, M₂[id] * yv[(id-1)*n+1:id*n] .<= _Ψ.λ)
    end
    @objective(md, Max, tr(_Bv))
    optimize!(md)

    if termination_status(md) != MOI.OPTIMAL
        @warn "Optimizer did not converge"
        @info "" termination_status(md)
        return md
    end
    y = value.(yv)
    x = vcat([1 ./ (1 .- _Ψ.γ) .* (_Ψ.λ - _Ψ.Γₕ * y[(id-1)*n+1:id*n])
              for (id, _Ψ) in enumerate(vec_Ψ)]...)
    _B = value.(_Bv)
    _A = inv(_B)

    Hₕ(τ) = _A * diagm(expt(τ)) * _A'
    gₕ(τ) = _A * diagm(expt(τ)) * Σ₁ * diagm(expt(τ)) * ones(N) + _A * Σ₂ * vcat(θ...)
    yₕ(τ) = Hₕ(τ) \ gₕ(τ)

    @info "" maximum(abs.(y ./ x - ρₛ))
    @info "" maximum(abs.(_A * _B - I)) < 1e-5
    @info "" maximum(abs.(yₕ(τₛ) .- y))
    @info "" (yₕ(τₛ .* 1000)) |> sum

    ω∇ω(y, τ) = begin
        _H = Hₕ(τ) + G
        # _g = -gₕ(τ) - _Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ
        _g = -gₕ(τ) - vcat([_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ for _Ψ in vec_Ψ]...)
        _w = 1 / 2 * y' * _H * y + y' * _g
        ∇ = _H * y + _g
        _w, ∇
    end
    return ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B, md
end

function tuning(n, Σ₁, Σ₂; style_mixin_monotonicity=3, ρₛ=nothing, τₛ=nothing)

    if cc.style_correlation == :uppertriangular
        Σ₁ = UpperTriangular(Σ₁)
        Σ₂ = UpperTriangular(Σ₂)
    else
        throw(ErrorException("not implemented"))
    end
    if !cc.style_correlation_subp
        Σ₁ = blockdiag([sparse(Σ₁[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
        Σ₂ = blockdiag([sparse(Σ₂[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
    end
    if cc.style_mixin_parameterization == :random
        _args = generate_random_Ω(N, n, ℜ)
    elseif cc.style_mixin_parameterization == :fitting
        _args = generate_fitting_Ω(
            N, n, ℜ;
            ρₛ=ρₛ,
            τₛ=τₛ,
            Σ₁=Σ₁,
            Σ₂=Σ₂,
            style_mixin_monotonicity=style_mixin_monotonicity
        )
    else
    end
    return _args
end