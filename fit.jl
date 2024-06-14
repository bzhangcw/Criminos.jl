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
    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec__Ψ]...)
    D, _ = eigs(G)
    D = real(D)
    _c = vcat([_Ψ.Q * _Ψ.λ for _Ψ in vec__Ψ]...)
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
    ρₛ=rand(n),
    τₛ=rand(n),
    Σ=nothing,
    bool_type1=true
)
    @info "" bool_type1 ρₛ τₛ
    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec_Ψ]...)
    D, _ = eigs(G)
    D = real(D)
    # shifting parameter
    ϵ = bool_type1 ? 1e-1 : 1e-2
    _Ψ = vec_Ψ[1]


    expt(τ) = 1 .- exp.(-τ .^ 2) .+ ϵ
    T = diagm(expt(τₛ))
    invT = inv(T)

    md = Model(optimizer_with_attributes(
        () -> COPT.ConeOptimizer(),
    ))

    # ι must exceed maximum ρₛ
    ι = max(cc.ι, ρₛ...)
    M₁ = -_Ψ.Γ + I + diagm(ρₛ) * _Ψ.Γₕ
    M₂ = -_Ψ.Γ + I + cc.ι .* _Ψ.Γₕ
    b₁ = ρₛ .* _Ψ.λ
    θ = _Ψ.Q * _Ψ.λ

    @variable(md, yv[1:n] .>= 0)
    @variable(md, _bv[1:n] .>= 0)
    @variable(md, _Bv[1:n, 1:n] .>= 0)
    set_upper_bound.(_bv, 1e4)
    @constraint(md, _Bv[1:n, 1:n] - diagm(_bv) .== 0)
    if bool_type1
        # quad H linear-quad g
        @constraint(md, (_Bv * invT * Σ * ones(n)) .== yv)
    else
        # linear H quad g
        @constraint(md, (_Bv * Σ * T * ones(n) + _Bv * invT * θ) .== yv)
        @constraint(md, _Bv .>= 0)
    end
    @constraint(md, M₁ * yv .== b₁)
    @constraint(md, M₂ * yv .<= ι .* _Ψ.λ)
    @objective(md, Max, tr(_Bv))

    optimize!(md)

    y = value.(yv)
    x = 1 ./ (1 .- _Ψ.γ) .* (_Ψ.λ - _Ψ.Γₕ * y)
    _B = value.(_Bv)
    _A = inv(_B)

    # not stable
    # H₁(τ) = diagm(expt(τ)) * _A * inv(Σ) * _A' * diagm(expt(τ))
    # g₁(τ) = diagm(expt(τ)) * _A * ones(n) + H₁(τ) * θ
    # y₁(τ) = H₁(τ) \ g₁(τ)
    H₁(τ) = _A * diagm(expt(τ)) .^ 2 * _A'
    g₁(τ) = _A * diagm(expt(τ)) * Σ * ones(n)
    y₁(τ) = H₁(τ) \ g₁(τ)

    H₂(τ) = _A * diagm(expt(τ)) * _A'
    g₂(τ) = _A * diagm(expt(τ)) * Σ * diagm(expt(τ)) * ones(n) + _A * θ
    y₂(τ) = H₂(τ) \ g₂(τ)


    Hₕ = bool_type1 ? H₁ : H₂
    gₕ = bool_type1 ? g₁ : g₂
    yₕ = bool_type1 ? y₁ : y₂

    @info "" maximum(abs.(y ./ x - ρₛ))
    @info "" maximum(abs.(_A * _B - I)) < 1e-5
    @info "" maximum(abs.(yₕ(τₛ) .- y))
    @info "" (yₕ(τₛ .* 1000)) |> sum

    ω∇ω(y, τ) = begin
        _H = Hₕ(τ) + _Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ
        _g = -gₕ(τ) - _Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ
        _w = 1 / 2 * y' * _H * y + y' * _g
        ∇ = _H * y + _g
        _w, ∇
    end
    return ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B
end

function tuning(n; type_monotone=true, ρₛ=nothing, τₛ=nothing)

    Σ = rand(n, n)
    Σ = Σ' * Σ + 1e-2 * I
    if cc.style_mixin_parameterization == :random
        _args = generate_random_Ω(N, n, ℜ)
    elseif cc.style_mixin_parameterization == :fitting
        _args = generate_fitting_Ω(
            N, n, ℜ;
            ρₛ=ρₛ,
            τₛ=τₛ,
            Σ=Σ,
            bool_type1=type_monotone
        )
    else
    end
    return _args
end