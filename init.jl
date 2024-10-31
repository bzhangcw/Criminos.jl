

using ForwardDiff
using LinearAlgebra
using Random
using Printf
using LaTeXStrings
using JuMP
using Criminos
using Plots
using Gurobi
using ProgressMeter
using ColorSchemes

using CSV, Tables, DataFrames



if cc.bool_init
    Random.seed!(cc.seed_number)
    # ----------------------------------------------------------------------------
    # decision cost
    # ----------------------------------------------------------------------------
    cₜ = rand(n) / 10

    # ----------------------------------------------------------------------------
    # initial state
    # ----------------------------------------------------------------------------
    τ = ones(n)
    xₙ = Int(round(n // 2; digits=0))
    # initialize τₗ and τₕ
    τ[1:xₙ] .= cc.τₗ
    τ[xₙ:end] .= cc.τₕ
    vec_z = [
        MarkovState(0, n; τ=τ, β=1.0) for idx in 1:ℜ
    ]

    # ----------------------------------------------------------------------------
    # system and mixed-in
    # ----------------------------------------------------------------------------
    vec_Ψ = [BidiagSys(n; style=cc.style_retention) for idx in 1:ℜ]
    for idx in 1:ℜ
        vec_Ψ[idx].λ *= group_size[idx]
        vec_Ψ[idx].q /= group_new_ratio[idx]
        vec_Ψ[idx].Q /= group_new_ratio[idx]
    end

    for idx in 1:ℜ
        Ψ = vec_Ψ[idx]
        z = vec_z[idx]
        vec_z[idx].x .= Ψ.λ + Ψ.Γ * z.x₋ - Ψ.Γₕ * z.y
    end

    # generate data for ℜ population
    N = n * ℜ

    #----------------------------------------------------------------------------
    # get the fixed-point plots 
    #----------------------------------------------------------------------------
    metrics = Dict(
        Criminos.Lₓ => L"\|x - x^*\|",
        Criminos.Lᵨ => L"\|y - y^*\|",
        Criminos.∑y => L"$\sum y$",
        Criminos.∑τ => L"$\sum \tau$",
        Criminos.∑x => L"$\sum x$",
        Criminos.θ => L"$\theta$",
        Criminos.fpr => L"\textrm{FPR}",
        Criminos.ρ => L"$\bar \rho$",
    )

    P = rand(N, N)
    _D, V = eigen(P + P')
end

