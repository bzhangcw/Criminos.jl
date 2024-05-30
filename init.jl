

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



if bool_init
    # ----------------------------------------------------------------------------
    # decision
    # ----------------------------------------------------------------------------
    cₜ = rand(n) / 10

    # ----------------------------------------------------------------------------
    # initial state
    # ----------------------------------------------------------------------------
    τ = ones(n)
    xₙ = Int(n // 2)
    α₁ = 0.05
    α₂ = 0.05
    τ[1:xₙ] .= α₁
    τ[xₙ:end] .= α₂
    vec_z = [
        MarkovState(0, n; τ=τ, β=group_size[idx]) for idx in 1:ℜ
    ]
    # ----------------------------------------------------------------------------
    # system and mixed-in
    # ----------------------------------------------------------------------------
    vec_Ψ = [BidiagSys(n; style=style_retention) for idx in 1:ℜ]
    for idx in 1:ℜ
        vec_Ψ[idx].λ *= group_size[idx]
    end

    # generate data for ℜ population
    N = n * ℜ
    Ω, G = generate_Ω(N, n, ℜ)
    Fp(vec_z) = F(
        vec_z, vec_Ψ;
        fₜ=style_decision, targs=nothing,
        fₘ=style_mixin, margs=Ω,
    )

    ################################################################################
    # get the fixed-point plots 
    ################################################################################
    metrics = Dict(
        Criminos.Lₓ => L"\|x - x^*\|",
        Criminos.Lᵨ => L"\|y - y^*\|",
        Criminos.∑y => L"$\sum y$"
    )

end