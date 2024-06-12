

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
    # ----------------------------------------------------------------------------
    # decision 
    # ----------------------------------------------------------------------------
    cₜ = rand(n) / 10

    # ----------------------------------------------------------------------------
    # initial state
    # ----------------------------------------------------------------------------
    τ = ones(n)
    xₙ = Int(n // 2)
    # initialize α₁ and α₂
    τ[1:xₙ] .= cc.α₁
    τ[xₙ:end] .= cc.α₂
    vec_z = [
        MarkovState(0, n; τ=τ, β=group_size[idx]) for idx in 1:ℜ
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
    _args = nothing
    # ρₛ = [1:n...] ./ n ./ 2
    ρₛ = rand(n)
    Uₛ = rand(n, n)
    Σ = rand(n, n)
    Σ = Σ' * Σ + 1e-2 * I
    if cc.style_mixin_parameterization == :random
        _args = generate_random_Ω(N, n, ℜ)
    elseif cc.style_mixin_parameterization == :fitting
        _args = generate_fitting_Ω(
            N, n, ℜ;
            ρₛ=ρₛ,
            τₛ=vec_z[1].τ,
            Uₛ=Uₛ,
            Σ=Σ,
            bool_type1=false
        )
    else
    end
    cₜ = rand(n)
    Fp(vec_z) = F!(
        vec_z, vec_Ψ;
        fₜ=cc.style_decision, targs=(cₜ, cc.α₁, cc.α₂),
        fₘ=cc.style_mixin, margs=_args,
    )

    ################################################################################
    # get the fixed-point plots 
    ################################################################################
    metrics = Dict(
        Criminos.Lₓ => L"\|x - x^*\|",
        Criminos.Lᵨ => L"\|y - y^*\|",
        Criminos.∑y => L"$\sum y$",
        # Criminos.∑τ => L"$\sum \tau$",
        Criminos.∑x => L"$\sum x$",
        Criminos.θ => L"$\theta$",
        Criminos.fpr => L"\textrm{FPR}",
    )
end