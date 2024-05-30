

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



function plot_convergence(ε, s)
    pls = []
    for id in 1:s
        pl = plot(
            size=(700 * s, 500),
            labelfontsize=20,
            xtickfont=font(25),
            ytickfont=font(25),
            legendfontsize=25,
            titlefontsize=25,
            extra_plot_kwargs=KW(
                :include_mathjax => "cdn",
            ),
        )
        for (idx, (func, fname)) in enumerate(metrics[id])
            vv = ε[id, fname]
            kₑ = vv |> length
            plot!(
                pl, 1:kₑ,
                vv[1:kₑ],
                label=fname
            )
        end
        push!(pls, pl)
    end
    fig = plot(pls...)
    title!(fig, "Convergence of the metrics")
    savefig(fig, "result/$style_name-convergence.$format")

    @info "write to" "result/$style_name-convergence.$format"
end


if bool_init
    # ----------------------------------------------------------------------------
    # decision
    # ----------------------------------------------------------------------------
    cₜ = rand(n) / 10

    # ----------------------------------------------------------------------------
    # mixed-in
    # ----------------------------------------------------------------------------
    Ψ = BidiagSys(n; style=style_retention)
    Ωp = []
    Fp = []
    metrics = []
    for _ in 1:ℜ
        Ω = nothing
        if style_correlation == :uppertriangular
            ∇₀ = style_correlation_seed(n, n)
            ∇ₜ = UpperTriangular(style_correlation_seed(n, n))
            Hₜ = Symmetric(style_correlation_seed(n, n))
            Hₜ = Hₜ' * Hₜ
            # this is the lower bound to
            #   guarantee the uniqueness of NE
            G = Ψ.M * Ψ.Γ
            H₀ = style_correlation_psd ? (G' * inv(I - Ψ.Γ) * G + 1e-3 * I(n)) : zeros(n, n)
            Ω = (∇₀, H₀, ∇ₜ, Hₜ)
        elseif style_correlation == :diagonal
            # uncorrelated case
            ∇₀ = Diagonal(style_correlation_seed(n, n))
            ∇ₜ = Diagonal(style_correlation_seed(n, n))
            Hₜ = Diagonal(style_correlation_seed(n, n))
            Hₜ = Hₜ' * Hₜ
            G = Ψ.M * Ψ.Γ
            H₀ = style_correlation_psd ? (opnorm(G' * inv(I - Ψ.Γ) * G) + 1e-3) * I(n) : zeros(n, n)
            Ω = (∇₀, H₀, ∇ₜ, Hₜ)
        else
            throw(ErrorException("not implemented"))
        end
        push!(Ωp, Ω)
        push!(Fp, z -> F(Ψ, z; fₘ=style_mixin, margs=Ω,))

        ################################################################################
        # get the fixed-point plots 
        ################################################################################
        N(z, z₊) = Criminos.no_mixed_in(z, z₊; args=Ω, Ψ=Ψ)
        H(z, z₊) = Criminos.pot_gnep(z, z₊; args=Ω, Ψ=Ψ)
        ∑y(z, z₊) = Criminos.∑y(z, z₊)

        ################################################################################
        # get the fixed-point plots 
        ################################################################################
        push!(metrics, Dict(
            Criminos.Lₓ => L"\|x - x^*\|",
            Criminos.Lᵨ => L"\|\rho - \rho^*\|",
            # ΔH => L"H - H^*",
            # N => L"\textrm{No-Mixed-In}",
            H => L"H",
            ∑y => L"$\sum y$"
        ))
    end
    zᵦ = MarkovState(0, [rand(n); rand(n)], ones(n) / 2)
end