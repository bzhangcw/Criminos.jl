

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



function plot_convergence(traj, ε, kₑ)
    fig = plot(
        size=(700, 500),
        labelfontsize=20,
        xtickfont=font(25),
        ytickfont=font(25),
        legendfontsize=25,
        titlefontsize=25,
        extra_plot_kwargs=KW(
            :include_mathjax => "cdn",
        ),
    )
    for (idx, (func, fname)) in enumerate(metrics)
        plot!(
            fig, 1:kₑ,
            ε[fname][1:kₑ],
            label=fname,
        )
    end
    title!(fig, "Convergence of the metrics")
    savefig(fig, "result/$(style_retention)-$(style_mixin_name)-convergence.$format")

    @info "write to" "result/$(style_retention)-$(style_mixin_name)-convergence.$format"
end


if bool_init

    Ψ = BidiagSys(n; style=style_retention)
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
        # ∇ₜ = Diagonal(rand(n, n)) / 2
    else
        throw(ErrorException("not implemented"))
    end
    Fp = z -> F(Ψ, z; ff=style_mixin, Z=Ω)
    Jp = z -> J(Ψ, z; ff=style_mixin, Z=Ω)


    ################################################################################
    # get the fixed-point plots 
    ################################################################################
    N(z, z₊) = Criminos.no_mixed_in(z, z₊; Z=Ω, Ψ=Ψ) #- Criminos.quad_linear(z₊, z; Z=Z, Ψ=Ψ)
    H(z, z₊) = Criminos.pot_gnep(z, z₊; Z=Ω, Ψ=Ψ)
    ∑y(z, z₊) = Criminos.∑y(z, z₊)
    zᵦ = MarkovState(0, [rand(n); rand(n)], ones(n) / 2)

    ################################################################################
    # get the fixed-point plots 
    ################################################################################
    metrics = Dict(
        Criminos.Lₓ => L"\|x - x^*\|",
        Criminos.Lᵨ => L"\|\rho - \rho^*\|",
        # ΔH => L"H - H^*",
        # N => L"\textrm{No-Mixed-In}",
        H => L"H",
        ∑y => L"$\sum y$"
    )
end