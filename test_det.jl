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

################################################################################
include("./conf.jl")
include("./tools.jl")
include("./fit.jl")
include("./init.jl")

K = 10000
series_color = palette(:default)
series_size = length(series_color)

if cc.bool_conv
    ################################################################################
    kₑ, ε, traj, bool_opt = Criminos.simulate(
        vec_z, vec_Ψ, Fp; K=K,
        metrics=metrics
    )
end

plot_convergence(ε, vec_z |> length)
r = traj[end]

@info "show equilibrium ≈ ρₛ"

fig = plot(
    extra_plot_kwargs=cc.bool_use_html ? KW(
        :include_mathjax => "cdn",
    ) : Dict(),
    labelfontsize=20,
    xtickfont=font(22),
    ytickfont=font(22),
    legendfontsize=22,
    titlefontsize=22,
    legend=:topright,
    legendfonthalign=:left,
)
plot!(1:n, ρₛ, label=L"$\rho_s$")
plot!(1:n, traj[end][1].ρ, label=L"$\bar{x}/\bar{y}$")
savefig(
    fig, "$(cc.result_dir)/fitting.$format"
)
