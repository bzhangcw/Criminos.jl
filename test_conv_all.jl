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

for (al, ff) in [
    ("esmall", "confs/conf_esmall.yaml"),
    ("elarge", "confs/conf_elarge.yaml"),
    ("nsmall", "confs/conf_nsmall.yaml"),
    ("nlarge", "confs/conf_nlarge.yaml"),
    ("fsmall", "confs/conf_fsmall.yaml"),
    ("flarge", "confs/conf_flarge.yaml"),
]
    ENV["CRIMINOS_ALIAS"] = al
    ENV["CRIMINOS_CONF"] = ff
    include("./conf.jl")
    include("./tools.jl")
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
end
