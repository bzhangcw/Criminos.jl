# Comparison of High-Risk vs Low-Risk policies across delta_inc values
# This file is included from jl-discrete-random.ipynb

using Plots

# Range of delta_inc values from 0 to 0.4
using LinearAlgebra, SparseArrays, CSV
using Criminos
using DataFrames
using LaTeXStrings
import Criminos as CR
include("../plot.jl")
pgfplotsx()


# after running jl-test-high-low-risk.jl
# plot all τ_sd shapes
z₀_rep = randz(n)
# for (i, δ_inc) in enumerate(delta_inc_values)
for (i, C) in enumerate(capacity_values)
    τ_sd = τ_sd_results[:, i]
    z_sd = z_sd_final[i]
    # Heatmap: τ (treatment rate) by j (score) and a (age)
    df = visualize_results(z_sd, τ_sd, data)
    fig = plot_tau_heatmap(df, figsize=(data[:jₘ] * 100, data[:aₘ] * 100))

    save("/tmp/fig_sd_$(C).pdf", fig)
end