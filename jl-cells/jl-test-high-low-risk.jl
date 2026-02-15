# Comparison of High-Risk vs Low-Risk policies across delta_inc values
# This file is included from jl-discrete-random.ipynb

using Plots

# Range of delta_inc values from 0 to 0.4
using LinearAlgebra, SparseArrays, CSV
using Criminos
using DataFrames
using LaTeXStrings
import Criminos as CR

# Load LaTeX label definitions
include("jl-tex.jl")
include("../plot.jl")
pgfplotsx()

bool_run = true
bool_init = true
bool_run_sd = true

if bool_init
    Δ = 100.0
    # ------------------------------------------------------------
    # treatment mode
    # choose from [:existing, :new, :both, uponentry]
    # mode = :uponentry
    mode = :new
    # ------------------------------------------------------------
    # arrival rate pattern
    # choose from :uniform, :j_0_only, :j_high_only
    style_λ = :j_0_only
    # style_λ = :age_1_only
    # style_λ = :uniform
    # style_λ = :special
    # ------------------------------------------------------------
    # idiosyncrasy
    idiosyncrasy = CR.ConstantIdiosyncrasy(2.6 / 2.05)
    # ------------------------------------------------------------
    # data = generate_random_data((2, 2), Δ)
    # data = generate_random_data((2, 2), Δ; δ_inc=0.2)
    # data = generate_random_data((5, 4), Δ)
    # data = CR.generate_random_data(
    #     (25, 6), Δ; δ_inc=0.1, T_f=1500.0, λ_coeff=0.1, style_λ=:j_0_only,
    #     idiosyncrasy=CR.GammaIdiosyncrasy()
    # )
    j_max = 10
    a_max = 4
    # ------------------------------------------------------------
    data = CR.generate_random_data(
        (j_max, a_max), Δ;
        δ_inc=0.08, T_f=365.0, λ_coeff=0.1,
        style_λ=style_λ,
        idiosyncrasy=idiosyncrasy
    )
    n = data[:n]
    C = 80 # round(sum(z_final[1:n]) / 25);
    p₀ = ones(n)
    p₁ = p₀ .* 0.342
    ϕ = zeros(n)
    @info """
    C = $C
    p₁ = $p₁
    """
    n = data[:n]
    z₀ = randz(n)
end

using Statistics

if bool_run
    delta_inc_values = collect(0.01:0.05:1.0)
    n_delta = length(delta_inc_values)

    τ_sd_results = zeros(n, n_delta)

    # Store final z states for each delta_inc value
    z_hr_final = Vector{Any}(undef, n_delta)
    z_lr_final = Vector{Any}(undef, n_delta)
    z_null_final = Vector{Any}(undef, n_delta)
    z_sd_final = Vector{Any}(undef, n_delta)

    # Save original delta_inc
    original_delta_inc = data[:δ_inc]

    @info "Running comparison for delta_inc from 0 to 1.0 with 1 repetitions..."

    for (i, δ_inc) in enumerate(delta_inc_values)
        # Update data with new delta_inc
        data[:δ_inc] = δ_inc

        # Define policy functions for this configuration
        fτ_hr = (z) -> CR.__policy_opt_priority(z, data, C, ϕ, p₁; obj_style=1, verbose=false, ascending=false, mode=mode)
        fτ_lr = (z) -> CR.__policy_opt_priority(z, data, C, ϕ, p₁; obj_style=1, verbose=false, ascending=true, mode=mode)
        fτ_null = (z) -> zeros(n)


        # Generate new random initial state for each repetition
        z₀_rep = randz(n)

        # Run simulation with policies
        z_hr, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_hr)
        z_lr, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_lr)
        z_null, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_null)

        # Save final z states
        z_hr_final[i] = z_hr
        z_lr_final[i] = z_lr
        z_null_final[i] = z_null

        # Compute τ values for high-risk and low-risk policies at final state
        τ_hr, _... = fτ_hr(z_hr)
        τ_lr, _... = fτ_lr(z_lr)

        # Visualize high-risk policy heatmap
        df_hr = visualize_results(z_hr, τ_hr, data)
        fig_hr = plot_tau_heatmap(df_hr, figsize=(data[:jₘ] * 100, data[:aₘ] * 100))
        save("/tmp/fig_tau_heat_hr_$(mode)-$(style_λ)-$(δ_inc).pdf", fig_hr)

        # Visualize low-risk policy heatmap
        df_lr = visualize_results(z_lr, τ_lr, data)
        fig_lr = plot_tau_heatmap(df_lr, figsize=(data[:jₘ] * 100, data[:aₘ] * 100))
        save("/tmp/fig_tau_heat_lr_$(mode)-$(style_λ)-$(δ_inc).pdf", fig_lr)

        # Run steady-state optimization if enabled
        if bool_run_sd
            τ_sd, _, z_sd, _... = CR.__policy_opt_sd_madnlp_adnlp(z₀, data, C, ϕ, p₁; obj_style=1, max_wall_time=1000.0, mode=mode, accuracy=1e-8)
            τ_sd_results[:, i] .= τ_sd
            z_sd_final[i] = z_sd

            # Visualize steady-state policy heatmap
            df_sd = visualize_results(z_sd, τ_sd, data)
            fig_sd = plot_tau_heatmap(df_sd, figsize=(data[:jₘ] * 100, data[:aₘ] * 100))
            save("/tmp/fig_tau_heat_sd_$(mode)-$(style_λ)-$(δ_inc).pdf", fig_sd)
        end


        @info "δ_inc = $δ_inc: μ_hr = $(z_hr_final[i].μ), μ_lr = $(z_lr_final[i].μ), μ_null = $(z_null_final[i].μ)"
    end
end

# Restore original delta_inc
data[:δ_inc] = original_delta_inc

# Performance metrics to plot
perf_metrics = (
    :μ => (z) -> z.μ,
    :x => (z) -> sum(z.x),
    :y => (z) -> sum(z.y),
)

for (metric_name, metric_fn) in perf_metrics
    # Extract metric values from final z states
    vals_hr = [metric_fn(z_hr_final[i]) for i in 1:n_delta]
    vals_lr = [metric_fn(z_lr_final[i]) for i in 1:n_delta]
    vals_null = [metric_fn(z_null_final[i]) for i in 1:n_delta]

    # Compute relative values (difference from null policy)
    hr_rel = vals_hr .- vals_null
    lr_rel = vals_lr .- vals_null
    null_rel = zeros(n_delta)

    fig = CR.generate_empty(false)

    Plots.plot!(
        fig,
        delta_inc_values, null_rel,
        xlabel=L"\delta_{\texttt{inc}}",
        ylabel=L"\Delta %$metric_name \textrm{ (relative to null)}",
        linewidth=2,
        legend=:best,
        label=TEX_POL_NULL,
        linecolor=CR.get_policy_color("null"),
    )

    Plots.plot!(
        fig,
        delta_inc_values, hr_rel,
        fillalpha=0.2,
        linewidth=2,
        label=TEX_POL_HIGH,
        linecolor=CR.get_policy_color("high-risk"),
        fillcolor=CR.get_policy_color("high-risk"),
    )

    Plots.plot!(
        fig,
        delta_inc_values, lr_rel,
        fillalpha=0.2,
        linewidth=2,
        label=TEX_POL_LOW,
        linecolor=CR.get_policy_color("low-risk"),
        fillcolor=CR.get_policy_color("low-risk"),
    )

    if bool_run_sd
        vals_sd = [metric_fn(z_sd_final[i]) for i in 1:n_delta]
        sd_rel = vals_sd .- vals_null
        Plots.plot!(
            fig,
            delta_inc_values, sd_rel,
            fillalpha=0.2,
            linewidth=2,
            label=TEX_POL_SD,
            linecolor=CR.get_policy_color("steady-state"),
            fillcolor=CR.get_policy_color("steady-state"),
        )
    end

    savefig(fig, "/tmp/jl-test-hl-$(mode)-$(style_λ)-$(metric_name).tex")
    savefig(fig, "/tmp/jl-test-hl-$(mode)-$(style_λ)-$(metric_name).png")
end

@info "Comparison complete."