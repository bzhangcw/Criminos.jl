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

pgfplotsx()

bool_run = true
bool_init = true
bool_run_sd = false

if bool_init
    Δ = 100.0
    # Treatment mode: :existing (existing only), :new (new arrivals only), :both (all), :uponentry
    # mode = :uponentry  # choose from [:existing, :new, :both, uponentry]
    mode = :both  # choose from [:existing, :new, :both, uponentry]
    # data = generate_random_data((2, 2), Δ)
    # data = generate_random_data((2, 2), Δ; δ_inc=0.2)
    # data = generate_random_data((5, 4), Δ)
    # data = CR.generate_random_data(
    #     (25, 6), Δ; δ_inc=0.1, T_f=1500.0, λ_coeff=0.1, style_λ=:j_0_only,
    #     idiosyncrasy=CR.GammaIdiosyncrasy()
    # )
    data = CR.generate_random_data(
        (25, 6), Δ; δ_inc=0.08, T_f=365.0, λ_coeff=0.1, style_λ=:j_0_only,
        idiosyncrasy=CR.ConstantIdiosyncrasy(2.6 / 2.05)
    )
    n = data[:n]
    rep = 1 # no need to run multiple times actually
    C = 100 # round(sum(z_final[1:n]) / 25);
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
    delta_inc_values = collect(0.01:0.01:0.16)
    n_delta = length(delta_inc_values)

    # Store results: each row is a delta_inc, each column is a repetition
    mu_hr_results = zeros(n_delta, rep)
    mu_lr_results = zeros(n_delta, rep)
    mu_null_results = zeros(n_delta, rep)
    mu_sd_results = zeros(n_delta, rep)
    τ_sd_results = zeros(n, n_delta)

    # Store final z states for each delta_inc value
    z_hr_final = Vector{Any}(undef, n_delta)
    z_lr_final = Vector{Any}(undef, n_delta)
    z_null_final = Vector{Any}(undef, n_delta)
    z_sd_final = Vector{Any}(undef, n_delta)

    # Save original delta_inc
    original_delta_inc = data[:δ_inc]

    @info "Running comparison for delta_inc from 0 to 0.4 with $rep repetitions..."

    for (i, δ_inc) in enumerate(delta_inc_values)
        # Update data with new delta_inc
        data[:δ_inc] = δ_inc

        # Define policy functions for this configuration
        fτ_hr = (z) -> CR.__policy_opt_priority(z, data, C, ϕ, p₁; obj_style=1, verbose=false, ascending=false, mode=mode)
        fτ_lr = (z) -> CR.__policy_opt_priority(z, data, C, ϕ, p₁; obj_style=1, verbose=false, ascending=true, mode=mode)
        fτ_null = (z) -> zeros(n)

        for r in 1:rep
            # Generate new random initial state for each repetition
            z₀_rep = randz(n)

            # Run simulation with policies
            z_hr, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_hr)
            z_lr, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_lr)
            z_null, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_null)

            mu_hr_results[i, r] = z_hr.μ
            mu_lr_results[i, r] = z_lr.μ
            mu_null_results[i, r] = z_null.μ

            # Save final z states (last repetition)
            z_hr_final[i] = z_hr
            z_lr_final[i] = z_lr
            z_null_final[i] = z_null

            # Run steady-state optimization if enabled
            if bool_run_sd
                τ_sd, _, z_sd, _... = CR.__policy_opt_sd_madnlp_adnlp(z₀, data, C, ϕ, p₁; obj_style=1, max_wall_time=1000.0, mode=mode)
                mu_sd_results[i, r] = z_sd.μ
                τ_sd_results[:, i] .= τ_sd
                z_sd_final[i] = z_sd
            end
        end

        @info "δ_inc = $δ_inc: μ_hr = $(mean(mu_hr_results[i, :])) ± $(std(mu_hr_results[i, :])), " *
              "μ_lr = $(mean(mu_lr_results[i, :])) ± $(std(mu_lr_results[i, :])), " *
              "μ_null = $(mean(mu_null_results[i, :])) ± $(std(mu_null_results[i, :]))"
    end

    # Compute mean and confidence intervals (95% CI = mean ± 1.96 * std / sqrt(n))
    z_score = 1.96  # 95% confidence interval

    mu_hr_mean = vec(mean(mu_hr_results, dims=2))
    mu_hr_std = vec(std(mu_hr_results, dims=2))
    mu_hr_ci = z_score * mu_hr_std / sqrt(rep)

    mu_lr_mean = vec(mean(mu_lr_results, dims=2))
    mu_lr_std = vec(std(mu_lr_results, dims=2))
    mu_lr_ci = z_score * mu_lr_std / sqrt(rep)

    mu_null_mean = vec(mean(mu_null_results, dims=2))
    mu_null_std = vec(std(mu_null_results, dims=2))
    mu_null_ci = z_score * mu_null_std / sqrt(rep)

    if bool_run_sd
        mu_sd_mean = vec(mean(mu_sd_results, dims=2))
        mu_sd_std = vec(std(mu_sd_results, dims=2))
        mu_sd_ci = z_score * mu_sd_std / sqrt(rep)
    end
end

# Restore original delta_inc
data[:δ_inc] = original_delta_inc

fig = CR.generate_empty(false)

# Compute relative values (difference from null policy)
hr_rel_mean = mu_hr_mean - mu_null_mean
lr_rel_mean = mu_lr_mean - mu_null_mean
null_rel_mean = mu_null_mean - mu_null_mean  # zeros

# Propagate uncertainty for differences
hr_rel_ci = sqrt.(mu_hr_ci .^ 2 + mu_null_ci .^ 2)
lr_rel_ci = sqrt.(mu_lr_ci .^ 2 + mu_null_ci .^ 2)
if bool_run_sd
    sd_rel_mean = mu_sd_mean - mu_null_mean
    sd_rel_ci = sqrt.(mu_sd_ci .^ 2 + mu_null_ci .^ 2)
end

# Create the plot with confidence intervals
Plots.plot!(
    fig,
    delta_inc_values, null_rel_mean,
    xlabel=L"\delta_{\texttt{inc}}",
    ylabel=L"\Delta\mu \textrm{ (relative to null)}",
    linewidth=2,
    legend=:best,
    label=TEX_POL_NULL,
    linecolor=CR.get_policy_color("null"),
)

Plots.plot!(
    fig,
    delta_inc_values, hr_rel_mean,
    # ribbon=hr_rel_ci,
    fillalpha=0.2,
    linewidth=2,
    label=TEX_POL_HIGH,
    linecolor=CR.get_policy_color("high-risk"),
    fillcolor=CR.get_policy_color("high-risk"),
)

Plots.plot!(
    fig,
    delta_inc_values, lr_rel_mean,
    # ribbon=lr_rel_ci,
    fillalpha=0.2,
    linewidth=2,
    label=TEX_POL_LOW,
    linecolor=CR.get_policy_color("low-risk"),
    fillcolor=CR.get_policy_color("low-risk"),
)

if bool_run_sd
    Plots.plot!(
        fig,
        delta_inc_values, sd_rel_mean,
        # ribbon=sd_rel_ci,
        fillalpha=0.2,
        linewidth=2,
        label=TEX_POL_SD,
        linecolor=CR.get_policy_color("steady-state"),
        fillcolor=CR.get_policy_color("steady-state"),
    )
end

# Also create a difference plot (high-risk minus low-risk)
diff_mean = mu_hr_mean - mu_lr_mean
diff_ci = sqrt.(mu_hr_ci .^ 2 + mu_lr_ci .^ 2)

fig_diff = Plots.plot(
    delta_inc_values, diff_mean,
    ribbon=diff_ci,
    fillalpha=0.2,
    xlabel=L"\delta_{\texttt{inc}}",
    ylabel=L"\Delta\mu",
    title=L"Difference: $\mu_{\texttt{HR}} - \mu_{\texttt{LR}}$",
    linewidth=2,
    label="HR - LR",
    linecolor=CR.get_policy_color("high-risk"),
    fillcolor=CR.get_policy_color("high-risk"),
)

hline!(fig_diff, [0], linestyle=:dash, color=:gray, label="")


@info "Comparison complete. Results stored in mu_hr_results and mu_lr_results."

savefig(fig_diff, "/tmp/jl-test-high-low-risk-diff.tex")
savefig(fig_diff, "/tmp/jl-test-high-low-risk-diff.png")
savefig(fig, "/tmp/jl-test-high-low-risk.tex")
savefig(fig, "/tmp/jl-test-high-low-risk.png")