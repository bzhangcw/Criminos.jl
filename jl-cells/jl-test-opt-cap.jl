# Comparison of steady-state optimization across capacity values
# This file is included from jl-discrete-random.ipynb

using Plots

# Range of capacity values
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

if bool_init
    T_e = 150.0
    T_f = 1500.0
    # Treatment mode: :existing (existing only), :new (new arrivals only), :both (all), :uponentry
    # mode = :existing  # choose from [:existing, :new, :both, :uponentry]
    mode = :uponentry  # choose from [:existing, :new, :both, :uponentry]
    style_λ = :uniform
    # idiosyncrasy = CR.GammaIdiosyncrasy()
    idiosyncrasy = CR.ConstantIdiosyncrasy(1.7 / 2.05)
    # data = generate_random_data((2, 2), Δ)
    # data = generate_random_data((2, 2), Δ; δ_inc=0.2)
    # data = generate_random_data((5, 4), Δ)
    # data = CR.generate_random_data((7, 3), Δ; δ_inc=0.08, T_f=1000.0, λ_coeff=0.1, style_λ=:j_0_only)
    data = CR.generate_random_data(
        (10, 4), T_e; δ_inc=0.01, T_f=T_f, λ_coeff=0.02,
        style_λ=style_λ,
        idiosyncrasy=idiosyncrasy
    )
    n = data[:n]
    rep = 1 # no need to run multiple times actually
    C = 180 # round(sum(z_final[1:n]) / 25);
    p₀ = ones(n)
    p₁ = p₀ .* 0.342
    ϕ = zeros(n)
    @info """
    C = $C
    p₁ = $p₁
    """
    n = data[:n]
end

using Statistics

if bool_run
    # capacity_values = collect(50:50:800)
    # capacity_values = [50, 100, 200, 300, 400, 450, 500, 600]
    # capacity_values = [50, 200, 300, 400, 500, 600, 800, 1000]
    capacity_values = [20, 40, 80, 90, 100, 200, 300, 400]
    n_cap = length(capacity_values)

    # Store results: each row is a capacity value, each column is a repetition
    mu_null_results = zeros(n_cap, rep)
    mu_sd_results = zeros(n_cap, rep)
    τ_sd_results = zeros(n, n_cap)

    # Store final z states for each capacity value
    z_null_final = Vector{Any}(undef, n_cap)
    z_sd_final = Vector{Any}(undef, n_cap)

    @info "Running comparison for capacity from $(first(capacity_values)) to $(last(capacity_values)) with $rep repetitions..."

    z₀ = randz(n)
    τ₀ = zeros(n)
    for (i, C_val) in enumerate(capacity_values)
        global z₀, τ₀
        # Define null policy function
        fτ_null = (z) -> zeros(n)
        for r in 1:rep
            # Generate new random initial state for each repetition
            z₀_rep = randz(n)

            # Run simulation with policies
            z_null, _ = CR.run(z₀_rep, data; p=p₁, mode=mode, verbose=false, validate=false, fτ=fτ_null)
            τ_sd, _, z_sd, _... = CR.__policy_opt_sd_madnlp_adnlp(copy(z₀), data, C_val, ϕ, p₁; obj_style=1, max_wall_time=300.0, accuracy=1e-9, τ₀=τ₀, mode=mode)

            mu_null_results[i, r] = z_null.μ
            mu_sd_results[i, r] = z_sd.μ
            τ_sd_results[:, i] .= τ_sd

            # Save final z states (last repetition)
            z_null_final[i] = z_null
            z_sd_final[i] = z_sd
            z₀ = copy(z_sd)
            τ₀ .= τ_sd
        end

        @info "C = $C_val: μ_sd = $(mean(mu_sd_results[i, :])) ± $(std(mu_sd_results[i, :])), " *
              "μ_null = $(mean(mu_null_results[i, :])) ± $(std(mu_null_results[i, :]))"
    end

    # Compute mean and confidence intervals (95% CI = mean ± 1.96 * std / sqrt(n))
    z_score = 1.96  # 95% confidence interval

    mu_null_mean = vec(mean(mu_null_results, dims=2))
    mu_null_std = vec(std(mu_null_results, dims=2))
    mu_null_ci = z_score * mu_null_std / sqrt(rep)

    mu_sd_mean = vec(mean(mu_sd_results, dims=2))
    mu_sd_std = vec(std(mu_sd_results, dims=2))
    mu_sd_ci = z_score * mu_sd_std / sqrt(rep)
end

fig = CR.generate_empty(false)

# Compute relative values (difference from null policy)
null_rel_mean = mu_null_mean - mu_null_mean  # zeros
sd_rel_mean = mu_sd_mean - mu_null_mean

# Propagate uncertainty for differences
sd_rel_ci = sqrt.(mu_sd_ci .^ 2 + mu_null_ci .^ 2)

# Create the plot with confidence intervals
Plots.plot!(
    fig,
    capacity_values, null_rel_mean,
    xlabel=L"C \textrm{ (capacity)}",
    ylabel=L"\Delta\mu \textrm{ (relative to null)}",
    linewidth=2,
    legend=:best,
    label=TEX_POL_NULL,
    linecolor=CR.get_policy_color("null"),
)

Plots.plot!(
    fig,
    capacity_values, sd_rel_mean,
    ribbon=sd_rel_ci,
    fillalpha=0.2,
    linewidth=2,
    label=TEX_POL_SD,
    linecolor=CR.get_policy_color("steady-state"),
    fillcolor=CR.get_policy_color("steady-state"),
)

@info "Comparison complete. Results stored in mu_sd_results and mu_null_results."

savefig(fig, "/tmp/jl-test-opt-cap.tex")
savefig(fig, "/tmp/jl-test-opt-cap.png")