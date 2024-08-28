ENV["CRIMINOS_CONF"] = "confs/conf_single_nij.yaml"
ENV["CRIMINOS_ALIAS"] = "test"
Base.istextmime(::MIME"application/vnd.plotly.v1+json") = true
using Revise
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

include("./conf.jl")
include("./tools.jl")
include("./fit.jl")
include("./init.jl")

bool_run_baseline = true
bool_plot_ushape = false
bool_plot_myopt = false
bool_plot_opt = true

if bool_run_baseline
    K = 20000
    # ρₛ = unimodal(n) ./ 2
    # ρₛ = rand(n) .* 0.1
    # ρₛ[2] = 0.8
    # ρₛ[end-2] = 0.8
    ρₛ = cc.variables_from_yaml["rate1"]
    xₛ = cc.variables_from_yaml["pop"]
    yₛ = cc.variables_from_yaml["y1"]

    @time begin
        # baseline
        for _z in vec_z
            _z.τ .= 0.3
        end
        _args = tuning(
            n, Σ, Σ₁, Σ₂, V;
            ρₛ=ρₛ,
            xₛ=xₛ,
            yₛ=yₛ,
            τₛ=vcat([_z.τ for _z in vec_z]...),
            style_mixin_monotonicity=cc.style_mixin_monotonicity
        )
    end
end

Fp(vec_z) = F!(
    vec_z, vec_Ψ;
    fₜ=Criminos.decision_identity!, targs=(cₜ, cc.τₗ, cc.τₕ),
    fₘ=cc.style_mixin, margs=_args,
)
kₑ, ε, traj, bool_opt = Criminos.simulate(
    vec_z, vec_Ψ, Fp; K=K,
    metrics=metrics
);

plot_convergence(ε, vec_z |> length)
# the equilibria
r = traj[end][1]
z₀ = traj[end];


series_color = palette(:default)
fig = generate_empty(cc.bool_use_html)
plot!(
    1:n, yₛ,
    label=L"$y_s^*$",
    linewidth=1,
    linestyle=:dot,
)
plot!(
    1:n, r.y,
    label=L"$y_s$",
    linewidth=1,
    linestyle=:dot,
)
plot!(
    1:n, r.τ,
    label=L"$\tau_s$",
    linewidth=1,
    linestyle=:dot,
)

E = []
ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B, md = _args

c = ones(N)
b = c' * expt_inc(vec_z[1].τ)
σ₁ = (ones(N)' * _B * Σ₁)'
σ₂ = (ones(N)' * _B * Σ₂)'
@time begin
    for alpha = 1:5
        global traj1
        m = Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(),
            "NonConvex" => 2
        ))


        @variable(m, z₊[1:N] .>= 0)
        set_upper_bound.(z₊, expt_inc(ones(N)))
        knapsack = @constraint(m, 1.5^alpha * c' * z₊ ≤ b)
        @objective(m, Min, σ₁' * z₊ + σ₂' * (1 .- z₊) .^ 2)
        optimize!(m)

        zp = value.(z₊)
        zn = (1 .- zp) .^ 2
        τₒ = inv_expt_inc(zp)

        plot!(
            1:n, τₒ,
            label=L"$\tau(\alpha=%$alpha)$",
            linewidth=2,
            linestyle=:dash,
            linecolor=series_color[alpha+1],
        )

        # evaluation
        _vec_z = [copy(_z) for _z in z₀]

        for _z in _vec_z
            _z.τ .= τₒ
        end
        _, ε1, traj1, _ = Criminos.simulate(
            _vec_z, vec_Ψ, Fp; K=K,
            metrics=metrics
        )
        plot!(
            1:n, traj1[end][1].y,
            label=L"$y(\alpha=%$alpha)$",
            linewidth=2,
            linestyle=:dash,
            linecolor=series_color[alpha+1],
        )
        # we also keep trajectory of y:
        push!(E, [
            ε[(1, L"$\sum y$")]...,
            ε1[(1, L"$\sum y$")]...
        ]
        )
    end
end

savefig(
    fig, "$(cc.result_dir)/param_fitting_opt.$format"
)

fig1 = generate_empty(cc.bool_use_html)
for alpha = 1:5
    plot!(E[alpha][1:end], label="α=$alpha")
end
savefig(
    fig1, "$(cc.result_dir)/param_fitting_opt_conv.$format"
)