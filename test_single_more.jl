# a script to test the optimization for fitting 
# the correct deterrence parameters

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

bool_run_baseline = true
if bool_run_baseline
    include("./init.jl")
    # ρₛ = rand(n)
    # ρₛ = [1:n...] ./ n ./ 2
    # ρₛ = [1:n...] ./ n ./ 2
    K = 10000
    ρₛ = unimodal(n) ./ 2

    @time begin
        # baseline
        for _z in vec_z
            _z.τ .= 0.3
        end
        _args = tuning(
            n, Σ₁, Σ₂;
            ρₛ=ρₛ,
            τₛ=vcat([_z.τ for _z in vec_z]...),
            style_mixin_monotonicity=cc.style_mixin_monotonicity
        )
        Fp(vec_z) = F!(
            vec_z, vec_Ψ;
            fₜ=Criminos.decision_identity!, targs=(cₜ, cc.τₗ, cc.τₕ),
            fₘ=cc.style_mixin, margs=_args,
        )
        kₑ, ε, traj, bool_opt = Criminos.simulate(
            vec_z, vec_Ψ, Fp; K=K,
            metrics=metrics
        )
    end
end

plot_convergence(ε, vec_z |> length)
r = traj[end]

@info "show equilibrium ≈ ρₛ"

series_color = palette(:default)
fig = plot(
    extra_plot_kwargs=cc.bool_use_html ? KW(
        :include_mathjax => "cdn",
    ) : Dict(),
    labelfontsize=20,
    xtickfont=font(15),
    ytickfont=font(15),
    legendfontsize=20,
    titlefontsize=20,
    xlabel=L"$j$",
    ylabel="Offending Rate",
    legend=:topright,
    legendfonthalign=:left,
    size=(800, 600),
)
plot!(
    1:n, ρₛ,
    label=L"$\rho_s$",
    linewidth=1,
    linestyle=:dot,
)
plot!(
    1:n, traj[end][1].ρ,
    label=L"$\bar{\rho}(\bar\tau)$",
    linewidth=2,
    linestyle=:dash,
    linecolor=series_color[1]
)
plot!(
    1:n, traj[end][1].τ,
    label=L"$\bar\tau$",
    linewidth=2,
    linestyle=:dash,
    linecolor=series_color[1]
)

traj1 = nothing
@time begin
    _vec_z = [copy(_z) for _z in vec_z]
    _jₘ = argmax(ρₛ)
    _ind = Int.([_jₘ-1:_jₘ+1...])
    _bitvec = falses(n)
    _bitvec[_ind] .= true
    x_fill = [_jₘ-1:_jₘ+1...]
    y_fill = ρₛ[_jₘ-1:_jₘ+1]
    plot!(x_fill, y_fill, fillrange=0, fillalpha=0.1, label="")
    for l = 1:4
        for _z in _vec_z
            _z.τ[_bitvec] .+= 0.01 * l
            _z.τ[.!_bitvec] .-= 0.003 * l
        end
        _, _, traj1, _ = Criminos.simulate(
            _vec_z, vec_Ψ, Fp; K=K,
            metrics=metrics
        )
        plot!(
            1:n, traj1[end][1].ρ,
            label=L"$\bar{\rho}(\tau^%$l)$",
            linewidth=2,
            linestyle=:dash,
            linecolor=series_color[l+1],
        )
        plot!(
            1:n, traj1[end][1].τ,
            label=L"$\tau^%$l$",
            linewidth=2,
            linestyle=:dash,
            linecolor=series_color[l+1],
        )
    end

end

savefig(
    fig, "$(cc.result_dir)/param_fitting.$format"
)
