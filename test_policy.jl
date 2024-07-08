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
ρₛ = min.(vcat([unimodal(n) ./ cc.group_new_ratio[idx] for idx in 1:ℜ]...), cc.ι * 0.9)
for _z in vec_z
    _z.τ .= 0.4
end
τₛ = vcat([_z.τ for _z in vec_z]...)
@time begin
    # baseline
    _args = tuning(
        n, Σ₁, Σ₂;
        ρₛ=ρₛ,
        τₛ=τₛ,
        style_mixin_monotonicity=cc.style_mixin_monotonicity
    )
    Fp(vec_z) = F!(
        vec_z, vec_Ψ;
        fₜ=Criminos.decision_identity!, targs=(cₜ, cc.τₗ, cc.τₕ),
        fₘ=cc.style_mixin, margs=_args,
    )
    Fpb(vec_z) = F!(
        vec_z, vec_Ψ;
        fₜ=cc.style_decision, targs=(cₜ, cc.τₗ, cc.τₕ),
        fₘ=cc.style_mixin, margs=_args,
    )
end

K = 10000
series_color = palette(:default)
series_size = length(series_color)

if cc.bool_conv
    ################################################################################
    kₑ, ε₁, traj₁, bool_opt = Criminos.simulate(
        vec_z, vec_Ψ, Fp; K=K,
        metrics=metrics
    )
    @info "" ρₛ - [traj₁[end][1].ρ..., traj₁[end][2].ρ...]
    kₑ, ε₂, traj₂, bool_opt = Criminos.simulate(
        traj₁[end], vec_Ψ, Fpb; K=K,
        metrics=metrics
    )
end
ε = Dict()
for (k, v) in ε₁
    ε[k] = [ε₁[k]..., ε₂[k]...]
end
plot_convergence(ε, vec_z |> length)
