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
include("./init.jl")

series_color = palette(:default)
series_size = length(series_color)

################################################################################
τ = ones(n) / 2
xₙ = Int(n // 2)
α₁ = 0.05
α₂ = 0.05
τ[1:xₙ] .= α₁
τ[xₙ:end] .= α₂
z₀ = MarkovState(0, n, τ)

kₑ, z₊, ε, traj, bool_opt = Criminos.simulate(
    z₀, Ψ, Fp; K=K,
    metrics=metrics
)


plot_convergence(traj, ε, kₑ)


if bool_compute
    # store the runs by equilibrium
    runs = Dict()
    pops = Dict()
    ################################################################################
    # get the gradient plot searching over the neighborhood of z₊
    ################################################################################
    sum_population(z) = [z.x'z.ρ; z.x' * (-z.ρ .+ 1)]
    # create a box surrounding z₊  
    x₊, y₊ = z₊.x'z₊.ρ, z₊.x' * (-z₊.ρ .+ 1)
    radius = 2
    xbox = [-x₊:x₊/5:x₊*radius...] .+ x₊
    ybox = [-y₊:y₊/5:y₊*radius...] .+ y₊
    totalsize = length(xbox)
    xbox = xbox[xbox.>0]
    ybox = ybox[ybox.>0]


    model = Criminos.default_xinit_option.model
    p = Progress(totalsize * (length(ybox)); showspeed=true)
    for _x in xbox
        for _y in ybox
            xx, pp = Criminos.find_x(n, _x, _y; x0=z₀.z[1:n])
            if (pp .- 1 |> maximum) > 1e-4
                @warn "skip invalid value"
                continue
            end
            _z = MarkovState(0, [xx; pp], z₀.τ)
            kₑ, z₊, ε, traj, bool_opt = Criminos.simulate(_z, Ψ, Fp; K=K, metrics=metrics)
            if bool_opt
                # only save converged ones
                pps = sum_population.(traj)
                key = tuple(
                    round.(pps[end]; digits=2)...,
                    round(ε[L"H"][kₑ]; digits=2)
                )
                println(key, "\t", length(traj))
                if key in keys(runs)
                    push!(runs[key], traj)
                    push!(pops[key], pps)
                else
                    runs[key] = [traj]
                    pops[key] = [pps]
                end
            end
            ProgressMeter.next!(p)
        end
    end

    ################################################################################
    # summarize the equilibrium
    ################################################################################
    equilibriums = unique(keys(pops))
    @info "\n" "Equilibriums: $equilibriums" "size: $(length(equilibriums))"
end

if bool_plot_trajectory && bool_use_html
    include("tools_traj.jl")
end

savefig(fig3, "result/$style_name-quiver.$format")