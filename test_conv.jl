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
include("tools_traj.jl")

series_color = palette(:default)
series_size = length(series_color)

################################################################################
τ = ones(n) / 2
xₙ = Int(n // 2)
α₁ = 0.05
α₂ = 0.05
τ[1:xₙ] .= α₁
τ[xₙ:end] .= α₂
zs = [
    MarkovState(0, n, τ) for _ in 1:ℜ
]

kₑ, ε, traj, bool_opt = Criminos.simulate(
    zs, Ψ, Fp; K=1,
    metrics=metrics
)

plot_convergence(ε, zs |> length)
r = traj[end]


if bool_compute
    # store the runs by equilibrium
    runs = Dict()
    pops = Dict()
    ################################################################################
    # get the gradient plot searching over the neighborhood of z₊
    ################################################################################
    sum_population(z) = [z.y |> sum; (z.x - z.y) |> sum]
    xbox = Dict()
    ybox = Dict()
    radius = 2
    maxsize = 1e6
    for (id, z₊) in enumerate(r)
        global maxsize
        # create a box surrounding z₊  
        y₊, x_y₊ = sum_population(z₊)
        _xbox = [-y₊:y₊/5:y₊*radius...] .+ y₊
        _ybox = [-x_y₊:x_y₊/5:x_y₊*radius...] .+ x_y₊
        _xbox = _xbox[_xbox.>0]
        _ybox = _ybox[_ybox.>0]
        xbox[id] = _xbox
        ybox[id] = _ybox
        maxsize = min(maxsize, length(_xbox), length(_ybox)) |> Integer
    end
    for (id, z₊) in enumerate(r)
        xbox[id] = xbox[id][1:maxsize]
        ybox[id] = ybox[id][1:maxsize]
    end

    @info "plotting over size: $(maxsize^2)"
    model = Criminos.default_xinit_option.model
    p = Progress(maxsize^2; showspeed=true)
    for i in 1:maxsize
        for j in 1:maxsize
            Vz = Vector{MarkovState{Float64,Vector{Float64}}}(undef, length(r))
            for id in 1:length(r)
                xx, yy = Criminos.find_x(n, xbox[id][i], ybox[id][i]; x0=zs[id].z[1:n])
                if ((yy ./ xx) .- 1 |> maximum) > 1e-4
                    @warn "skip invalid value"
                    continue
                end
                _z = MarkovState(0, [xx; yy ./ xx], zs[id].τ)
                Vz[id] = _z
            end
            kₑ, ε, traj, bool_opt = Criminos.simulate(Vz, Ψ, Fp; K=K, metrics=metrics)
            traj = hcat(traj...)
            if bool_opt
                # only save converged ones
                for id in 1:length(r)
                    pps = sum_population.(traj[id, :])
                    key = tuple(
                        id,
                        round.(pps[end]; digits=2)...,
                        round(ε[id, L"H"][kₑ]; digits=2)
                    )
                    println(key, "\t", length(traj[id, :]))
                    if key in keys(runs)
                        push!(runs[key], traj)
                        push!(pops[key], pps)
                    else
                        runs[key] = [traj]
                        pops[key] = [pps]
                    end
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
    fig3 = plot_trajectory(
        runs, pops, style_name=style_name, format=format,
        bool_show_equilibrium=true,
    )
    savefig(fig3, "result/$style_name-quiver.$format")
end
