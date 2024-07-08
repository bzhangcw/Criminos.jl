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
# !!! set ℜ = 1 first
################################################################################
include("./conf.jl")
# if ℜ > 1
#     @warn "setting ℜ = 1, no policy for decision"
#     exit(0)
# end
include("./init.jl")
include("tools.jl")

series_color = palette(:default)
series_size = length(series_color)
################################################################################

if cc.bool_compute
    # store the runs by equilibrium
    runs = Dict()
    pops = Dict()
    runs_same_α = Dict()
    sum_population(z) = [z.x'z.ρ; z.x' * (-z.ρ .+ 1)]
    ℓ = [0.05:0.05:0.4...]
    h = [0.05:0.05:0.95...]
    totalsize = length(ℓ) * length(h)
    model = Criminos.default_xinit_option.model
    p = Progress(totalsize; showspeed=true)
    for τₗ in ℓ
        for τₕ in h
            τ = ones(n) / 2
            xₙ = Int(n // 2)
            τ[1:xₙ] .= τₗ
            τ[xₙ:end] .= τₕ
            _z = MarkovState(0, zᵦ.z, τ)
            # _z = MarkovState(0, n, τ)
            kₑ, ε, traj, bool_opt = Criminos.simulate(_z, Ψ, Fp; K=K, metrics=metrics)
            traj = traj[1]
            if !bool_opt
                @warn "not converged" τₗ τₕ
                # continue
            end
            # only save converged ones
            pps = sum_population.(traj)
            key = tuple(
                τₗ, τₕ,
            )
            println(key, "\t", length(traj))
            if key in keys(runs)
                push!(runs[key], traj)
                push!(pops[key], pps)
            else
                runs[key] = [traj]
                pops[key] = [pps]
            end
            # group same τₗ
            if τₗ in keys(runs_same_α)
                push!(runs_same_α[τₗ], [pps[end]..., τₕ])
            else
                runs_same_α[τₗ] = [[pps[end]..., τₕ]]
            end

            ProgressMeter.next!(p)
            # break
        end
        # break
    end
end

if bool_plot_trajectory && bool_use_html
    fig3 = plot_trajectory(
        runs, pops, style_name=style_name, format=format,
        bool_show_equilibrium=false,
        bool_label=false
    )
    for (neidx, (key, trajs)) in enumerate(runs_same_α)
        data = hcat(trajs...)
        x, y, τₕ = data[1, :], data[2, :], data[3, :]
        kk = key
        indx = indexin(kk, ℓ)[]
        color = indx == 0 ? :black : series_color[indx%series_size+1]
        plot!(
            fig3,
            x, y,
            markershape=:circle,
            markeralpha=1.0,
            markercolor=color,
            markerstrokecolor=:match,
            linecolor=color,
            label="$kk",
            arrow=true,
            arrowhead=4,
            linewidth=4,
            hovers=string.(τₕ),
        )
    end

    savefig(fig3, "$result_dir/$style_name-quiver.$format")

end

if bool_plot_surface
    @info "Plotting the surface"
    surface_metrics = (
        "clean rate" => (z) -> sum(z.x - z.y) / sum(z.x),
        "population size" => (z) -> sum(z.x)
    )

    surface_names = Dict(
        "clean rate" => L"\text{clean rate}: \sum_j(x_j-y_j)/\sum x_j",
        "population size" => L"\text{population size}: \sum x_j"
    )

    lay = @layout [
        a{0.8h} b{0.8h}
    ]


    fig4 = plot(
        legend=true,
        leg=:topright,
        dpi=1000,
        size=(800, 400),
        legendfonthalign=:left,
        xlabel=L"\tau_\ell",
        ylabel=L"\tau_h",
        layout=length(surface_metrics),
        extra_plot_kwargs=KW(
            :include_mathjax => "cdn",
        ),
        aspect_ratio=0.9
    )
    for (idf, (k, fₘ)) in enumerate(surface_metrics)
        contourfz = [[a, b, fₘ(runs[(a, b)][1][end])] for a in ℓ, b in h]
        if bool_use_html

            contour!(ℓ, h, (x, y) -> fₘ(runs[(x, y)][1][end]),
                fill=true, c=reverse(cgrad(:ice)),
                subplot=idf, title=surface_names[k],
                figsize=(1800, 600),
            )
        else
            contourfz = hcat(contourfz...)
            contour!(ℓ, h, (x, y) -> fₘ(runs[(x, y)][1][end]),
                fill=true, levels=100, c=reverse(cgrad(:ice)),
                subplot=idf, title=surface_names[k]
            )
        end
        df = DataFrame(hcat(contourfz...)', :auto)
        CSV.write("$result_dir/$style_name-contour-$(k).csv", df)
    end
    savefig(fig4, "$result_dir/$style_name-contour.$format")
end