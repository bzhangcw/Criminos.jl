using LinearAlgebra, SparseArrays, Arpack

switch_to_pdf = () -> begin
    pgfplotsx()
    format = "pdf"
end
function generate_empty(use_html; title="")
    return plot(
        extra_plot_kwargs=use_html ? Dict(
            :include_mathjax => "cdn",
        ) : Dict(),
        labelfontsize=20,
        xtickfont=font(15),
        ytickfont=font(15),
        legendfontsize=20,
        titlefontsize=20,
        xlabel=L"$j$",
        ylabel="value",
        legend=:topright,
        legendfonthalign=:left,
        title=title,
        size=(800, 600),
    )
end

function plot_convergence(ε, s)
    pls = []
    for id in 1:s
        pl = plot(
            size=(700 * (s), 900),
        )
        for (func, fname) in metrics
            vv = ε[id, fname]
            kₑ = vv |> length
            plot!(
                pl, 1:kₑ,
                vv[1:kₑ],
                label=fname,
                title="group-$id: " * L"$\alpha_1: %$(cc.τₗ), \alpha_2: %$(cc.τₕ)$",
                legend_column=length(metrics)
            )
        end
        push!(pls, pl)
    end
    pl = plot(
        size=(700 * (s + 1), 900),
    )
    push!(pls, pl)
    fig = plot(pls...,
        legend=:bottom,
        labelfontsize=20,
        xtickfont=font(25),
        ytickfont=font(25),
        legendfontsize=25,
        titlefontsize=25,
        extra_plot_kwargs=cc.bool_use_html ? KW(
            :include_mathjax => "cdn",
        ) : Dict(),
        # xscale=:log2,
        # layout=@layout([° °; _ °])
    )
    savefig(fig, "$(cc.result_dir)/convergence.$format")
    @info "write to" "$(cc.result_dir)/convergence.$format"

end


################################################################################
# get the quiver plot
################################################################################
function plot_trajectory(
    runs, pops;
    style_name="quiver",
    format="png",
    bool_label=true,
    bool_show_equilibrium=false,
)
    warmup = 1
    fig3 = plot(
        legend=:outerright,
        legendfonthalign=:left,
        # xscale=:log2,
        yscale=:log2,
        size=(1800, 1200),
        title="Trajectory by $style_name[1:10]",
        extra_plot_kwargs=cc.bool_use_html ? KW(
            :include_mathjax => "cdn",
        ) : Dict(),
    )
    annotated = Set()
    for (neidx, (key, trajs)) in enumerate(pops)
        ((neidx + 1) % ratio_group != 0) && continue
        @printf "printing %d:%s\n" neidx key
        cc = [0 0 0 0]
        ub = min(10, length(trajs))
        bool_to_label = true
        potfunval = key[1:2]
        kk = key[1]
        for idt in shuffle(1:length(trajs))[1:ub]
            data = hcat(trajs[idt]...)

            x, y = data[1, warmup:end][1:end], data[2, warmup:end][1:end]
            plot!(
                fig3,
                x, y,
                labelfontsize=18,
                xtickfont=font(15),
                ytickfont=font(15),
                legendfontsize=14,
                titlefontsize=22,
                xlabel="recidivists",
                ylabel="non-recidivists",
                arrow=true,
                linealpha=0.8,
                dpi=1000,
                label=bool_label ? (@sprintf "%s" key) : "",
                hovers="",
            )
            bool_to_label = false

            u = data[1, warmup+1:end] - data[1, warmup:end-1]
            v = data[2, warmup+1:end] - data[2, warmup:end-1]
            # if warmup == 1
            cc = [cc; data[1, warmup:end-1] data[2, warmup:end-1] u v]

            # write out a DataFrame to csv file
            df = DataFrame(cc[2:end, :], :auto)
            CSV.write("$(cc.result_dir)/traj-data.csv", df)
        end
        annot = key[1:2]
        if annot in annotated
            continue
        end
        if bool_show_equilibrium
            annotate!([
                (trajs[1][end][1], trajs[1][end][2], (L"$\odot$", 22, 45.0, :center, :black)),
                # (trajs[1][end][1], trajs[1][end][2], (L"%$annot", 22, 1.0, :right, :black))
            ],
            )
        end
        push!(annotated, annot)
    end

    @info "Writing out $style_name-quiver.csv"
    return fig3
end


# some empirical distribution tools
unimodal(n) = [sqrt(exp(-abs(j - n / 3.7))) for j in 1:n]

Base.@kwdef mutable struct AgentData{Tv,Tx}
    n::Int
    Tₘ::Int
    γ::Tv
    λ::Tv
    traj_x::Tx
    traj_x_lb::Tx
    traj_x_ub::Tx
    traj_y::Tx
    traj_y_lb::Tx
    traj_y_ub::Tx

    AgentData(yaml, Tₘ; n=nothing) = (
        this = new{Vector{Float64},Matrix{Float64}}();
        this.Tₘ = Tₘ;
        _construct_callback(this; yaml=yaml, n=n)
    )
end
fillna(x) = isnan(x) ? 0.0 : x
function _construct_callback(this; yaml=yaml, n=nothing)
    # maximum number of slots used in the estimation
    Tₘ = this.Tₘ
    if n === nothing
        this.n = n = (hcat(yaml["x"]...)'[:, Tₘ] .> 0) |> findlast |> Int
        this.n = n = (hcat(yaml["x"]...)'[:, Tₘ] .> 0) |> findlast |> Int
        this.traj_x = hcat(yaml["x"]...)'[1:n, 1:Tₘ] .|> fillna
        this.traj_x_lb = hcat(yaml["lb_x"]...)'[1:n, 1:Tₘ] .|> fillna
        this.traj_x_ub = hcat(yaml["ub_x"]...)'[1:n, 1:Tₘ] .|> fillna
        this.traj_y = hcat(yaml["y"]...)'[1:n, 1:Tₘ] .|> fillna
        this.traj_y_lb = hcat(yaml["lb_y"]...)'[1:n, 1:Tₘ] .|> fillna
        this.traj_y_ub = hcat(yaml["ub_y"]...)'[1:n, 1:Tₘ] .|> fillna
        this.γ = yaml["gamma"][1:n] .|> fillna
        this.λ = yaml["lambda"][1:n] .|> fillna
        return this
    end
    this.n = n
    n₁ = (hcat(yaml["x"]...)'[:, Tₘ] .> 0) |> findlast |> Int
    traj_x = hcat(yaml["x"]...)'[1:n₁, 1:Tₘ] .|> fillna
    traj_x_lb = hcat(yaml["lb_x"]...)'[1:n₁, 1:Tₘ] .|> fillna
    traj_x_ub = hcat(yaml["ub_x"]...)'[1:n₁, 1:Tₘ] .|> fillna
    traj_y = hcat(yaml["y"]...)'[1:n₁, 1:Tₘ] .|> fillna
    traj_y_lb = hcat(yaml["lb_y"]...)'[1:n₁, 1:Tₘ] .|> fillna
    traj_y_ub = hcat(yaml["ub_y"]...)'[1:n₁, 1:Tₘ] .|> fillna

    n₁, Tmax = traj_y |> size
    γ = yaml["gamma"][1:n₁] .|> fillna
    λ = yaml["lambda"][1:n₁] .|> fillna
    # expand dims to n
    this.traj_x = [traj_x; zeros(n - n₁, Tmax)]
    this.traj_x_lb = [traj_x_lb; zeros(n - n₁, Tmax)]
    this.traj_x_ub = [traj_x_ub; zeros(n - n₁, Tmax)]
    this.traj_y = [traj_y; zeros(n - n₁, Tmax)]
    this.traj_y_lb = [traj_y_lb; zeros(n - n₁, Tmax)]
    this.traj_y_ub = [traj_y_ub; zeros(n - n₁, Tmax)]
    this.γ = [γ; yaml["gamma"][1] * ones(n - n₁)]
    this.λ = [λ; zeros(n - n₁)]
    return this
end