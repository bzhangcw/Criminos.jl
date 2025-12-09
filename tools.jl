using LinearAlgebra, SparseArrays, Arpack
using Plots, JSON

switch_to_pdf = () -> begin
    pgfplotsx()
    format = "pdf"
end

function reset_size(n, R)
    cc.n = n
    cc.R = R
    cc.N = R * n
    @info """reset size to
    n: $(cc.n)
    R: $(cc.R)
    N: $(cc.N) 
    """
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
            size=(1000 * (s), 900),
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
fillna(x) = isnan(x) ? 0.0 : x
parseint(i_str) = isa(i_str, Int) ? i_str : parse(Int, i_str)


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

    AgentData(file, Tₘ; n=10, bool_use_yaml=false) = (
        this = new{Vector{Float64},Matrix{Float64}}();
        this.Tₘ = Tₘ;
        if bool_use_yaml
            _construct_callback(this, file; n=n)
        else
            _construct_callback_json(this, file; n=n)
        end
    )
    AgentData(n, Tₘ, traj_x, traj_y) = begin
        this = new{Vector{Float64},Matrix{Float64}}()
        this.n = n
        this.Tₘ = Tₘ
        this.traj_x = traj_x
        @info "" "size of trajectory = (n, Tₘ) = $(size(traj_x))"
        this.traj_y = traj_y
        this.γ = zeros(n)
        this.λ = zeros(n)
        this.traj_x_lb = similar(traj_x)
        this.traj_x_ub = similar(traj_x)
        this.traj_y_lb = similar(traj_y)
        this.traj_y_ub = similar(traj_y)
        return this
    end
end
function _construct_callback(this, file; n=nothing)
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
        @info "size corrected to" this.n
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
    @info "size corrected to" this.n
    return this
end

function _construct_callback_json(this, file; n=10)
    throw(ArgumentError("not implemented"))
end

function _read_json(file; max_n=10)
    # load data
    data = JSON.parsefile(file)
    T = length(data)

    # discover neighborhoods & attrs
    nbs, attrs = Set{String}(), Set{String}()
    for entry in data, (nb, ad) in entry
        push!(nbs, nb)
        for attr in keys(ad)
            push!(attrs, attr)
        end
    end

    # preallocate capped storage
    A = Dict{String,Dict{String,Array{Float64,2}}}()
    Y = Dict{String,Array{Float64,3}}()
    for nb in nbs
        A[nb] = Dict(attr => zeros(T, max_n) for attr in attrs if attr != "y")
        Y[nb] = zeros(T, max_n, max_n)
    end

    # fill with clamping at max_n
    for (t, entry) in enumerate(data)
        for (nb, ad) in entry
            for (attr, val) in ad
                if attr == "y"
                    for tri in val
                        i, j, v = parseint(tri[1]), parseint(tri[2]), tri[3]
                        ii = min(i + 1, max_n)
                        jj = min(j + 1, max_n)
                        Y[nb][t, ii, jj] += v
                    end
                else
                    for (i_str, v) in val
                        i = parseint(i_str)
                        ii = min(i + 1, max_n)
                        A[nb][attr][t, ii] += v
                    end
                end
            end
        end
    end
    return A, Y
end

function _solve_for_gamma(A, Y, nb; optimizer=nothing, epsilon=0.01)

    mean_xe = mean(A[nb]["x0"]; dims=1)[:]
    mean_xa = mean(A[nb]["x"]; dims=1)[:]
    mean_yout = mean(A[nb]["yout"]; dims=1)[:]
    mean_yin = mean(A[nb]["yin"]; dims=1)[:]
    mean_lv = mean(A[nb]["lv"]; dims=1)[:]
    mean_y = mean(Y[nb]; dims=1)[1, :, :]
    mean_λ = mean(A[nb]["lmd"]; dims=1)[:]
    # model
    md = Model(optimizer)
    γ = @variable(md, γ[1:n], lower_bound = 0.0, upper_bound = 0.99)
    ϵ = @variable(md, ϵ, lower_bound = 0.0, upper_bound = epsilon)
    for i in 1:n
        expr = mean_xa[i] - γ[i] * (mean_xe[i] - mean_yout[i] + mean_λ[i] + mean_yin[i])
        @constraint(md, expr ≤ ϵ)
        @constraint(md, expr ≥ -ϵ)
    end
    @objective(md, Min, sum(γ))
    optimize!(md)
    γv = value.(γ)
    γv .= (γv .== 0.0) * 0.5 + γv
    return γv
end

function _solve_for_gamma(mean_xe, mean_xa, mean_yout, mean_yin, mean_λ; optimizer=nothing, epsilon=0.01)

    # model
    md = Model(optimizer)
    γ = @variable(md, γ[1:n], lower_bound = 0.0, upper_bound = 0.99)
    ϵ = @variable(md, ϵ, lower_bound = 0.0, upper_bound = epsilon)
    for i in 1:n
        expr = mean_xa[i] - γ[i] * (mean_xe[i] - mean_yout[i] + mean_λ[i] + mean_yin[i])
        @constraint(md, expr ≤ ϵ)
        @constraint(md, expr ≥ -ϵ)
    end
    @objective(md, Min, sum(γ))
    optimize!(md)
    γv = value.(γ)
    γv .= (γv .== 0.0) * 0.5 + γv
    return γv
end

function plot_fitting_results(data, r; affix="null", bool_pdf=false, bool_traj=false)
    yₛ = data.traj_y[:, end]
    xₛ = data.traj_x[:, end]

    if bool_pdf
        switch_to_pdf()
    else
        plotly()
    end
    series_color = palette(:default)
    fig = generate_empty(cc.bool_use_html)
    plot!(
        1:n, yₛ,
        labels=L"y_T^{\texttt{a}}",
        linestyle=:dot,
        color=series_color[1],
        linewidth=3,
    )
    plot!(
        1:n, data.traj_y_lb[:, end], fillrange=data.traj_y_ub[:, end],
        labels=L"$y_T^\texttt{a}$ (CI)",
        fillalpha=0.3,
        linestyle=:dot,
        color=series_color[1],
    )

    plot!(
        1:n, r.y,
        label=L"$y_T$",
        linewidth=3,
        color=series_color[2]
    )
    fig1 = generate_empty(cc.bool_use_html)
    plot!(
        1:n, xₛ,
        labels=L"x_T^{\texttt{a}}",
        linestyle=:dot,
        color=series_color[1],
        linewidth=3,
    )
    plot!(
        1:n, data.traj_x_lb[:, end], fillrange=data.traj_x_ub[:, end],
        labels=L"$x_T^{\texttt{a}}$ (CI)",
        fillalpha=0.3,
        linestyle=:dot,
        color=series_color[1],
    )

    plot!(
        1:n, r.x,
        label=L"$x_T$",
        linewidth=3,
        color=series_color[2]
    )

    # total number
    fig2 = generate_empty(cc.bool_use_html)

    ysum = [sum(j[1].y) for j in traj]
    L = min(length(ysum), Tₘ)
    plot!(
        sum(data.traj_y; dims=1)[:][1:L],
        labels=L"y_T^{\texttt{a}}",
        linestyle=:dot,
        color=series_color[1],
        linewidth=3,
        xlabel=L"t"
    )
    plot!(
        sum(data.traj_y_lb; dims=1)[:][1:L], fillrange=sum(data.traj_y_ub; dims=1)[:][1:L],
        labels=L"$y_T^{\texttt{a}}$ (CI)",
        fillalpha=0.3,
        linestyle=:dot,
        color=series_color[1],
    )
    plot!(
        ysum[1:L],
        label=L"$y_T$",
        linewidth=3,
        color=series_color[2]
    )
    fig3 = generate_empty(cc.bool_use_html)

    xsum = [sum(j[1].x) for j in traj]
    plot!(
        sum(data.traj_x; dims=1)[:][1:L],
        labels=L"x_T^{\texttt{a}}",
        linestyle=:dot,
        color=series_color[1],
        linewidth=3,
        xlabel=L"t"
    )
    plot!(
        sum(data.traj_x_lb; dims=1)[:][1:L], fillrange=sum(data.traj_x_ub; dims=1)[:][1:L],
        labels=L"$x_T^{\texttt{a}}$ (CI)",
        fillalpha=0.3,
        linestyle=:dot,
        color=series_color[1],
    )

    plot!(
        xsum[1:L],
        label=L"$x_T$",
        linewidth=3,
        color=series_color[2]
    )
    if bool_traj
        ff = plot(fig, fig1, fig2, fig3)
    else
        ff = plot(fig, fig1)
    end
    savefig(ff, "$(cc.result_dir)/fitting-results-$(affix).$format")

    return fig, fig1, fig2, fig3
end


function rate_visualization(rate_vec, V)
    K = length(V[1])
    colnames = Symbol[]
    coldata = Any[]
    if K >= 1
        push!(colnames, :j)
        push!(coldata, [v[1] for v in V])
    end
    if K >= 2
        push!(colnames, :a)
        push!(coldata, [v[2] for v in V])
    end
    for k in 3:K
        push!(colnames, Symbol("d$(k)"))
        push!(coldata, [v[k] for v in V])
    end
    push!(colnames, :rate)
    push!(coldata, rate_vec)

    df_rate_long = DataFrame(coldata, colnames)

    if K >= 2
        df_rate = unstack(df_rate_long, :a, :j, :rate)
    end
    return df_rate
end

function safe_ratio(y, x; ϵ=1e-4)
    return ifelse(abs(x) <= ϵ, 0.0, y / x)
end

function visualize_results(z, τ, data)
    n = data[:n]
    x0 = z.x[:, 1] + z.x[:, 2]
    y0 = z.y[:, 1] + z.y[:, 2]
    x1 = z.x[:, 3] + z.x[:, 4]
    y1 = z.y[:, 3] + z.y[:, 4]
    Z = [data[:V] x0 x1 y0 y1 round.(τ, digits=4) data[:scores_level]]
    df = DataFrame(Z, [:v, :x₀, :x₁, :y₀, :y₁, :τ, :score])
    df[!, :x] = df[!, :x₀] + df[!, :x₁]
    df[!, :y] = df[!, :y₀] + df[!, :y₁]
    return df
end

sum_dict(d) = sum(values(d))
sum_all(d) = sum(sum_dict(d))