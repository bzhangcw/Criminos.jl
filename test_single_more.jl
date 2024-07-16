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
bool_plot_ushape = false
bool_plot_myopt = false
bool_plot_opt = true
if bool_run_baseline
    include("./init.jl")
    K = 10000
    # ρₛ = unimodal(n) ./ 2
    ρₛ = rand(n) .* 0.1
    # ρₛ[2] = 0.8
    ρₛ[end-2] = 0.8

    @time begin
        # baseline
        for _z in vec_z
            _z.τ .= 0.3
        end
        _args = tuning(
            n, Σ, Σ₁, Σ₂, V;
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
r = traj[end][1]

@info "show equilibrium ≈ ρₛ" r.ρ - ρₛ

if bool_plot_ushape
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
        # yscale=:log10,
        size=(800, 600),
    )
    traj1 = nothing
    L = 2
    @time begin
        _vec_z = [copy(_z) for _z in vec_z]
        ys = []
        al = 0.01:0.1:5
        for l in al
            global traj1
            for _z in _vec_z
                _z.τ .= l * vec_z[1].τ
            end
            _, _, traj1, _ = Criminos.simulate(
                _vec_z, vec_Ψ, Fp; K=K,
                metrics=metrics
            )
            push!(ys, sum(traj1[end][1].y))
        end
        plot!(
            al, ys,
            xlabel=L"$\alpha $",
            ylabel=L"$\sum y(\alpha)$",
            linewidth=3,
            label=""
        )
    end
    savefig(
        fig, "$(cc.result_dir)/param_fitting_ushape.$format"
    )
end
if bool_plot_myopt
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
        # yscale=:log10,
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
    # plot!(
    #     1:n, traj[end][1].τ,
    #     label=L"$\bar\tau$",
    #     linewidth=2,
    #     linestyle=:dash,
    #     linecolor=series_color[1]
    # )

    traj1 = nothing
    L = 2
    @time begin
        _vec_z = [copy(_z) for _z in vec_z]
        _jₘ = argmax(ρₛ)
        _ind = Int.([_jₘ-1:_jₘ+1...])
        _bitvec = falses(n)
        _bitvec[_ind] .= true
        x_fill = [_jₘ-1:_jₘ+1...]
        y_fill = ρₛ[_jₘ-1:_jₘ+1]
        plot!(x_fill, y_fill, fillrange=0, fillalpha=0.1, label="")
        for l = 1:L
            global traj1
            for _z in _vec_z
                # _z.τ[_bitvec] .+= 0.1 * l
                # _z.τ[.!_bitvec] .-= 0.05 * l
                # _z.τ[end-3:end] .+= 0.1 * l
                # _z.τ[1:4] .+= 0.1 * l
                _z.τ[_jₘ] += 0.8 * l
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
            # plot!(
            #     1:n, traj1[end][1].τ,
            #     label=L"$\tau^%$l$",
            #     linewidth=2,
            #     linestyle=:dash,
            #     linecolor=series_color[l+1],
            # )
        end
    end
    savefig(
        fig, "$(cc.result_dir)/param_fitting.$format"
    )
end

if bool_plot_opt
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
        ylabel="value",
        legend=:topright,
        legendfonthalign=:left,
        title=L"\alpha\cdot c^Tz^+ \leq c^T\bar z^+",
        size=(800, 600),
    )
    plot!(
        1:n, ρₛ,
        label=L"$\rho_s$",
        linewidth=1,
        linestyle=:dot,
    )
    plot!(
        1:n, r.y,
        label=L"$y_s$",
        linewidth=1,
        linestyle=:dot,
    )
    c = rand(N)
    ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B, md = _args

    b = c' * expt_inc(vec_z[1].τ)
    σ₁ = (ones(N)' * _B * Σ₁)'
    σ₂ = (ones(N)' * _B * Σ₂)'
    @time begin
        for alpha = 1:5
            m = Model(optimizer_with_attributes(
                () -> Gurobi.Optimizer(),
                "NonConvex" => 2
            ))


            @variable(m, z₊[1:N] .>= 0)
            set_upper_bound.(z₊, expt_inc(ones(N)))
            knapsack = @constraint(m, alpha * c' * z₊ ≤ b)
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
            _vec_z = [copy(_z) for _z in vec_z]
            # _jₘ = argmax(ρₛ)
            # _ind = Int.([_jₘ-1:_jₘ+1...])
            # _bitvec = falses(n)
            # _bitvec[_ind] .= true
            # x_fill = [_jₘ-1:_jₘ+1...]
            # y_fill = ρₛ[_jₘ-1:_jₘ+1]
            # plot!(x_fill, y_fill, fillrange=0, fillalpha=0.1, label="")

            for _z in _vec_z
                _z.τ .= τₒ
            end
            _, _, traj1, _ = Criminos.simulate(
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
        end
    end

    savefig(
        fig, "$(cc.result_dir)/param_fitting_opt.$format"
    )
end
