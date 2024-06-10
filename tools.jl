using LinearAlgebra, SparseArrays, Arpack

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
                title="group-$id: " * L"$\alpha_1: %$(cc.α₁), \alpha_2: %$(cc.α₂)$",
                legend_column=length(metrics)
            )
        end
        push!(pls, pl)
    end
    pl = plot(
        size=(700 * (s), 900),
    )
    # plot!(
    #     pl, 1:2,
    #     1:2,
    #     label="",
    #     alpha=0.0
    # )
    push!(pls, pl)
    fig = plot(pls...,
        legend=:bottom,
        labelfontsize=20,
        xtickfont=font(25),
        ytickfont=font(25),
        # xscale=:log2,
        legendfontsize=25,
        titlefontsize=25,
        extra_plot_kwargs=KW(
            :include_mathjax => "cdn",
        ),
        layout=@layout([° °; _ °])
    )
    savefig(fig, "$(cc.result_dir)/convergence.$format")
    @info "write to" "$(cc.result_dir)/convergence.$format"
end

function generate_Ω(N, n, ℜ)
    G = blockdiag([sparse(Ψ.Γₕ' * inv(I - Ψ.Γ) * Ψ.Γₕ) for Ψ in vec_Ψ]...)
    D, _ = eigs(G)
    D = real(D)
    if cc.style_correlation == :uppertriangular
        ∇₀ = Matrix(G)
        ∇ₜ = UpperTriangular(cc.style_correlation_seed(N, N))
        Hₜ = Symmetric(cc.style_correlation_seed(N, N))
        Hₜ = Hₜ' * Hₜ
        # this is the lower bound to
        #   guarantee the uniqueness of NE
        H₀ = (
            cc.style_correlation_psd ? G + 1e-3 * I(N) : zeros(N, N)
        )
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    elseif cc.style_correlation == :diagonal
        # uncorrelated case
        ∇₀ = Diagonal(style_correlation_seed(N, N))
        ∇ₜ = Diagonal(style_correlation_seed(N, N))
        Hₜ = Diagonal(style_correlation_seed(N, N))
        Hₜ = Hₜ' * Hₜ
        H₀ = cc.style_correlation_psd ? (opnorm(G) + 1e-3) * I(N) : zeros(N, N)
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    else
        throw(ErrorException("not implemented"))
    end
    if !cc.style_correlation_subp
        Hₜ = blockdiag([sparse(Hₜ[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    end
    return Ω, G
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
        extra_plot_kwargs=KW(
            :include_mathjax => "cdn",
        ),
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
            CSV.write("$(style_retention)-$(style_mixin_name)-data.csv", df)
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
