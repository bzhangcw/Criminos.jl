

################################################################################
# get the quiver plot
################################################################################
warmup = 1
fig3 = plot(
    legend=:outerright,
    legendfonthalign=:left,
    xscale=:log2,
    yscale=:log2,
    size=(1800, 1200),
    title="Trajectory by $style_name",
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
            label=(@sprintf "%s" key),
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
    # annotate!(
    #     [(trajs[1][end][1], trajs[1][end][2], (L"$\odot$", 22, 45.0, :center, :black)),
    #     (trajs[1][end][1], trajs[1][end][2], (L"%$annot", 22, 1.0, :right, :black))
    # ],
    # )
    push!(annotated, annot)
end

@info "Writing out $style_name-quiver.csv"


