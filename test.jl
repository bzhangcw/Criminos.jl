using ForwardDiff
using LinearAlgebra
using Random
using Printf
using LaTeXStrings
using JuMP
using Criminos
using Plots
using Gurobi
using PGFPlotsX
using CSV, Tables, DataFrames

# pgfplotsx()
# gr()
plotly()

function find_x(size, _x, _y)

    model = Model(Gurobi.Optimizer)
    set_attribute(model, "LogToConsole", true)
    set_attribute(model, "NonConvex", 2)
    # size = ρ |> length
    _c = rand(Float64, size)
    @variable(model, x[1:size] .>= 0)
    @variable(model, 1 .>= ρ[1:size] .>= 0)
    @constraint(model, ρ' * x == _x)
    @constraint(model, (1 .- ρ)' * x == _y)
    @objective(model, Max, _c' * x)

    optimize!(model)
    return value.(x), value.(ρ)
end

style = :fullrg
# style = :rand
Random.seed!(1)

n = 5
K = 1e5
kk = 1
specr(x) = maximum(abs.(eigvals(x)))
z₀ = MarkovState(0, n)
# z₀.z[1:n] .*= 0.5
# z₀ = MarkovState(0, z₀.z)
Ψ = BidiagSys(n; style=style)

Fp = z -> F(Ψ, z; ff=Criminos.linear_mixin)
Jp = z -> J(Ψ, z; ff=Criminos.linear_mixin)


metrics = [
    Criminos.Lₓ,
    Criminos.Lᵨ,
    # Criminos.ΔR,
    # Criminos.KL,
    # Criminos.KLx,
    Criminos.R
]

kₑ, z₊, ε, traj = Criminos.simulate(z₀, Ψ, Fp; K=K, metrics=metrics)
rg = 1:kₑ

fig = plot()
for (idx, func) in enumerate(metrics)
    plot!(
        fig, rg, ε[idx][rg],
        label=string(func),
        yscale=:log10,
        size=(1000, 800),
        labelfontsize=14,
        xtickfont=font(13),
        ytickfont=font(13),
        legendfontsize=14,
        titlefontsize=22,
    )
end
title!(fig, "Convergence of the metrics")
# savefig(fig, "/tmp/convergence.png")
# savefig(fig, "/tmp/$(style)convergence.pdf")



# get the gradient plot
sum_population(z) = [z.x'z.ρ; z.x' * (-z.ρ .+ 1)]
# create a box surrounding z₊  
x₊, y₊ = z₊.x'z₊.ρ, z₊.x' * (-z₊.ρ .+ 1)
xbox = [-2:0.4:2...] .+ x₊
ybox = [-2:0.4:2...] .+ y₊
xbox = xbox[xbox.>0]
ybox = ybox[ybox.>0]

runs = []
pops = []
for _x in xbox
    for _y in ybox
        xx, pp = find_x(n, _x, _y)
        _z = MarkovState(0, [xx; pp])
        kₑ, z₊, ε, traj = Criminos.simulate(_z, Ψ, Fp; K=K, metrics=metrics)
        push!(runs, traj)
        pps = sum_population.(traj)
        push!(pops, pps)
    end
end

if style ∉ [:full, :fullrg]
    cc = [0 0 0 0]
    warmup = 2
    fig3 = plot()
    for idt in 1:3:length(pops)
        global cc
        data = hcat(pops[idt]...)
        # quiver!(
        #     data[1, warmup:end], data[2, warmup:end],
        #     quiver=(
        #         data[1, warmup+1:end] / 10 - data[1, warmup:end-1] / 10,
        #         data[2, warmup+1:end] / 10 - data[2, warmup:end-1] / 10
        #     ),
        plot!(
            data[1, warmup:end], data[2, warmup:end],
            size=(900, 600),
            labelfontsize=14,
            xtickfont=font(13),
            ytickfont=font(13),
            legendfontsize=14,
            titlefontsize=22,
            label="",
            # yscale=:log10,
            # xscale=:log10,
            dpi=1000
        )
        u = data[1, warmup+1:end] - data[1, warmup:end-1]
        v = data[2, warmup+1:end] - data[2, warmup:end-1]
        cc = [cc; data[1, warmup:end-1] data[2, warmup:end-1] u v]

        # write out a DataFrame to csv file
        df = DataFrame(cc[2:end, :], :auto)
        CSV.write("data$(style).csv", df)
    end
    annotate!([(x₊, y₊, (L"$x^*$", 8, 45.0, :bottom, :black))])
    plot!(showlegend=false)

    # savefig(fig3, "/tmp/quiver.png")
    # savefig(fig3, "/tmp/quiver.pdf")

end

if style ∈ [:full, :fullrg]
    cc = [0 0 0 0]
    warmup = 2
    fig3 = plot(showlegend=false)
    for idt in 1:5:length(pops)
        global cc
        data = hcat(pops[idt]...)
        plot!(
            data[1, warmup:end], data[2, warmup:end],
            # quiver=(
            #     data[1, warmup+1:end] / 10 - data[1, warmup:end-1] / 10,
            #     data[2, warmup+1:end] / 10 - data[2, warmup:end-1] / 10
            # ),
            size=(1200, 800),
            labelfontsize=14,
            xtickfont=font(13),
            ytickfont=font(13),
            legendfontsize=14,
            titlefontsize=22,
            # yscale=:log10,
            # xscale=:log10,
            showlegend=false
        )
        u = data[1, warmup+1:end] - data[1, warmup:end-1]
        v = data[2, warmup+1:end] - data[2, warmup:end-1]
        cc = [cc; data[1, warmup:end-1] data[2, warmup:end-1] u v]
        # write out a DataFrame to csv file
        df = DataFrame(cc[2:end, :], :auto)
        CSV.write("data$(style).csv", df)
    end
    annotate!([(x₊, y₊, (L"$x^*$", 8, 45.0, :bottom, :black))])
    plot!(showlegend=false)
end

