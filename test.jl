using ForwardDiff
using LinearAlgebra
using Random
using Printf
using LaTeXStrings
using JuMP
using Criminos
using Plots
using Gurobi

using CSV, Tables, DataFrames

model = Model(Gurobi.Optimizer)
set_attribute(model, "LogToConsole", 0)
set_attribute(model, "OutputFlag", 0)
set_attribute(model, "NonConvex", 2)

function find_x(size, _x, _y; x0=nothing)
    empty!(model)

    # size = ρ |> length

    _c = rand(Float64, size)
    if x0 |> isnothing
        @variable(model, x[1:size] .>= 0)
    else
        x = normalize(x0, 1)
        x *= (_x + _y)
    end
    @variable(model, 1 .>= ρ[1:size] .>= 0)
    @constraint(model, ρ' * x == _x)
    @constraint(model, (1 .- ρ)' * x == _y)
    @objective(model, Max, _c' * x)

    optimize!(model)
    return value.(x), value.(ρ)
end
################################################################################
# !!!todo, change to argparse
################################################################################
# gr()
# plotly()
# style = :fullrg
style = :rand
style_mixin = Criminos.entropy_mixin
# style_mixin = Criminos.binary_zigzag_mixin
style_mixin_name = style_mixin |> nameof
bool_get_gradient_plot = false
bool_use_html = false

if bool_use_html
    plotly()
    format = "html"
else
    pgfplotsx()
    format = "pdf"
end
################################################################################

Random.seed!(2)
n = 8
K = 1e3
kk = 1
specr(x) = maximum(abs.(eigvals(x)))
z₀ = MarkovState(0, n)
# z₀.z[1:n] .*= 0.5
# z₀ = MarkovState(0, z₀.z)
Ψ = BidiagSys(n; style=style)

Z = -(rand(n, n) .* 0.2 .+ 0.6) * 5
# Z = +(rand(n, n) .* 0.2 .+ 0.6) * 5
# Z = +(randn(n, n) .* 1)
normalize!(Z, 2)
Z[diagind(Z)] .= 1 / sqrt(n) * 1.5 * sign.(Z[diagind(Z)])
Z[diagind(Z)] = Z[diagind(Z)] .* (rand(n) .* 0.3 .+ 0.7)
Z = UpperTriangular(Z)
# Z = +(randn(n, n))
# Z = -(Z' * Z / 2) / 5

Fp = z -> F(Ψ, z; ff=style_mixin, Z=Z)
Jp = z -> J(Ψ, z; ff=style_mixin, Z=Z)

################################################################################
# get the fixed-point plots 
################################################################################
H(z, z₊) = Criminos.quad_linear(z, z₊; Z=Z, Ψ=Ψ) - Criminos.quad_linear(z₊, z; Z=Z, Ψ=Ψ)
metrics = Dict(
    Criminos.Lₓ => L"\|x - x^*\|",
    Criminos.Lᵨ => L"\|\rho - \rho^*\|",
    H => L"H - H^*",
)

kₑ, z₊, ε, traj = Criminos.simulate(
    z₀, Ψ, Fp; K=K,
    metrics=metrics
)

rg = 1:kₑ

fig = plot(
    size=(700, 500),
    labelfontsize=20,
    xtickfont=font(25),
    ytickfont=font(25),
    legendfontsize=25,
    titlefontsize=25,
)
for (idx, (func, fname)) in enumerate(metrics)
    plot!(
        fig, rg,
        ε[fname][rg],
        label=fname,
    )
end
title!(fig, "Convergence of the metrics")
savefig(fig, "/tmp/$(style)-$(style_mixin_name)-convergence.$format")

@info "write to" "/tmp/$(style)-$(style_mixin_name)-convergence.$format"


################################################################################
# the entropy potential
################################################################################
# function entropy_mixin_potential(x, ρ, τ)
#     A = LowerTriangular(Z)
#     μ = 0.1
#     _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(ρ)
#     y = x .* ρ
#     _x₊ = _Φ * x + Ψ.λ
#     _y₊ = _Φ * y + Ψ.Q * Ψ.λ
#     φ(x, ρ) = -4 * μ * (τ) .* (A * ρ)
#     return
# end


if bool_get_gradient_plot
    ################################################################################
    # get the gradient plot
    ################################################################################
    sum_population(z) = [z.x'z.ρ; z.x' * (-z.ρ .+ 1)]
    # create a box surrounding z₊  
    x₊, y₊ = z₊.x'z₊.ρ, z₊.x' * (-z₊.ρ .+ 1)
    radius = 2
    xbox = [-x₊:x₊/5:x₊*radius...] .+ x₊
    ybox = [-y₊:y₊/5:y₊*radius...] .+ y₊
    xbox = xbox[xbox.>0]
    ybox = ybox[ybox.>0]

    runs = []
    pops = []
    for _x in xbox
        for _y in ybox
            xx, pp = find_x(n, _x, _y; x0=z₀.z[1:n])
            _z = MarkovState(0, [xx; pp])
            kₑ, z₊, ε, traj = Criminos.simulate(_z, Ψ, Fp; K=K, metrics=metrics)
            push!(runs, traj)
            pps = sum_population.(traj)
            push!(pops, pps)
        end
    end

    ################################################################################
    # summarize the equilibrium
    ################################################################################
    ends = map((x) -> round.(x; digits=2), last.(pops))
    equilibriums = unique(ends)
    @info "\n" "Equilibriums: $equilibriums" "size: $(length(equilibriums))"

    ################################################################################
    # get the quiver plot
    ################################################################################
    if style ∉ [:none]
        cc = [0 0 0 0]
        warmup = 3
        fig3 = plot()
        for idt in 1:10:length(pops)
            global cc
            data = hcat(pops[idt]...)
            # quiver!(
            #     data[1, warmup:end], data[2, warmup:end],
            #     quiver=(
            #         data[1, warmup+1:end] / 10 - data[1, warmup:end-1] / 10,
            #         data[2, warmup+1:end] / 10 - data[2, warmup:end-1] / 10
            #     ),
            x, y = data[1, warmup:end][1:end], data[2, warmup:end][1:end]
            plot!(
                x, y,
                # data[warmup:end, warmup:end], data[warmup:end, warmup:end],
                size=(900, 600),
                labelfontsize=14,
                xtickfont=font(13),
                ytickfont=font(13),
                legendfontsize=14,
                titlefontsize=22,
                label="$idt",
                arrow=true,
                hovers=1:length(x),
                dpi=1000
            )
            # annotate!(
            #     [(data[1, 1], data[2, 1], ("0", 8, 45.0, :bottom, :black))],
            # )

            u = data[1, warmup+1:end] - data[1, warmup:end-1]
            v = data[2, warmup+1:end] - data[2, warmup:end-1]
            # if warmup == 1
            cc = [cc; data[1, warmup:end-1] data[2, warmup:end-1] u v]

            # write out a DataFrame to csv file
            df = DataFrame(cc[2:end, :], :auto)
            CSV.write("$(style)-$(style_mixin_name)-data.csv", df)
        end
        annotate!(
            [(x₊, y₊, ("*", 8, 45.0, :bottom, :black))],
        )
        plot!(showlegend=true)
        @info "Writing out $(style)-$(style_mixin_name)-data.csv"
        # savefig(fig3, "/tmp/quiver.png")
        # savefig(fig3, "/tmp/quiver.pdf")
        savefig(fig3, "/tmp/$(style)-$(style_mixin_name)-quiver.$format")


    end

end