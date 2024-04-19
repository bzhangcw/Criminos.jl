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

using CSV, Tables, DataFrames

################################################################################
# !!!todo, change to argparse
################################################################################
# gr()
# plotly()
# style = :fullrg
style = :rand
# style_mixin = Criminos.identity_mixin
style_mixin = Criminos.mixed_in_gnep_best
# style_mixin = Criminos.mixed_in_gnep_grad
# style_mixin = Criminos.entropy_mixin
# style_mixin = Criminos.binary_zigzag_mixin
style_mixin_name = style_mixin |> nameof
bool_get_gradient_plot = true
bool_use_html = false
style_treatement = :none

if bool_use_html
    plotly()
    format = "html"
else
    pgfplotsx()
    format = "pdf"
end
################################################################################

Random.seed!(5)
n = 8
K = 1e3
kk = 1
specr(x) = maximum(abs.(eigvals(x)))

if style_treatement == :uniform
    τ = ones(n) / 2
    α₁ = α₂ = 0.2
else
    τ = ones(n) / 2
    xₙ = Int(n // 2)
    α₁ = 0.05
    α₂ = 0.05
    τ[1:xₙ] .= α₁
    τ[xₙ:end] .= α₂
end

z₀ = MarkovState(0, n, τ)
Ψ = BidiagSys(n; style=style)

# asymmetric
A = (rand(n, n))
B = (rand(n, n))
C = Symmetric(randn(n, n))
C = (zeros(n, n))
C = C' * C
C = C' * C + 100 * I

Fp = z -> F(Ψ, z; ff=style_mixin, Z=(A, B, C))
Jp = z -> J(Ψ, z; ff=style_mixin, Z=(A, B, C))

################################################################################
# get the fixed-point plots 
################################################################################
N(z, z₊) = Criminos.no_mixed_in(z, z₊; Z=(A, B, C), Ψ=Ψ) #- Criminos.quad_linear(z₊, z; Z=Z, Ψ=Ψ)
H(z, z₊) = Criminos.pot_gnep(z, z₊; Z=(A, B, C), Ψ=Ψ)
∑y(z, z₊) = Criminos.∑y(z, z₊)
metrics = Dict(
    Criminos.Lₓ => L"\|x - x^*\|",
    Criminos.Lᵨ => L"\|\rho - \rho^*\|",
    # ΔH => L"H - H^*",
    # N => L"\textrm{No-Mixed-In}",
    H => L"H",
    ∑y => L"$\sum y$")

kₑ, z₊, ε, traj, bool_opt = Criminos.simulate(
    z₀, Ψ, Fp; K=K,
    metrics=metrics
)


# dx, dy = Criminos.kkt_box_opt(z₊; Z=Z, Ψ=Ψ)

rg = 2:kₑ

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
    totalsize = length(xbox)
    xbox = xbox[xbox.>0]
    ybox = ybox[ybox.>0]

    runs = Dict()
    pops = Dict()
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
                # only save optimal ones
                pps = sum_population.(traj)
                key = tuple(round.(pps[end]; digits=2)...)
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

    ################################################################################
    # get the quiver plot
    ################################################################################
    zq = nothing
    if style ∉ [:none]
        cc = [0 0 0 0]
        warmup = 3
        fig3 = plot(
            title=L"Trajectories: $\alpha_\ell$: %$α₁ $\alpha_h$: %$α₂",
            legend=false
        )
        for (key, trajs) in pops
            ub = min(10, length(trajs))
            for idt in shuffle(1:length(trajs))[1:ub]
                global cc, zq
                data = hcat(trajs[idt]...)
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
                    labelfontsize=18,
                    xtickfont=font(15),
                    ytickfont=font(15),
                    legendfontsize=14,
                    titlefontsize=22,
                    xlabel="recidivists",
                    ylabel="non-recidivists",
                    arrow=true,
                    linealpha=0.8,
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
                [(trajs[1][end][1], trajs[1][end][2], (L"$\odot$", 22, 45.0, :center, :black)),
                (trajs[1][end][1], trajs[1][end][2], (L"%$key", 22, 1.0, :right, :black))
            ],
            )
        end
        plot!(showlegend=true)
        @info "Writing out $(style)-$(style_mixin_name)-data.csv"
        # savefig(fig3, "/tmp/quiver.png")
        # savefig(fig3, "/tmp/quiver.pdf")
        savefig(fig3, "/tmp/$(style)-$(style_mixin_name)-quiver.$format")
        run(`cp /tmp/$(style)-$(style_mixin_name)-quiver.$format /tmp/$α₁-$α₂.$format`)


    end

end