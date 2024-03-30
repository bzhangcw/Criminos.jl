module Criminos

greet() = print("Hello World!")


include("state.jl")
include("bidiag.jl")
include("fixpoint.jl")
include("mixin.jl")



function simulate(z₀, Ψ, Fp; K=10, metrics=[Lₓ, Lᵨ, ΔR, KL], bool_opt=true)

    z₊ = Criminos.forward(z₀, Fp)
    ε = Dict()
    for (idx, _) in enumerate(metrics)
        ε[idx] = zeros(z₊.k)
    end

    # @printf(
    #     # "%3s | %31s | %7s | %4s | %4s | %4s | %4s   \n",
    #     "%3s | %7s | %7s | %4s | %4s | %4s | %4s   \n",
    #     "k",
    #     # "jac diff (ad/analytical)",
    #     "ε₁", "ε₂", "ρ(Φ)", "|Φ|", "ρ(J)", "|J|",
    # )

    z = copy(z₀)
    traj = [z]
    kₑ = 0
    for k in 1:z₊.k+1
        for (idx, func) in enumerate(metrics)
            ε[idx][k] = func(z, z₊)
        end
        # ε₁[k] = Criminos.Lₓ(z, z₊; p=1)
        # ε₂[k] = Criminos.Lᵨ(z, z₊)
        # ε₃[k] = Criminos.ΔR(z, z₊)
        # R[k] = Criminos.R(z, z₊)
        # @printf(
        #     # "%3d | %.1e %.1e %.1e %.1e | %.1e | %.2f | %.2f | %.2f | %.2f\n",
        #     "%3d | %.2f | %.2f | %.2f | %.2f\n",
        #     k,
        #     specr(_Φ),
        #     opnorm(_Φ),
        #     specr(jac),
        #     opnorm(jac),
        # )

        # one-step forward
        z₁ = MarkovState(k, Fp(z))

        kₑ = k
        push!(traj, z)
        if (z₁.z - z.z) |> norm < 1e-7
            break
        end
        k += 1

        z = z₁
    end
    return kₑ, z₊, ε, traj
end


export MarkovState
export BidiagSys
export F, Φ, J
end # module Criminos
