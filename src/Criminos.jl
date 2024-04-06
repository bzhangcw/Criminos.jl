module Criminos

greet() = print("Hello World!")


include("state.jl")
include("bidiag.jl")
include("fixpoint.jl")
include("mixin.jl")

# a debugging function on φ
# φ1(_x, _r, _Φ, _Q, _λ, _τ, μ, A, kwargs...) = begin
#     4 * μ * (_τ) .* (A * (_r .^ 2.5) .* (_x .^ 1.9)) # this will give two equilibrium
#     # 4 * μ * (_τ) .* (A * (_r .^ 0.5) .* (_x .^ 0.9)) # this will give two equilibrium
# end

# φ2(_x, _r, _Φ, _Q, _λ, _τ, μ, A, kwargs...) = begin
#     4 * μ * (_τ) .* (A * _r)
# end

φ3(_x, _r, _Φ, _τ, μ, A; Ψ=nothing, kwargs...) = begin
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    (4 * μ * Diagonal(_τ) * A) * _r
end

# from the theory
φ4(_x, _r, _Φ, _τ, A; Ψ=nothing, baropt=default_barrier_option, kwargs...) = begin
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    4 * baropt.μ * (Diagonal(_τ) * A)' * (_x₊ - _x) + Ψ.Γ' * Ψ.M' * _x₊
end
φ = φ3

function simulate(z₀, Ψ, Fp; K=10, metrics=[Lₓ, Lᵨ, ΔR, KL], bool_opt=true)

    z₊ = Criminos.forward(z₀, Fp; K=K)
    ε = Dict()
    for (idx, (func, fname)) in enumerate(metrics)
        ε[fname] = zeros(z₊.k + 1)
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
        for (idx, (func, fname)) in enumerate(metrics)
            ε[fname][k] = func(z, z₊)
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
