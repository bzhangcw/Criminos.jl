module Criminos

greet() = print("Hello World!")

EPS_FP = 1e-7

ℓ = 1
struct BarrierOption
    μ::Float64
end
default_barrier_option = BarrierOption(2)

include("state.jl")
include("bidiag.jl")
include("fixpoint.jl")
include("mixin.jl")
include("potfunc.jl")

ψ_linear(_x, _r, _Φ, _τ, A; Ψ=nothing, baropt=default_barrier_option, kwargs...) = begin
    _x₊ = _Φ * _x + Ψ.λ
    # pure linear function
    _y = _x .* _r #+ Ψ.Q * Ψ.λ
    ∇ = (2*A*_τ+Ψ.Q*Ψ.λ)[:]
    E = ∇' * (_y)
    return ∇, E
end

ψ_quadratic(_x, _r, _Φ, _τ, A; Ψ=nothing, baropt=default_barrier_option, kwargs...) = begin
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    _y = _x .* _r + Ψ.Q * Ψ.λ
    # nvx quadratic
    _T = Diagonal(_τ)
    H = A' * _T + _T * A
    L = opnorm(H)
    ∇ = (0.24/L*H*_y)[:]
    E = ∇' * _y / 2
    return ∇, E
end

ψ_quadratic_cvx(_x, _r, _Φ, _τ, A; Ψ=nothing, baropt=default_barrier_option, kwargs...) = begin
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    _y = _x .* _r + Ψ.Q * Ψ.λ
    # nvx quadratic
    H = A' * Diagonal(_τ .^ 2) * A
    L = opnorm(H)
    ∇ = (0.24/L*H*_y)[:]
    E = ∇' * _y / 2
    return ∇, E
end

ψ_sublinear(_x, _r, _Φ, _τ, A; Ψ=nothing, baropt=default_barrier_option, kwargs...) = begin
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    # quadratic
    _T = diagm(0 => _τ)
    _y = _x .* _r + Ψ.Q * Ψ.λ
    E(x) = 1 / ((A * _T * x) .^ 2 |> sum)
    ∇E(x) = ForwardDiff.gradient(E, x)
    return ∇E(_y), E(_y)
end
# ψ = ψ_linear
# ψ = ψ_quadratic
ψ = ψ_linear
# ψ = ψ_sublinear

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

        # one-step forward
        z₁ = MarkovState(k, Fp(z), z.τ)

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
