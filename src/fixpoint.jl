using ForwardDiff


function Φ(Ψ::BidiagSys, z::MarkovState)
    _M = Ψ.M
    _Γ = Ψ.Γ
    _r = z.z[z.n+1:2z.n]
    # transition
    _Φ = _Γ - _M * _Γ * Diagonal(_r)
    return _Φ
end
"""
    F(Ψ::BidiagSys, z::MarkovState; ff=Criminos.linear_mixin)

Compute the fix-point iteration given the BidiagSys Ψ and MarkovState z.

# Arguments
- `Ψ::BidiagSys`: The BidiagSys object.
- `z::MarkovState`: The MarkovState object.
- `ff=Criminos.linear_mixin`: The mixin function.

# Returns
- An array containing the values of z=F(z).

"""
function F(Ψ::BidiagSys, z::MarkovState; ff=Criminos.linear_mixin, Z=ones(z.n, z.n) * 0.1, kwargs...)
    _M = Ψ.M
    _Γ = Ψ.Γ
    _λ = Ψ.λ
    _Q = Ψ.Q
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    # transition
    _Φ = Φ(Ψ, z)
    return [Fₓ(_x, _λ, _Φ); ff(z, Ψ, _Φ; Z=Z, kwargs...)]
end

function J(Ψ::BidiagSys, z::MarkovState; ff=Criminos.linear_mixin, Z=ones(z.n, z.n) * 0.1, kwargs...)
    _M = Ψ.M
    _Γ = Ψ.Γ
    _λ = Ψ.λ

    # transition
    _Fp(_z) = (
        _Φ = Φ(Ψ, z);
        [Fₓ(_z[1:z.n], _λ, _Φ); ff(z, Ψ, _Φ; Z=Z, kwargs...)]
    )
    return ForwardDiff.jacobian(_Fp, z.z)

end

# population
function Fₓ(_x, _λ, _Φ)
    _f1 = _Φ * _x + _λ
    return _f1
end


function forward(z₀, F; K=10000, ϵ=1e-6)
    z = copy(z₀)
    fp = 1e6
    for k in 1:K

        z₁ = MarkovState(k, F(z))
        fp = (z₁.z - z.z) |> norm
        if fp ≤ ϵ * sum(z.z)
            @printf("converged in %d iterations\n", k)
            break
        end

        z = copy(z₁)
    end
    @printf("final residual %.1e\n", fp)
    return z
end



############################################
# potential functions
############################################
function Lᵨ(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return LinearAlgebra.norm(_r - _r₊, p) + 1e-9
end

function Lₓ(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return LinearAlgebra.norm(_x - _x₊, p) + 1e-9
end

function R(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return _x'_r
end

function ΔR(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return norm(_x .* _r - _x₊ .* _r₊, p)^2 + 1e-9
end



function KL(z, z₊; p=1)
    _kl = sum(z₊.z .* log.(z₊.z ./ z.z))
    return _kl
end

function KLx(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    # compute KL-divergence of _r and _r₊
    _kl = sum(_x₊ .* log.(_x₊ ./ _x))
    return _kl
end

function KLy(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ

    _y = _x .* _r
    _y₊ = _x₊ .* _r₊
    # compute KL-divergence of _r and _r₊
    _kl = sum(_y₊ .* log.(_y₊ ./ _y))
    return _kl
end

function entropy_xy(z, z₊)
    _τ = z.τ
    _x = z.x
    _r = z.ρ
    _y = _x .* _r
    # A = LowerTriangular(Z)
    # _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(_r)
    # _x₊ = _Φ * _x + Ψ.λ
    # _y₊ = _Φ * _y + Ψ.Q * Ψ.λ
    # #
    # _φ = φ(_x, _r, _Φ, Ψ.Q, Ψ.λ, _τ, μ, A)
    f = _y' * log.(_y) + (_x - _y)' * log.(_x - _y)
    return f
end


function quad_linear(z, z₊;
    baropt=default_barrier_option,
    Z=nothing,
    Ψ=nothing
)
    # do not modify in place
    A = Z

    _τ = z.τ
    _x = z.x
    _r = z.ρ
    _y = _x .* _r

    _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(_r)
    _x₊ = _Φ * _x + Ψ.λ
    L(x) = 1 / 2 * x' * (I - Ψ.Γ) * x + x' * (Ψ.M * Ψ.Γ * _y - Ψ.λ)

    μ = baropt.μ
    _φ = φ(_x, _r, _Φ, _τ, μ, A; Ψ=Ψ)

    _L = L(_x)
    _e = _y' * _φ
    @debug "" (L(_x₊) - L(_x)) (((_x₊ - _x) |> norm)^2) _e _L

    return _e + _L
end
