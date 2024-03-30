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
function F(Ψ::BidiagSys, z::MarkovState; ff=Criminos.linear_mixin)
    _M = Ψ.M
    _Γ = Ψ.Γ
    _λ = Ψ.λ
    _Q = Ψ.Q
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    # transition
    _Φ = Φ(Ψ, z)
    return [Fₓ(_x, _λ, _Φ); ff(_x, _r, _Φ, _Q, _λ, _τ)]
end

function J(Ψ::BidiagSys, z::MarkovState; ff=Criminos.linear_mixin)
    _M = Ψ.M
    _Γ = Ψ.Γ
    _λ = Ψ.λ
    _Q = Ψ.Q
    _τ = z.τ

    # transition
    _Fp(_z) = (
        _Φ = Φ(Ψ, z);
        [Fₓ(_z[1:z.n], _λ, _Φ); ff(_z[1:z.n], _z[z.n+1:2z.n], _Φ, _Q, _λ, _τ)]
    )
    return ForwardDiff.jacobian(_Fp, z.z)

end

# population
function Fₓ(_x, _λ, _Φ)
    _f1 = _Φ * _x + _λ
    return _f1
end


function forward(z₀, F; K=1000, ϵ=1e-8)
    z = copy(z₀)
    for k in 1:K

        z₁ = MarkovState(k, F(z))

        if (z₁.z - z.z) |> norm ≤ ϵ
            @printf("converged in %d iterations\n", k)
            break
        end

        z = copy(z₁)
    end
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
    return LinearAlgebra.norm(_r - _r₊, p)
end

function Lₓ(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return LinearAlgebra.norm(_x - _x₊, p)
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
    return norm(_x .* _r - _x₊ .* _r₊, p)
end

function KL(z, z₊; p=1)
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

function KLx(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    # compute KL-divergence of _r and _r₊
    _kl = sum(_x₊ .* log.(_x₊ ./ _x))
    return _kl
end

