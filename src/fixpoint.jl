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

Compute the fixed-point iteration given the BidiagSys Ψ and MarkovState z.

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


function forward(z₀, F; K=10000, ϵ=EPS_FP)
    z = copy(z₀)
    fp = 1e6
    signal = ""
    bool_opt = false
    for k in 1:K
        # move to next iterate
        z₁ = MarkovState(k, F(z), z.τ)
        # assign mixed-in function value
        z₁.f = z.f
        # copy previous recidivists
        z₁.y₋ = copy(z.y)

        fp = (z₁.z - z.z) |> norm
        if fp ≤ ϵ * (z.z |> norm)
            @printf("converged in %d iterations\n", k)
            signal = "⋆"
            bool_opt = true
            break
        end
        z = copy(z₁)
    end
    @printf("%s final residual %.1e\n", signal, fp)
    return z, bool_opt
end


