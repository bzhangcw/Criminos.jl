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
    F(Ψ::BidiagSys, z::MarkovState; fₘ=Criminos.linear_mixin)

Compute the fixed-point iteration given the BidiagSys Ψ and MarkovState z.

# Arguments
- `Ψ::BidiagSys`: The BidiagSys object.
- `z::MarkovState`: The MarkovState object.
- `fₘ=Criminos.linear_mixin`: The mixin function.

# Returns
- An array containing the values of z=F(z).

"""
function F(
    Ψ::BidiagSys, z::MarkovState;
    fₘ=Criminos.mixed_in_identity, margs=nothing, # function of mixed-in effect
    kwargs...
)

    fₘ(z, Ψ; args=margs, kwargs...)
    Fₓ(z, Ψ)


    z.z = [z.x; z.ρ]
end

# population
function Fₓ(z, Ψ)
    z.x .= Ψ.Γ * z.x + Ψ.λ - Ψ.M * Ψ.Γ * z.y
    z.ρ .= z.y ./ z.x
    z.ρ[z.ρ.==Inf] .= 0
end


