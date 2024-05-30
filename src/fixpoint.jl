using ForwardDiff


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
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    fₜ=Criminos.mixed_in_identity, targs=nothing, # function of decision
    fₘ=Criminos.mixed_in_identity, margs=nothing, # function of mixed-in effect
    kwargs...
) where {R,Tx,Tm}
    # τ step, collectively
    fₜ(vector_ms, vec_Ψ; args=targs, kwargs...)
    # y step, collectively
    fₘ(vector_ms, vec_Ψ; args=margs, kwargs...)
    # x step, individually
    for (idx, z) in enumerate(vector_ms)
        Fₓ(z, vec_Ψ[idx])
    end
end

"""
population dynamics `x`
"""
function Fₓ(z, Ψ)
    z.x .= Ψ.Γ * z.x + Ψ.λ - Ψ.Γₕ * z.y
    z.ρ .= z.y ./ z.x
    z.ρ[z.ρ.==Inf] .= 0
    # update z, to be delected
    z.z = [z.x; z.ρ]
end

# "transition matrix"
function Φ(Ψ::BidiagSys, z::MarkovState)
    _r = z.z[z.n+1:2z.n]
    # transition
    _Φ = _Γ - Ψ.Γₕ * Diagonal(_r)
    return _Φ
end