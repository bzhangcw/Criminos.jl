using ForwardDiff



"""
    F!(
        vector_ms::Vector{MarkovState{R,Tx}},
        vec_Ψ::Vector{BidiagSys{Tx,Tm}};
        fₜ=Criminos.decision_identity, targs=nothing, # function of decision
        fₘ=Criminos.mixed_in_identity, margs=nothing, # function of mixed-in effect
        kwargs...
    ) where {R,Tx,Tm}

The `F!` function performs a fixed-point iteration on a vector of `MarkovState` objects and a vector of `BidiagSys` objects. It updates the states in `vector_ms` and `vec_Ψ` according to the specified functions `fₜ`, `fₘ`, and `Fₓ`.

## Arguments
- `vector_ms::Vector{MarkovState{R,Tx}}`: A vector of `MarkovState` objects.
- `vec_Ψ::Vector{BidiagSys{Tx,Tm}}`: A vector of `BidiagSys` objects.
- `fₜ`: A function that updates the states in `vector_ms` collectively. Default is `Criminos.decision_identity`.
- `targs`: Additional arguments for the function `fₜ`. Default is `nothing`.
- `fₘ`: A function that updates the states in `vec_Ψ` collectively. Default is `Criminos.mixed_in_identity`.
- `margs`: Additional arguments for the function `fₘ`. Default is `nothing`.
- `kwargs...`: Additional keyword arguments.

"""
function F!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    fₜ=Criminos.decision_identity, targs=nothing, # function of decision
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
    z.x₋ = copy(z.x)
    z.ρ .= z.y₋ ./ z.x₋
    z.x .= Ψ.Γ * z.x + Ψ.λ - Ψ.Γₕ * z.y
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