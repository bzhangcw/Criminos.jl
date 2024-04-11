

"""
    linear_mixin(Ψ::BidiagSys, z::MarkovState; α=0.1, Z=ones(n, n) * 0.1)

Apply linear mixin to the given BidiagSys Ψ and MarkovState z.

Arguments:
- `Ψ::BidiagSys`: The BidiagSys object.
- `z::MarkovState`: The MarkovState object.
- `α=0.1`: The mixing parameter (default is 0.1).
- `Z=ones(n, n) * 0.1`: The mixing matrix (default is a matrix of ones multiplied by 0.1).

Returns:
- The aftermath probability of `z`.

"""


function entropy_mixin(
    z, Ψ, _Φ;
    Z=nothing,
    baropt=default_barrier_option,
    kwargs...
)
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    d = _x |> length

    μ = baropt.μ
    # !!!do not modify in place
    # if isnothing(Z)
    #     A = LowerTriangular(rand(d, d)) * 0.9
    # else
    #     A = Z
    # end
    A = Z
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    _ψ = ψ(_x, _r, _Φ, _τ, A; Ψ=Ψ, baropt=baropt)
    _φ = _ψ - ℓ * Ψ.Γ' * Ψ.M' * _x₊
    # _f2 = 1 ./ (exp.(_φ ./ μ) .+ 1)
    _f2 = _y₊ ./ _x₊
    return _f2
end

@doc raw"""
    A 0-1 mixed-in to verify convergence of $X$ without $\rho$
    this only works with $\gamma < 1$
"""
function binary_zigzag_mixin(
    z, Ψ, _Φ;
    α=0.1, Z=nothing, kwargs...
)
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    if (_r |> minimum) > 2e-2
        return ones(_x |> length) * 1e-2
    else
        return ones(_x |> length) * 0.98
    end
end