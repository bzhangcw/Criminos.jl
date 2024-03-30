

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
function mixin(Ψ::BidiagSys, z::MarkovState;
    α=0.1, Z=ones(z.n, z.n) * 0.1, ff=linear_mixin
)
    _x = z.x
    _r = z.ρ
    _M = Ψ.M
    _Γ = Ψ.Γ
    _λ = Ψ.λ
    _Q = Ψ.Q

    _τ = z.τ
    # transition
    _Φ = _Γ - _M * _Γ * Diagonal(_r)
    # half update
    return ff(_x, _r, _Φ, _Q, _λ, _τ; α=α, Z=Z)
end

function linear_mixin(_x, _r, _Φ, _Q, _λ, _τ;
    α=0.1, Z=ones(_x |> length, _x |> length) * 0.1
)
    # half update
    n = _x |> length
    _x₊ = _Φ * _x + _λ
    _y = _Φ * (_x .* _r) + _Q * _λ
    _A = α * Z * (_τ .* (_x .* _r))
    # logit transformation
    _σ = log.(_y ./ (_x₊ .- _y)) + _A
    # exbit (logistic) transformation
    _f2 = 1 ./ (1 .+ exp.(-(_σ)))
    return _f2
end

function linear_mixin_uniform(_x, _r, _Φ, _Q, _λ, _τ;
    α=0.1, Z=ones(_x |> length, _x |> length) * 0.1
)
    # half update
    n = _x |> length
    _x₊ = _Φ * _x + _λ
    _y = _Φ * (_x .* _r) + _Q * _λ
    _A = α * (ones(_x |> length)' * (_τ .* _x .* _r)) * (20 / n) .* ones(_x |> length)
    # logit transformation
    _σ = log.(_y ./ (_x₊ .- _y)) + _A
    # exbit (logistic) transformation
    _f2 = 1 ./ (1 .+ exp.(-(_σ)))
    return _f2
end

function sublinear_mixin(_x, _r, _Φ, _Q, _λ, _τ;
    α=0.1, Z=ones(_x |> length, _x |> length) * 0.1
)
    # half update
    _x₊ = _Φ * _x + _λ
    _y = _Φ * (_x .* _r) + _Q * _λ
    _A = α * Z * (_τ .* (_x .* _r))
    # logit transformation
    _σ = log.(_y ./ (_x₊ .- _y)) + _A
    # exbit (logistic) transformation
    _f2 = 1 ./ (1 .+ exp.(-_σ))
    return _f2
end