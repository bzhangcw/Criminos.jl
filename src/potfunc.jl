############################################
# potential functions
############################################
function quad_linear_old(z, z₊;
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
    ε = 1e-3

    _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(_r)
    _x₊ = _Φ * _x + Ψ.λ
    L(x) = 1 / 2 * x' * (I - Ψ.Γ) * x + x' * (Ψ.M * Ψ.Γ * _y - Ψ.λ)

    _ψ = ψ(_x, _r, _Φ, _τ, A; Ψ=Ψ, baropt=baropt)
    _L = ℓ * L(_x)
    _e = _y' * _ψ / 2 + ℓ * _y' * Ψ.Γ' * Ψ.M' * _x
    _p = baropt.μ * ((_x .+ ε)' * log.(_x .+ ε) + (_x - _y .+ ε)' * log.(_x - _y .+ ε))
    @debug "" (L(_x₊) - L(_x)) (((_x₊ - _x) |> norm)^2) _e _L

    return _e + _L #+ _p
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
    ε = 1e-3

    _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(_r)
    _x₊ = _Φ * _x + Ψ.λ
    L(x) = 1 / 2 * x' * (I - Ψ.Γ) * x + x' * (Ψ.M * Ψ.Γ * _y - Ψ.λ)

    _L = L(_x)
    _, _e = ψ(_x, _r, _Φ, _τ, A; Ψ=Ψ, baropt=baropt)
    _p = baropt.μ * ((_x .+ ε)' * log.(_x .+ ε) + (_x - _y .+ ε)' * log.(_x - _y .+ ε))
    @debug "" (L(_x₊) - L(_x)) (((_x₊ - _x) |> norm)^2) _e _L

    return _L + ℓ * _e + _p
end


############################################
# ordinary potential functions
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
