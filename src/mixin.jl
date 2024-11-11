################################################################################
# Vanilla Mixed-in Functions 
################################################################################

function mixed_in_identity(
    z, Ψ;
    args=nothing,
    baropt=default_barrier_option,
    kwargs...
)
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _x₊ = _Φ * _x + Ψ.λ
    _y₊ = _Φ * (_x .* _r) + Ψ.Q * Ψ.λ
    _f2 = _y₊ ./ _x₊
    return _f2
end

@doc raw"""
    A 0-1 mixed-in to verify convergence of $X$ without $\rho$
    this only works with $\gamma < 1$
"""
function mixed_in_binary_zigzag(
    z, Ψ;
    α=0.1, args=nothing, kwargs...
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

include("./mixin_gnep.jl")