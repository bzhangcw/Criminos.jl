"""
Mixed-in Functions using GNEP and GPG
"""

using JuMP, Gurobi
using ForwardDiff


################################################################################
# DIVERGENCE FUNCTIONS
################################################################################
_dist_quad(x, y) = 1 / 2 * sum((x - y) .^ 2)
_dist_kl(x, y) = sum(x .* log.(x ./ y))

dist = _dist_quad
∇P(x, y) = ForwardDiff.gradient((x) => dist(x, y), x)
################################################################################


# pure linear function
ψ_linearquad(_x, _r, _Φ, _τ, Z; Ψ=nothing, baropt=default_barrier_option, kwargs...) = begin
    _y = _x .* _r
    _c = Ψ.Q * Ψ.λ
    A, B, _... = Z

    ∇ = -(A*_τ)[:] - (B*_c)[:]
    return ∇, 0.0
end
ψ = ψ_linearquad


@doc raw"""
    best response for the GNEP problem 
    ```math
    y \in [0, x_{k+1}]
    ```
"""
function mixed_in_gnep_best(
    z, Ψ, _Φ;
    α=0.1,
    Z=nothing,
    baropt=default_barrier_option,
    kwargs...
)
    A, B, C, _... = Z
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    _y = _x .* _r
    d = _x |> length
    _x₊ = _Φ * _x + Ψ.λ
    model = default_gnep_mixin_option.model
    if default_gnep_mixin_option.is_setup == false
        @variable(model, y[1:d] .>= 0)
        default_gnep_mixin_option.y = y
        default_gnep_mixin_option.is_setup = true
    end
    y = default_gnep_mixin_option.y
    ##################################################
    # constraints for y
    set_upper_bound.(y, _x₊)
    ##################################################
    _ψ, _ = ψ(_x₊, _r, _Φ, _τ, Z; Ψ=Ψ, baropt=baropt)
    # add cross term
    _φ = _ψ + Ψ.Γ' * Ψ.M' * _x₊
    # add proximal term
    _f_expr = _φ' * y + dist(y, _y) / baropt.μ + y' * diagm(_τ) * C * diagm(_τ) * y / 2
    @objective(model, Min, _f_expr)
    ##################################################
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "Gurobi did not converge"
        @warn "" _r _x _x₊ _y
    end
    z.f = objective_value(model)

    _y₊ = value.(y)
    _p = _y₊ ./ _x₊
    if (_p .- 1 |> maximum) > 0.1
        @warn "best response is not feasible"
        @warn "" _r _x _x₊ _y
    end
    return _p
end

@doc raw"""
    "better" response for the GNEP problem 
    ```math
    y \in [0, x_{k+1}]
    ```
"""
function mixed_in_gnep_grad(
    z, Ψ, _Φ;
    α=0.1,
    Z=nothing,
    baropt=default_barrier_option,
    kwargs...
)
    A, B, C, _... = Z
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    _y = _x .* _r
    d = _x |> length
    _x₊ = _Φ * _x + Ψ.λ

    z.f = 0.0 # unused
    # we let y proceed a gradient step
    _ψ, _ = ψ(_x₊, _r, _Φ, _τ, Z; Ψ=Ψ, baropt=baropt)
    # add cross term
    _φ = _ψ + Ψ.Γ' * Ψ.M' * _x₊
    u(y) = _φ' * y + dist(y, _y) / baropt.μ + y' * diagm(_τ) * C * diagm(_τ) * y / 2
    # do a line search
    α = 1 / baropt.μ
    _y₊ = similar(_y)
    _u = u(_y)
    while true
        _y₊ = _y - α * (_φ + diagm(_τ) * C * diagm(_τ) * _y)
        _y₊ = min.(max.(_y₊, 1e-3), _x₊ .- 1e-3)
        _u₊ = u(_y₊)
        # @info "line search" α _u _u₊
        if (_u₊ < _u) || (α < 1e-8)
            break
        end
        α /= 1.2
    end
    _p = _y₊ ./ _x₊
    return _p
end

function pot_gnep(z, z₊;
    baropt=default_barrier_option,
    Z=nothing,
    Ψ=nothing
)
    A, B, C, _... = Z
    _τ = z.τ
    _x = z.x
    _r = z.ρ
    _y = _x .* _r

    _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(_r)
    L(x) = 1 / 2 * x' * (I - Ψ.Γ) * x - x' * Ψ.λ

    _ψ, _e = ψ(_x, _r, _Φ, _τ, Z; Ψ=Ψ, baropt=baropt)
    _φ = _ψ + Ψ.Γ' * Ψ.M' * _x
    # return L(_x) + _e
    return L(_x) + _φ' * _y + dist(_y, z.y₋) / baropt.μ + _y' * diagm(_τ) * C * diagm(_τ) * _y / 2
end


# verify KKT
function kkt_box_opt(z;
    baropt=default_barrier_option,
    Z=nothing,
    Ψ=nothing
)

    _τ = z.τ
    _x = z.x
    _r = z.ρ
    _y = _x .* _r


    _Φ = Ψ.Γ - Ψ.M * Ψ.Γ * Diagonal(_r)

    Z = Ψ.M * Ψ.Γ * Diagonal(_y)
    e = ones(_x |> length)

    _ψ, _e = ψ(_x, _r, _Φ, _τ, Z; Ψ=Ψ, baropt=baropt)
    dy = _ψ #+ Ψ.Γ' * Ψ.M' * _x
    dx = (I - Ψ.Γ) * _x + (Z * e - Ψ.λ)
    @info "KKT\n" dx |> norm [dy _y _x ((_x - _y) .* (dy .<= 0) + (_y .* (dy .>= 0)))]

    return dx, dy
end
