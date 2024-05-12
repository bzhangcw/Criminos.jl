################################################################################
# Mixed-in Functions using GNEP and GPG dynamics
################################################################################

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
# UTILITY FUNCTIONS
################################################################################
# u₁(x) without externality
L(x, Ψ) = 1 / 2 * x' * (I - Ψ.Γ) * x - x' * Ψ.λ
# u₂(x) without externality
# linear quadratic function
function w_linearquad(
    y, _τ, Z;
    Ψ=nothing, baropt=default_barrier_option,
    kwargs...
)
    ∇₀, H₀, ∇ₜ, Hₜ, _... = Z
    _c = Ψ.Q * Ψ.λ

    Cₜ = diagm(_τ) * Hₜ * diagm(_τ) + H₀
    _w = (
        y' * Cₜ * y / 2 +    # second-order
        y' * ((∇ₜ*_τ)[:] - (∇₀*_c)[:])               # first-order
    )
    ∇ = Cₜ * y + ((∇ₜ*_τ)[:] - (∇₀*_c)[:])
    return _w, ∇, 0.0
end
w = w_linearquad


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
    _x = z.z[1:z.n]
    _r = z.z[z.n+1:2z.n]
    _τ = z.τ
    _y = _x .* _r
    d = _x |> length # problem dimension
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
    _w, ∇, ∇² = w(y, _τ, Z; Ψ=Ψ, baropt=baropt)
    # add cross term: the externality term
    _φ = y' * Ψ.Γ' * Ψ.M' * _x₊
    # add proximal term
    _f_expr = _φ + dist(y, _y) / baropt.μ + _w
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
    throw(ErrorException("not implemented correctly"))
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
    _τ = z.τ
    _x = z.x
    _r = z.ρ
    _y = _x .* _r

    _L = L(_x, Ψ)
    _w, ∇, ∇² = w(_y, _τ, Z; Ψ=Ψ, baropt=baropt)
    _φ = _y' * Ψ.Γ' * Ψ.M' * _x
    return _L + _φ + dist(_y, z.y₋) / baropt.μ + _w
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

    _w, ∇, ∇² = w(_y, _τ, Z; Ψ=Ψ, baropt=baropt)
    dx = (I - Ψ.Γ) * _x + Ψ.M * Ψ.Γ * _y - Ψ.λ
    dy = ∇ + Ψ.Γ' * Ψ.M' * _x
    v₊ = max.(0, -dy)
    v₋ = max.(0, dy)
    complementary = norm((_x - _y) .* v₊) + norm((_y) .* v₋)

    @info "KKT\n" dx |> norm dy |> norm complementary

    return dx, dy, v₊, v₋, Ψ.Γ' * Ψ.M' * _x
end
