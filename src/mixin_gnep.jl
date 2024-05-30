################################################################################
# Mixed-in Functions using GNEP and GPG dynamics
################################################################################

using JuMP, Gurobi, Ipopt
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
# u(x) without externality
L(x, Ψ) = 1 / 2 * x' * (I - Ψ.Γ) * x - x' * Ψ.λ

# w(x) without externality
# linear quadratic function
"""
    linear quadratic function
"""
function w_linearquad(
    y, _τ, _c, args;
    baropt=default_barrier_option,
    kwargs...
)
    ∇₀, H₀, ∇ₜ, Hₜ, _... = args

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
function mixed_in_gnep_best!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vector_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    baropt=default_barrier_option,
    kwargs...
) where {R,Tx,Tm}

    _n = vector_ms[1].n
    _N = _n * length(vector_ms)
    model = default_gnep_mixin_option.model
    if default_gnep_mixin_option.is_setup == false
        @variable(model, y[1:_N] .>= 0)
        default_gnep_mixin_option.y = y
        default_gnep_mixin_option.is_setup = true
    end
    y = default_gnep_mixin_option.y
    ##################################################
    # constraints for each y
    # sum of individual externality and proximity
    _φ = 0.0
    for (id, z) in enumerate(vector_ms)
        _x = z.x
        _y = y[(id-1)*_n+1:id*_n]
        set_upper_bound.(_y, _x)
        if !isnothing(default_gnep_mixin_option.ycon)
            delete(model, default_gnep_mixin_option.ycon)
            unregister(model, :ycon)
        end
        if default_gnep_mixin_option.is_kl
            # if use smooth box constraint
            default_gnep_mixin_option.ycon = @constraint(
                model, ycon, _y' * log.(_y .+ 1e-3) + (_x - _y)' * log.(_x - _y) <= 1.1 + _x' * log.(_x / 2)
            )
        end
        _φ += _y' * vector_Ψ[id].Γ' * vector_Ψ[id].M' * _x + dist(_y, z.y) / baropt.μ
    end
    # repeat the blocks
    _c = vcat([Ψ.Q * Ψ.λ for Ψ in vector_Ψ]...)
    _τ = vcat([z.τ for z in vector_ms]...)
    ##################################################
    _w, ∇, ∇² = w(y, _τ, _c, args; baropt=baropt)
    # add proximal term and the externality term
    _f_expr = _φ + _w
    @objective(model, Min, _f_expr)
    ##################################################
    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        @warn "Gurobi did not converge"
    end
    yv = value.(y)
    for (id, z) in enumerate(vector_ms)
        z.y = yv[(id-1)*_n+1:id*_n]
    end
end

@doc raw"""
    "better" response for the GNEP problem 
    ```math
    y \in [0, x_{k+1}]
    ```
"""
function mixed_in_gnep_grad!(
    z, Ψ, ;
    α=0.1,
    args=nothing,
    baropt=default_barrier_option,
    kwargs...
)
    throw(ErrorException("not implemented correctly"))
    # _x = z.z[1:z.n]
    # 
    # _τ = z.τ
    # _y = _x .* _r
    # d = _x |> length
    # _x =  * _x + Ψ.λ
    # z.f = 0.0 # unused
    # # we let y proceed a gradient step
    # _ψ, _ = ψ(_x, _r, , _τ, args; Ψ=Ψ, baropt=baropt)
    # # add cross term
    # _φ = _ψ + Ψ.Γ' * Ψ.M' * _x
    # u(y) = _φ' * y + dist(y, _y) / baropt.μ + y' * diagm(_τ) * C * diagm(_τ) * y / 2
    # # do a line search
    # α = 1 / baropt.μ
    # _y₊ = similar(_y)
    # _u = u(_y)
    # while true
    #     _y₊ = _y - α * (_φ + diagm(_τ) * C * diagm(_τ) * _y)
    #     _y₊ = min.(max.(_y₊, 1e-3), _x .- 1e-3)
    #     _u₊ = u(_y₊)
    #     # @info "line search" α _u _u₊
    #     if (_u₊ < _u) || (α < 1e-8)
    #         break
    #     end
    #     α /= 1.2
    # end
    # _p = _y₊ ./ _x
    # return _p
end

function pot_gnep(
    z::MarkovState{R,Tx}, z₊::MarkovState{R,Tx};
    baropt=default_barrier_option,
    args=nothing,
    Ψ=nothing
) where {R,Tx}
    _τ = z.τ
    _x = z.x
    _y = z.y
    _c = Ψ.Q * Ψ.λ
    _L = L(_x, Ψ)
    _w, ∇, ∇² = w(_y, _τ, _c, args; Ψ=Ψ, baropt=baropt)
    _φ = _y' * Ψ.Γ' * Ψ.M' * _x
    return _L + _φ + dist(_y, z.y₋) / baropt.μ + _w
end


# verify KKT
function kkt_box_opt(
    z::MarkovState{R,Tx};
    baropt=default_barrier_option,
    args=nothing,
    Ψ=nothing
) where {R,Tx}

    _τ = z.τ
    _x = z.x
    _r = z.ρ
    _y = z.y
    _c = Ψ.Q * Ψ.λ
    _w, ∇, ∇² = w(_y, _τ, _c, args; Ψ=Ψ, baropt=baropt)
    dx = (I - Ψ.Γ) * _x + Ψ.M * Ψ.Γ * _y - Ψ.λ
    dy = ∇ + Ψ.Γ' * Ψ.M' * _x
    v₊ = max.(0, -dy)
    v₋ = max.(0, dy)
    complementary = norm((_x - _y) .* v₊) + norm((_y) .* v₋)

    @info "KKT\n" dx |> norm dy |> norm complementary

    return dx, dy, v₊, v₋, Ψ.Γ' * Ψ.M' * _x
end
