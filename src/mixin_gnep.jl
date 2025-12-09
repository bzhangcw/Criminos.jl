################################################################################
# Mixed-in Functions using GNEP and GPG Dynamics
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
    # unpacking args,
    mipar, ω∇ω, G, _... = args
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
    ycons = []
    for (id, z) in enumerate(vector_ms)
        z.y₋ = copy(z.y)
        _x = z.x
        _x₋ = z.x₋
        _y = y[(id-1)*_n+1:id*_n]
        _Ψ = vector_Ψ[id]
        # --------------------------------------------------
        # set upper bound of the boxes
        # --------------------------------------------------
        # set_upper_bound.(_y, _x .* ι[id])
        set_upper_bound.(_y, _x)
        _φ += (_x + _Ψ.λ)' * _Ψ.Γₕ * _y + dist(_y, z.y) / baropt.μ
    end
    # repeat the blocks
    _τ = vcat([z.τ .* z.β for z in vector_ms]...)
    _x = vcat([z.x for z in vector_ms]...)
    ##################################################
    _w, ∇ω = ω∇ω(y, _τ, _x)
    _f_expr = _φ + _w
    @objective(model, Min, _f_expr)
    ##################################################
    optimize!(model)

    if termination_status(model) ∉ (
        MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT, MOI.SLOW_PROGRESS
    )
        @warn "Optimizer did not converge"
        @info "" latex_formulation(model)
        @info "" termination_status(model)
        write_to_file(model, "model.lp")
        for c in ycons
            delete(model, c)
        end
        return 0
    end

    yv = value.(y)
    for (id, z) in enumerate(vector_ms)
        z.y = yv[(id-1)*_n+1:id*_n]
    end
    for c in ycons
        delete(model, c)
    end
    return 1
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
