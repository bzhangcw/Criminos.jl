################################################################################
# Decision Rules and Wrappers
################################################################################
# the decision is made as a centralized player τ
using Optim, LineSearches
optalg = LBFGS(; m=10,
    alphaguess=LineSearches.InitialStatic(),
    linesearch=LineSearches.HagerZhang(),
)

# false positive rate, 
#  I is the index of predicted positive samples
fp(I, z) = I' * (z.x₋ - z.y) / sum(z.x₋ - z.y)

function decision_null!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    # do nothing
    for (id, z) in enumerate(vector_ms)
        z.τ .= 0.0
    end
end

function decision_identity!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    # do nothing
    cₜ, τₗ, τₕ, _unused_args... = args
    for (id, z) in enumerate(vector_ms)
        _I = (z.τ .- τₗ) ./ (τₕ .- τₗ)
        z.fpr = fp(_I, z)
    end
end


ν(x) = x > 0 ? exp(-1 / (x)) : 0;
σ(x; ℓ=0, ϵ=0.1) = begin
    ν((x - ℓ + ϵ / 2) / ϵ) / (ν((x - ℓ + ϵ / 2) / ϵ) + ν((ℓ + ϵ / 2 - x) / ϵ))
end

@doc raw"""
decision_priority_by_opt!(vector_ms, vec_Ψ; args=nothing, Δ=0.02, δ=0.1, kwargs...)

This function performs decision matching optimization. 
@note:
 - this regard the decision-making process as another player in the GNEP
 - compared to, e.g., `decision_matching_lh_opt_with_threshold!`, 
   this function is a simplified version that does not include the I variable.

# Arguments
- `vector_ms::Vector{MarkovState{R,Tx}}`: A vector of MarkovState objects.
- `vec_Ψ::Vector{BidiagSys{Tx,Tm}}`: A vector of BidiagSys objects.
- `args=nothing`: Additional arguments.
- `Δ=0.02`: A parameter.
- `δ=0.1`: A parameter.
- `kwargs...`: Additional keyword arguments.

# Type Parameters
- `R`: Type parameter.
- `Tx`: Type parameter.
- `Tm`: Type parameter.
"""
function decision_priority_by_opt!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    mipar, decision_option, cₜ, τₗ, τₕ, Δ, δ, _unused_args... = args
    _n = vector_ms[1].n
    _N = _n * length(vector_ms)
    model = decision_option.model
    if decision_option.is_setup == false
        @variable(model, τ[1:_N] .>= 0)
        @variable(model, h[1:_N] .>= 0)
        @variable(model, θ[1:_N]) # redundant if not conic
        decision_option.is_setup = true
        decision_option.τ = τ
        decision_option.h = h
        decision_option.τcon = []
        @constraint(model, τ .- τₗ .* (1 .- h) - h .* τₕ .== 0)
        if decision_option.is_conic
            expt_inc_cone!.(τ, θ, model)
        end
    end
    _φ = 0.0
    τ = decision_option.τ
    h = decision_option.h
    θ = model[:θ]
    for c in decision_option.τcon
        delete(model, c)
    end
    decision_option.τcon = []
    for (id, z) in enumerate(vector_ms)
        _x = z.x₋
        _y = z.y
        _τ = τ[(id-1)*_n+1:id*_n]
        _h = h[(id-1)*_n+1:id*_n]

        set_upper_bound.(_h, min.(_y ./ _x .+ Δ, 1.0))
        set_lower_bound.(_h, max.(_y ./ _x .- Δ, 0.0))
        push!(
            decision_option.τcon,
            # Equal Opportunity
            @constraint(model, δ * sum(_x - _y) - _h' * (_x - _y) .>= 0)
            # Equal Probability
            # @constraint(model, δ * sum(_x) - _τ' * _y .>= 0)
        )
        # the revenue/cost is unit to the population size
        # that is, accumulated by each individual
        if decision_option.is_conic
            _θ = θ[(id-1)*_n+1:id*_n]
            _φ += sum(gety(mipar, zeros(_θ |> length), _θ))
        else
            _φ += -_τ' * (cₜ .* _x)
        end
    end
    @objective(model, Min, _φ)
    optimize!(model)
    if termination_status(model) ∉ (
        MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT, MOI.SLOW_PROGRESS
    )
        @warn "Optimizer did not converge"
        @info "" termination_status(model)
        suff = decision_option.is_conic ? "task" : "lp"
        write_to_file(model, "decision_priority_by_opt.$suff")
    end
    for (id, z) in enumerate(vector_ms)
        _τ = τ[(id-1)*_n+1:id*_n]
        _h = h[(id-1)*_n+1:id*_n]
        z.τ = value.(_τ)
        z.I = value.(_h)
        z.fpr = fp(z.I, z)
    end
end

@doc raw"""
Alternative to `decision_priority_by_opt!` that uses simple priority rule, 
    it is equivalent
"""
function decision_priority!(
    z, Ψ;
    args=nothing,
    baropt=default_barrier_option,
    kwargs...
)
    cₜ, τₗ, τₕ, Δ, δ, _... = args
    _x = z.x₋
    _y = z.y
    c = cₜ .* _x
    a = _x - _y
    b = sum(a .* (-_y ./ (_x .+ 1e-4) .+ (δ + Δ)))

    q, total_a, total_c = knapsack_with_ub(c, a, b, 2 * Δ)

    h = _y ./ _x .- Δ + q

    return h
end


function knapsack_with_ub(
    c::Vector{Float64},
    a::Vector{Float64},
    b::Float64,
    d::Float64
)
    n = length(c)
    @assert length(a) == n "Arrays 'c' and 'a' must be of the same length."

    # Compute the ratios c_i / a_i
    ratios = c ./ a

    # Handle cases where a_i is zero to avoid division by zero
    for i in eachindex(a)
        if a[i] == 0.0
            ratios[i] = Inf  # Assign infinite ratio if a_i is zero
        end
    end

    # Get the indices that would sort the ratios in descending order
    sorted_indices = sortperm(ratios, rev=true)

    x = zeros(Float64, n)  # Quantities of each item selected
    total_a = 0.0
    total_c = 0.0

    for idx in sorted_indices
        if total_a >= b
            break
        end
        remaining_capacity = b - total_a
        max_possible_xi = min(d, remaining_capacity / a[idx])
        if max_possible_xi <= 0.0
            continue  # Cannot take any more of this item
        end
        x_i = max_possible_xi
        x[idx] = x_i
        total_a += a[idx] * x_i
        total_c += c[idx] * x_i
    end

    selected_indices = findall(x .> 0)
    return x, total_a, total_c
end

include("decision.depre.jl")