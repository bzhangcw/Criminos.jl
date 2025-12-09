################################################################################
# Decision Rules and Wrappers based on τ directly
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
expcone!(τ, r, model) = @constraint(
    model, [τ + 1, 1, r] in MOI.ExponentialCone()
)
function decision_priority_by_τ!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    decision_option, cₜ, C, lb, ub, _unused_args... = args
    _n = vector_ms[1].n
    _N = _n * length(vector_ms)
    model = decision_option.model
    if decision_option.is_setup == false
        @variable(model, τ[1:_N] .>= 0)
        @variable(model, r[1:_N])
        decision_option.is_setup = true
        decision_option.τ = τ
        decision_option.τcon = []

        if decision_option.is_conic
            # -log.(τ + 1) <= r => [τ + 1, 1, -r] in EC
            for j in 1:_N
                @constraint(
                    model, [τ[j] + 1, 1, r[j]] in MOI.ExponentialCone()
                )
            end
        end
    end
    _φ = 0.0
    τ = decision_option.τ
    h = decision_option.h
    r = model[:r]
    for c in decision_option.τcon
        delete(model, c)
    end
    decision_option.τcon = []
    set_lower_bound.(τ, lb)
    set_upper_bound.(τ, ub)
    for (id, z) in enumerate(vector_ms)
        _x = z.x₋
        _y = z.y
        _τ = τ[(id-1)*_n+1:id*_n]
        _r = r[(id-1)*_n+1:id*_n]

        _treat = _τ .* _x
        push!(
            decision_option.τcon,
            # Equal Opportunity
            @constraint(model, sum(_treat) <= C)
            # Equal Probability
            # @constraint(model, δ * sum(_x) - _τ' * _y .>= 0)
        )
        # that is, accumulated by each individual
        if decision_option.is_conic
            _φ += cₜ' * _r
        else
            _φ += cₜ' * _treat
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
        z.τ = value.(_τ)
    end
end


include("decisionh.jl")
include("decision.depre.jl")