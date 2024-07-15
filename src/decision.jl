################################################################################
# Decision rules and wrappers
################################################################################
# the decision is made as a centralized partτ
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
    cₜ, τₗ, τₕ, _... = args
    for (id, z) in enumerate(vector_ms)
        _I = (z.τ .- τₗ) ./ (τₕ .- τₗ)
        z.fpr = fp(_I, z)
    end
end


ν(x) = x > 0 ? exp(-1 / (x)) : 0;
σ(x; ℓ=0, ϵ=0.1) = begin
    ν((x - ℓ + ϵ / 2) / ϵ) / (ν((x - ℓ + ϵ / 2) / ϵ) + ν((ℓ + ϵ / 2 - x) / ϵ))
end


"""
    decision_matching_lh!(
        vector_ms::Vector{MarkovState{R,Tx}},
        vec_Ψ::Vector{BidiagSys{Tx,Tm}};
        args=nothing,
        ℓ=0.15,
        kwargs...
    ) where {R,Tx,Tm}

The `decision_matching_lh!` function performs decision matching for a given set of Markov states and a vector of BidiagSys objects. 
    It updates the threshold values (`z.θ`) and the risk values (`z.τ`) for each Markov state 
    based on the logistic regression model.

## Arguments
- `vector_ms`: A vector of MarkovState objects representing the Markov states.
- `vec_Ψ`: A vector of BidiagSys objects representing the BidiagSys values.
- `args`: Additional arguments for the function (optional).
- `ℓ`: The threshold risk value (default: 0.15).
- `kwargs`: Additional keyword arguments for the function.

## Type Parameters
- `R`: The type of the Markov state.
- `Tx`: The type of the BidiagSys object.
- `Tm`: The type of the BidiagSys value.

"""
function decision_matching_lh!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    ℓ=0.15,
    kwargs...
) where {R,Tx,Tm}
    cₜ, τₗ, τₕ, _... = args
    # at this model, we use logistic regression to 
    # match X^{-1}y to N, the number of records
    for (id, z) in enumerate(vector_ms)
        _y = z.y
        _N = [0:z.n-1...]
        h(a) = 1 ./ (1 .+ exp.(-_N * a[1] .+ a[2]))
        f(a) = -1 / (z.x₋ |> sum) * sum(
            _y .* log.(h(a))
            +
            (z.x₋ - _y) .* log.(1 .- h(a))
        )
        opt = Optim.optimize(f, zeros(2), optalg)
        _hy = 1 ./ (1 .+ exp.(-_N * opt.minimizer[1] .+ opt.minimizer[2]))

        _hy = _y ./ z.x₋
        _hl = sort(_hy; rev=true)

        # compute the threshold risk
        _I = nothing
        for l in _hl
            # _I = (_hy .> l)
            z.θ = l
            _I = σ.(_hy; ℓ=l, ϵ=3.0)
            _fpr = fp(_I, z)
            z.fpr = _fpr
            if _fpr > ℓ || l < 1e-2
                break
            end
        end

        # indices = _hy .>= θ
        # z.τ[.~indices] .= τₗ
        # z.τ[indices] .= τₕ
        z.τ = τₗ * (1 .- _I) .+ τₕ * _I
    end
end

"""
    decision_matching_lh_opt!(vector_ms, vec_Ψ; args=nothing, θ=0.02, δ=0.1, kwargs...)

This function performs decision matching optimization. 
@note:
 - this regard the decision-making process as another player in the GNEP
 - compared to, e.g., `decision_matching_lh!`, this function is more computationally expensive

# Arguments
- `vector_ms::Vector{MarkovState{R,Tx}}`: A vector of MarkovState objects.
- `vec_Ψ::Vector{BidiagSys{Tx,Tm}}`: A vector of BidiagSys objects.
- `args=nothing`: Additional arguments.
- `θ=0.02`: A parameter.
- `δ=0.1`: A parameter.
- `kwargs...`: Additional keyword arguments.

# Type Parameters
- `R`: Type parameter.
- `Tx`: Type parameter.
- `Tm`: Type parameter.

"""
function decision_matching_lh_opt!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    θ=0.02,
    δ=0.1,
    baropt=default_barrier_option,
    kwargs...
) where {R,Tx,Tm}
    cₜ, τₗ, τₕ, _... = args
    _n = vector_ms[1].n
    _N = _n * length(vector_ms)
    model = default_decision_option.model
    if default_decision_option.is_setup == false
        @variable(model, τ[1:_N] .>= 0)
        @variable(model, I[1:_N] .>= 0)
        @variable(model, h[1:_N] .>= 0)
        @variable(model, ℓ[1:length(vector_ms)] .>= 0)
        set_upper_bound.(I, 1.0)
        default_decision_option.is_setup = true
        default_decision_option.τ = τ
        default_decision_option.I = I
        default_decision_option.h = h
        default_decision_option.τcon = []
        @constraint(model, τ .- τₗ .* (1 .- I) - I .* τₕ .== 0)
        for (id, z) in enumerate(vector_ms)
            _I = I[(id-1)*_n+1:id*_n]
            _h = h[(id-1)*_n+1:id*_n]
            @constraint(model, _I .- (_h .- ℓ[id]) .>= 0)
            @constraint(model, (1 .- _I) .- (-_h .+ ℓ[id]) .>= 0)
        end
    end
    _φ = 0.0
    τ = default_decision_option.τ
    I = default_decision_option.I
    h = default_decision_option.h
    for c in default_decision_option.τcon
        delete(model, c)
    end
    default_decision_option.τcon = []
    for (id, z) in enumerate(vector_ms)
        _x = z.x₋
        _y = z.y
        _τ = τ[(id-1)*_n+1:id*_n]
        _I = I[(id-1)*_n+1:id*_n]
        _h = h[(id-1)*_n+1:id*_n]
        set_upper_bound.(_h, _y ./ _x .+ θ)
        set_lower_bound.(_h, _y ./ _x .- θ)
        push!(default_decision_option.τcon, @constraint(model, δ * sum(_x - _y) - _I' * (_x - _y) .>= 0))
        _φ += -_τ'cₜ + dist(_τ, z.τ) / baropt.μ
    end
    @objective(model, Min, _φ)
    optimize!(model)
    if termination_status(model) ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT)
        @warn "Optimizer did not converge"
        @info "" termination_status(model)
    end
    for (id, z) in enumerate(vector_ms)
        _τ = τ[(id-1)*_n+1:id*_n]
        _I = I[(id-1)*_n+1:id*_n]
        z.τ = value.(_τ)
        z.I = value.(_I)
        z.fpr = fp(z.I, z)
        z.θ = value(model[:ℓ][id])
    end

end
