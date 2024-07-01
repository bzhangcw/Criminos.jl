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
    for (id, z) in enumerate(vector_ms)
        z.fpr = 0.0
    end
end


ν(x) = x > 0 ? exp(-1 / (x)) : 0;
σ(x; ℓ=0, ϵ=0.1) = begin
    ν((x - ℓ + ϵ / 2) / ϵ) / (ν((x - ℓ + ϵ / 2) / ϵ) + ν((ℓ + ϵ / 2 - x) / ϵ))
end
function decision_matching_lh!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    ℓ=0.15,
    kwargs...
) where {R,Tx,Tm}
    cₜ, α₁, α₂, _... = args
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
        # z.τ[.~indices] .= α₁
        # z.τ[indices] .= α₂
        z.τ = α₁ * (1 .- _I) .+ α₂ * _I
    end
end


# matching rule
function decision_matching!(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    _c, _... = args
    _n = vector_ms[1].n
    _N = _n * length(vector_ms)
    model = default_decision_option.model
    if default_decision_option.is_setup == false
        @variable(model, τ[1:_N] .>= 0)
        default_decision_option.τ = τ
        default_decision_option.is_setup = true
    end
    τ = default_decision_option.τ
    _φ = 0.0

    for (id, z) in enumerate(vector_ms)
        _x = z.x
        _y = z.y
        _τ = τ[(id-1)*_n+1:id*_n]

        _p = _y ./ (_x .+ 1e-3)
        # if use smooth box constraint
        set_upper_bound.(_τ, _p .+ 0.02)
        set_lower_bound.(_τ, max.(0, _p .- 0.02))
        _φ += _τ' * _c # + sum(_τ .^ 2) * 10
    end
    @objective(model, Min, _φ)
    optimize!(model)
    if termination_status(model) ∉ (MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.FEASIBLE_POINT)
        @warn "Optimizer did not converge"
        @info "" termination_status(model)
    end
    τv = value.(τ)
    for (id, z) in enumerate(vector_ms)
        z.τ = τv[(id-1)*_n+1:id*_n]
    end
end
