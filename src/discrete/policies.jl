using JuMP, MadNLP
using ForwardDiff, Optim

__optim_ipnewton = IPNewton(;
    linesearch=Optim.backtrack_constrained_grad,
    μ0=:auto,
    show_linesearch=false
)

__get_options(tol=1e-4, verbose=true, store_trace=true) = Optim.Options(
    f_abstol=tol,
    g_abstol=tol,
    iterations=1_000,
    store_trace=store_trace,
    show_trace=verbose,
    show_every=5,
    time_limit=100
)

function __policy_opt_grad_fp(z, data, C, ϕ, p; μ=1e-8, kwargs...)
    n = data[:n]

    f(zv) = begin
        xv = reshape(zv[1:4n], n, 4)
        yv = reshape(zv[4n+1:8n], n, 4)
        τv = zv[8n+1:9n]
        x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4 = F(xv, yv, sum(yv) / (sum(xv) + 1e-5), data; τ=τv, p=p)
        return sum(y₊1) + sum(y₊2) + sum(y₊3) + sum(y₊4) - μ * log(C - sum(x₊3))
    end

    ∇f!(buffer, zv) = ForwardDiff.gradient!(buffer, f, zv)

    lb = zeros(9n)
    ub = ones(9n) .* 1e10
    z₀ = ones(9n) .* 1e-2

    results = Optim.optimize(f, ∇f!, lb, ub, z₀, Optim.Fminbox(Optim.ConjugateGradient()), __get_options())

    zs = results.minimizer
    τ₊ = zs[8n+1:9n]
    z₊ = copy(z)
    Fc!(z, z₊, data; τ=τ₊, p=p)

    x₊ = z₊.x
    y₊ = z₊.y
    μ₊ = z₊.μ
    return τ₊, y₊, State(n, x₊, y₊, μ₊), nothing
end


@doc """
    __policy_opt_priority(z, data, C, ϕ, p; obj_style=1, verbose=false)

Priority-based treatment assignment policy.

Ranks cohorts by priority score and assigns treatment (τ[v] = 1) in order until
capacity C is exhausted. The capacity constraint is on the treated population in p1.

# Priority ranking (based on obj_style):
- obj_style=1: prioritize by recidivism risk φ (higher risk → higher priority)
- obj_style=2: prioritize by ϕ (user-provided priority weights, higher ϕ → higher priority)

# Capacity consumption:
When τ[v] = 1 for cohort v, the capacity consumed is:
- Existing p0 population: x[v, 1]
- New arrivals: β[v]
- Plus existing p1 population: x[v, 3] (already treated)

The algorithm greedily assigns τ[v] = 1 until adding another cohort would exceed C.
"""
function __policy_opt_priority(z, data, C, ϕ, p; obj_style=1, verbose=false, ascending=false)
    n = data[:n]

    # Compute recidivism probabilities for priority ranking
    h₊ = data[:Fh](z.μ; p=p)
    φ₊ = data[:Fφ](h₊; p=p)
    φ0 = φ₊[0]  # untreated recidivism probability
    φ1 = φ₊[1]  # treated recidivism probability

    # Priority score: higher score = higher priority for treatment
    if obj_style == 1
        # Prioritize by recidivism risk reduction potential
        # Higher φ0 → higher priority
        priority_score = φ0
    elseif obj_style == 2
        # Prioritize by user-provided weights (higher ϕ → higher priority)
        priority_score = ϕ
    else
        throw(ArgumentError("obj_style must be 1 or 2"))
    end

    # Sort cohorts by priority (descending)
    sorted_idx = sortperm(priority_score, rev=!ascending)

    # Initialize τ = 0 (no treatment)
    τ₊ = zeros(n)

    # Current treated population (existing p1 + f1)
    current_treated = sum(z.x[:, 3])

    # Greedily assign treatment in priority order
    for v in sorted_idx
        # Check if capacity is already exhausted
        remaining_capacity = C - current_treated
        if remaining_capacity <= 1e-10
            break
        end

        # Capacity consumed if we treat cohort v:
        # - existing p0 in v moves to p1: z.x[v, 1]
        # - new arrivals in v go to p1: data[:β][v]
        capacity_needed = z.x[v, 1] + data[:β][v]

        if capacity_needed <= 1e-10
            # Skip cohorts with zero capacity need
            continue
        end

        if current_treated + capacity_needed <= C
            τ₊[v] = 1.0
            current_treated += capacity_needed
            if verbose
                @info "Treating cohort $v: capacity used = $(current_treated) / $C"
            end
        elseif remaining_capacity > 0
            # Partial treatment: τ[v] = fraction that fits
            τ₊[v] = remaining_capacity / capacity_needed
            current_treated += remaining_capacity
            if verbose
                @info "Partially treating cohort $v (τ=$(τ₊[v])): capacity used = $(current_treated) / $C"
            end
            # Capacity is now exhausted
            break
        end
        # If this cohort doesn't fit at all and no partial treatment possible,
        # continue to check remaining cohorts (they might be smaller and fit)
    end

    if verbose
        @info "Final capacity utilization: $(current_treated) / $C"
        @info "Treatment assignment: $(sum(τ₊ .> 0)) cohorts treated"
    end

    # Apply treatment and compute next state
    z₊ = copy(z)
    Fc!(z, z₊, data; τ=τ₊, p=p)

    y₊ = z₊.y
    return τ₊, y₊, z₊, nothing
end