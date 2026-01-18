using UnoSolver, JuMP

function __policy_opt_sd(z, data, C, ϕ, p; obj_style=1)
    # m = Model(MadNLP.Optimizer)
    opt = optimizer_with_attributes(UnoSolver.Optimizer,
        "output_flag" => false,
        "preset" => "filtersqp",
        "primal_tolerance" => 1e-5,
        "dual_tolerance" => 1e-5,
        "hessian_model" => "identity",
    )
    m = Model(opt)
    n = data[:n]
    # columns labels: p0, f0, p1, f1
    @variable(m, τv[1:n] >= 0)
    set_upper_bound.(τv, 1.0)
    # x, and y
    @variable(m, xv[1:n, 1:4] >= 0) # x
    @variable(m, yv[1:n, 1:4] >= 0)
    @constraint(m, cap, sum(xv[:, 3]) <= C)
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4 = F(xv, yv, sum(yv) / (sum(xv) + 1e-5), data; τ=τv, p=p)

    # fixed point constraints.
    @constraint(m, xv[:, 1] .== x₊1)
    @constraint(m, xv[:, 2] .== x₊2)
    @constraint(m, xv[:, 3] .== x₊3)
    @constraint(m, xv[:, 4] .== x₊4)
    @constraint(m, yv[:, 1] .== y₊1)
    @constraint(m, yv[:, 2] .== y₊2)
    @constraint(m, yv[:, 3] .== y₊3)
    @constraint(m, yv[:, 4] .== y₊4)
    if obj_style == 1
        @objective(m, Min, sum(yv))
    elseif obj_style == 2
        @objective(m, Min, sum(xv; dims=2)' * ϕ)
    end
    optimize!(m)
    τ₊ = value.(τv)
    x₊ = value.(xv)
    y₊ = value.(yv)
    μ₊ = safe_ratio(sum(y₊), sum(x₊))
    return τ₊, y₊, State(n, x₊, y₊, μ₊), nothing
end


function __policy_opt_myopic(z, data, C, ϕ, p; obj_style=1, verbose=false)
    # m = Model(MadNLP.Optimizer)
    opt = optimizer_with_attributes(UnoSolver.Optimizer,
        "output_flag" => true,
        "preset" => "filtersqp",
        "primal_tolerance" => 1e-5,
        "dual_tolerance" => 1e-5,
    )
    m = Model(opt)
    n = data[:n]
    @variable(m, τv[1:n] >= 0)
    set_upper_bound.(τv, 1.0)
    # one-step lookahead
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4 = F(z.x, z.y, z.μ, data; τ=τv, p=p)
    @constraint(m, cap, sum(x₊3) <= C)

    if obj_style == 1
        @objective(m, Min, sum(y₊1) + sum(y₊2) + sum(y₊3) + sum(y₊4))
    else
        throw(ValueError("obj_style must be 1"))
    end
    @info "optimization started"
    optimize!(m)
    @info "optimization finished"
    τ₊ = value.(τv)
    z₊ = copy(z)
    Fc!(z, z₊, data; τ=τ₊, p=p)

    x₊ = z₊.x
    y₊ = z₊.y
    μ₊ = z₊.μ
    return τ₊, y₊, z₊, nothing
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
        # Capacity consumed if we treat cohort v:
        # - existing p0 in v moves to p1: z.x[v, 1]
        # - new arrivals in v go to p1: data[:β][v]
        capacity_needed = z.x[v, 1] + data[:β][v]

        if current_treated + capacity_needed <= C
            τ₊[v] = 1.0
            current_treated += capacity_needed
            if verbose
                @info "Treating cohort $v: capacity used = $(current_treated) / $C"
            end
        else
            # Can we partially treat this cohort?
            remaining_capacity = C - current_treated
            if remaining_capacity > 0 && capacity_needed > 0
                # Partial treatment: τ[v] = fraction that fits
                τ₊[v] = remaining_capacity / capacity_needed
                current_treated += remaining_capacity
                if verbose
                    @info "Partially treating cohort $v (τ=$(τ₊[v])): capacity used = $(current_treated) / $C"
                end
            end
            # No more capacity
            break
        end
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