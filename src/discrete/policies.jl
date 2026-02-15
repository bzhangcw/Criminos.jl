@doc """
    __policy_opt_priority(z, data, C, ϕ, p; obj_style=1, verbose=false, mode=:existing)

Priority-based treatment assignment policy.

Ranks cohorts by priority score and assigns treatment (τ[v] = 1) in order until
capacity C is exhausted. The capacity constraint is on the treated population in p1.

# Priority ranking (based on obj_style):
- obj_style=1: prioritize by recidivism risk φ (higher risk → higher priority)
- obj_style=2: prioritize by ϕ (user-provided priority weights, higher ϕ → higher priority)

# Treatment mode:
- mode=:existing: treatment applies to existing p0 only, capacity = x[v, 1]
- mode=:new: treatment applies to new arrivals only, capacity = β[v]
- mode=:both: treatment applies to both, capacity = x[v, 1] + β[v]
- mode=:uponentry: treatment upon entry, capacity = β[v] + b[v] (eligible inflow)

The algorithm greedily assigns τ[v] = 1 until adding another cohort would exceed C.
"""
function __policy_opt_priority(z, data, C, ϕ, p; obj_style=1, verbose=false, ascending=false, mode=:existing)
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

        # Capacity consumed if we treat cohort v (depends on mode):
        if mode == :new
            capacity_needed = data[:β][v]  # new arrivals only
        elseif mode == :existing
            @warn "mode == :existing is not recommended, because it allows treatment in mid-of-probation"
            capacity_needed = z.x[v, 1]  # existing p0 only
        elseif mode == :both
            @warn "mode == :both is not recommended, because it allows treatment in mid-of-probation"
            capacity_needed = z.x[v, 1] + data[:β][v]  # both
        elseif mode == :uponentry
            capacity_needed = z.b[v] + data[:β][v]  # eligible inflow (arrivals + untreated returns)
        else
            throw(ArgumentError("mode must be :existing, :new, :both, or :uponentry, got $mode"))
        end

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
    Fc!(z, z₊, data; τ=τ₊, p=p, mode=mode)

    y₊ = z₊.y
    return τ₊, y₊, z₊, nothing
end