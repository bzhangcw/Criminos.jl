using JuMP, NLPModels, NLPModelsJuMP, MadNLP, MadNLPHSL
using ADNLPModels
solver = MadNLPHSL.Ma57Solver

@doc raw"""
    __policy_opt_sd_madnlp_adnlp(z, data, C, ϕ, p; obj_style=1, verbose=true, max_wall_time=500.0, accuracy=1e-4, τ₀=nothing, mode=:existing)

Steady-state optimization using ADNLPModels directly (no JuMP).

# Variables
- τ (n): treatment assignment
- x (4n): population state [p0, f0, p1, f1]
- y (4n): reoffense counts
- b (n): untreated returns (b₀)
- Total: 10n variables

# Arguments
- `z`: current state
- `data`: problem data
- `C`: capacity constraint
- `ϕ`: priority weights (for obj_style=2)
- `p`: treatment effect parameter
- `obj_style`: 1 = min sum(y), 2 = min sum(x)' * ϕ
- `mode`: treatment mode (:existing, :new, :both, or :uponentry)

# Returns
- `τ₊`: optimal treatment assignment
- `y₊`: resulting reoffense counts
- `z₊`: resulting state
- `stats`: solver statistics
"""
function __policy_opt_sd_madnlp_adnlp(z, data, C, ϕ, p; obj_style=1, verbose=true, max_wall_time=500.0, accuracy=1e-4, τ₀=nothing, mode=:existing)
    n = data[:n]
    nvar = 10n  # τ (n) + x (4n) + y (4n) + b (n)
    ncon = 9n + 1  # fixed point constraints (8n) + b constraint (n) + capacity (1)

    # Variable layout: [τ; x_p0; x_f0; x_p1; x_f1; y_p0; y_f0; y_p1; y_f1; b]
    # Indices
    idx_τ = 1:n
    idx_x = (n+1):(5n)
    idx_y = (5n+1):(9n)
    idx_b = (9n+1):(10n)

    # Bounds: τ ∈ [0,1], x ≥ 0, y ≥ 0, b ≥ 0
    lvar = zeros(nvar)
    uvar = fill(Inf, nvar)
    uvar[idx_τ] .= 1.0  # τ ≤ 1

    # Initial point from current state
    x0 = zeros(nvar)
    if τ₀ === nothing
        x0[idx_τ] .= 0.0 # must be feasible.
    else
        x0[idx_τ] .= τ₀
    end
    x0[idx_x] .= vec(z.x)  # flatten column-major: [p0; f0; p1; f1]
    x0[idx_y] .= vec(z.y)
    x0[idx_b] .= z.b

    # Objective function
    function obj(w)
        if obj_style == 1
            # min sum(y)
            return sum(w[idx_y])
        else
            # min sum(x, dims=2)' * ϕ
            x_mat = reshape(w[idx_x], n, 4)
            return sum(sum(x_mat, dims=2) .* ϕ)
        end
    end

    # Constraint function: c(w) = 0 for equality, c(w) ≤ 0 for inequality
    # [x - F_x; y - F_y; b - b₀; sum(x_p1) - C]
    function con!(c, w)
        τv = w[idx_τ]
        xv = reshape(w[idx_x], n, 4)
        yv = reshape(w[idx_y], n, 4)
        bv = w[idx_b]

        # Compute μ
        μ = sum(yv) / (sum(xv) + 1e-8)

        # Compute F (b₀ is the 9th return, xhalf is the 10th)
        x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, b₀, xhalf = F(xv, yv, bv, μ, data; τ=τv, p=p, mode=mode)

        # Fixed point constraints: x == F_x, y == F_y
        c[1:n] .= xv[:, 1] .- x₊1
        c[n+1:2n] .= xv[:, 2] .- x₊2
        c[2n+1:3n] .= xv[:, 3] .- x₊3
        c[3n+1:4n] .= xv[:, 4] .- x₊4
        c[4n+1:5n] .= yv[:, 1] .- y₊1
        c[5n+1:6n] .= yv[:, 2] .- y₊2
        c[6n+1:7n] .= yv[:, 3] .- y₊3
        c[7n+1:8n] .= yv[:, 4] .- y₊4

        # Fixed point constraint for b: b == b₀
        c[8n+1:9n] .= bv .- b₀

        # Capacity constraint: sum(x_p1) ≤ C  =>  sum(x_p1) - C ≤ 0
        # apply treatment to tlx: treatment-allocated population in p1
        c[9n+1] = sum(xhalf[:, 3]) - C

        return c
    end

    # Constraint bounds: equality for fixed point (8n + n), inequality for capacity (1)
    lcon = zeros(ncon)
    ucon = zeros(ncon)
    ucon[9n+1] = 0.0  # capacity: c ≤ 0
    lcon[9n+1] = -Inf  # capacity: -∞ ≤ c

    # Create ADNLPModel
    nlp = ADNLPModel!(obj, x0, lvar, uvar, con!, lcon, ucon)

    @info "starting to solve with MadNLP"
    # Solve with MadNLP
    lvl = verbose ? MadNLP.INFO : MadNLP.ERROR
    stats = madnlp(nlp,
        linear_solver=solver,
        max_wall_time=max_wall_time,
        max_iter=10000,
        print_level=lvl,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        hessian_approximation=MadNLP.CompactLBFGS,
        tol=accuracy
    )

    # Extract solution
    sol = stats.solution
    τ₊ = sol[idx_τ]
    x₊ = reshape(sol[idx_x], n, 4)
    y₊ = reshape(sol[idx_y], n, 4)
    b₊ = sol[idx_b]
    μ₊ = safe_ratio(sum(y₊), sum(x₊))

    return τ₊, y₊, State(n, x₊, y₊, μ₊; b=b₊), stats
end

@doc raw"""
    __policy_opt_sd_madnlp_jump(z, data, C, ϕ, p; obj_style=1)

**DEPRECATED**: Use `__policy_opt_sd_madnlp_adnlp` instead.

Steady-state optimization using JuMP + MadNLP.

# Note
This is a slower alternative to `__policy_opt_sd_madnlp_adnlp` that uses JuMP for model formulation.
Does not support `mode` parameter (uses default treatment assignment).
"""
function __policy_opt_sd_madnlp_jump(z, data, C, ϕ, p; obj_style=1)
    m = Model(MadNLP.Optimizer)
    n = data[:n]
    # columns labels: p0, f0, p1, f1
    @variable(m, τv[1:n] >= 0)
    set_upper_bound.(τv, 1.0)
    # x, and y
    @variable(m, xv[1:n, 1:4] >= 0) # x
    @variable(m, yv[1:n, 1:4] >= 0)
    @variable(m, bv[1:n] >= 0)      # b (untreated returns)
    @constraint(m, cap, sum(xv[:, 3]) <= C)
    μv = sum(yv) / (sum(xv) + 1e-5)
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, b₀, xhalf = F(xv, yv, bv, μv, data; τ=τv, p=p)

    # fixed point constraints.
    @constraint(m, xv[:, 1] .== x₊1)
    @constraint(m, xv[:, 2] .== x₊2)
    @constraint(m, xv[:, 3] .== x₊3)
    @constraint(m, xv[:, 4] .== x₊4)
    @constraint(m, yv[:, 1] .== y₊1)
    @constraint(m, yv[:, 2] .== y₊2)
    @constraint(m, yv[:, 3] .== y₊3)
    @constraint(m, yv[:, 4] .== y₊4)
    @constraint(m, bv .== b₀)  # b fixed point
    if obj_style == 1
        @objective(m, Min, sum(yv))
    elseif obj_style == 2
        @objective(m, Min, sum(xv; dims=2)' * ϕ)
    end

    # translate to NLPModels
    @info "starting to parse as a NLP"
    nlp = MathOptNLPModel(m)
    @info "finished to parse as a NLP"
    @info "push to MadNLP"
    stats = madnlp(nlp,
        linear_solver=solver,
        max_wall_time=900.0,
        max_iter=1000,
        print_level=MadNLP.INFO,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        hessian_approximation=MadNLP.CompactLBFGS,
        tol=1e-6
    )
    τ₊ = stats.solution[1:n]
    x₊ = reshape(stats.solution[n+1:5n], n, 4)
    y₊ = reshape(stats.solution[5n+1:9n], n, 4)
    b₊ = stats.solution[9n+1:10n]

    μ₊ = safe_ratio(sum(y₊), sum(x₊))
    return τ₊, y₊, State(n, x₊, y₊, μ₊; b=b₊), nothing
end



@doc raw"""
    __policy_opt_myopic_madnlp(z, data, C, ϕ, p; obj_style=1, verbose=false)

Myopic (one-step lookahead) optimization using JuMP + MadNLP.

Optimizes treatment for the next period only, without considering long-term steady-state effects.

# Note
Does not support `mode` parameter (uses default treatment assignment).
"""
function __policy_opt_myopic_madnlp(z, data, C, ϕ, p; obj_style=1, verbose=false)
    m = Model(MadNLP.Optimizer)
    n = data[:n]
    @variable(m, τv[1:n] >= 0)
    set_upper_bound.(τv, 1.0)
    # one-step lookahead
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, _, _ = F(z.x, z.y, z.b, z.μ, data; τ=τv, p=p)
    @constraint(m, cap, sum(x₊3) <= C)

    if obj_style == 1
        @objective(m, Min, sum(y₊1) + sum(y₊2) + sum(y₊3) + sum(y₊4))
    else
        throw(ValueError("obj_style must be 1"))
    end

    # translate to NLPModels
    nlp = MathOptNLPModel(m)
    lvl = verbose ? MadNLP.INFO : MadNLP.ERROR
    stats = madnlp(nlp,
        linear_solver=solver,
        max_wall_time=900.0,
        max_iter=1000,
        print_level=lvl,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        hessian_approximation=MadNLP.CompactLBFGS,
        tol=1e-4
    )
    τ₊ = stats.solution[1:n]

    z₊ = copy(z)
    Fc!(z, z₊, data; τ=τ₊, p=p)

    x₊ = z₊.x
    y₊ = z₊.y
    μ₊ = z₊.μ
    return τ₊, y₊, z₊, nothing
end

@doc raw"""
    __policy_opt_myopic_madnlp_adnlp(z, data, C, ϕ, p; obj_style=1, verbose=false)

Myopic (one-step lookahead) optimization using ADNLPModels directly (no JuMP).

Optimizes treatment for the next period only, without considering long-term steady-state effects.
This is faster than `__policy_opt_myopic_madnlp` as it uses ADNLPModels directly.

# Variables
- τ (n): treatment assignment

# Note
Does not support `mode` parameter (uses default treatment assignment).
"""
function __policy_opt_myopic_madnlp_adnlp(z, data, C, ϕ, p; obj_style=1, verbose=false)
    n = data[:n]
    nvar = n  # only τ
    ncon = 1  # capacity constraint

    # Bounds: τ ∈ [0,1]
    lvar = zeros(nvar)
    uvar = ones(nvar)

    # Initial point
    x0 = fill(0.5, nvar)

    # Precompute current state (fixed during optimization)
    x_curr = z.x
    y_curr = z.y
    b_curr = z.b
    μ_curr = z.μ

    # Objective: min sum(y₊) where y₊ = F_y(x, y, μ; τ)
    function obj(τv)
        x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, _, _ = F(x_curr, y_curr, b_curr, μ_curr, data; τ=τv, p=p)
        return sum(y₊1) + sum(y₊2) + sum(y₊3) + sum(y₊4)
    end

    # Constraint: sum(x₊3) ≤ C  =>  sum(x₊3) - C ≤ 0
    function con!(c, τv)
        x₊1, x₊2, x₊3, x₊4, _, _, _, _, _, _ = F(x_curr, y_curr, b_curr, μ_curr, data; τ=τv, p=p)
        c[1] = sum(x₊3) - C
        return c
    end

    # Constraint bounds: -∞ ≤ c ≤ 0
    lcon = [-Inf]
    ucon = [0.0]

    # Create ADNLPModel
    nlp = ADNLPModel!(obj, x0, lvar, uvar, con!, lcon, ucon)

    # Solve with MadNLP
    lvl = verbose ? MadNLP.INFO : MadNLP.ERROR
    stats = madnlp(nlp,
        linear_solver=solver,
        max_wall_time=900.0,
        max_iter=1000,
        print_level=lvl,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        hessian_approximation=MadNLP.CompactLBFGS,
        tol=1e-4
    )

    τ₊ = stats.solution

    # Apply the policy to get next state
    z₊ = copy(z)
    Fc!(z, z₊, data; τ=τ₊, p=p)

    return τ₊, z₊.y, z₊, stats
end