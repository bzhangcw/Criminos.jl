using JuMP, NLPModels, NLPModelsJuMP, MadNLP, MadNLPHSL
solver = MadNLPHSL.Ma57Solver

function __policy_opt_sd_madnlp(z, data, C, ϕ, p; obj_style=1)
    m = Model(MadNLP.Optimizer)
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

    μ₊ = safe_ratio(sum(y₊), sum(x₊))
    return τ₊, y₊, State(n, x₊, y₊, μ₊), nothing
end


function __policy_opt_myopic_madnlp(z, data, C, ϕ, p; obj_style=1, verbose=false)
    m = Model(MadNLP.Optimizer)
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