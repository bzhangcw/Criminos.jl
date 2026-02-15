using UnoSolver, JuMP
using ForwardDiff

function __policy_opt_sd_uno(z, data, C, ϕ, p; obj_style=1)
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
    optimize!(m)
    τ₊ = value.(τv)
    x₊ = value.(xv)
    y₊ = value.(yv)
    b₊ = value.(bv)
    μ₊ = safe_ratio(sum(y₊), sum(x₊))
    return τ₊, y₊, State(n, x₊, y₊, μ₊; b=b₊), nothing
end


function __policy_opt_myopic_uno(z, data, C, ϕ, p; obj_style=1, verbose=false)
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
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, _, _ = F(z.x, z.y, z.b, z.μ, data; τ=τv, p=p)
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
