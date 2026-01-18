

column_names = [:p0, :f0, :p1, :f1]
randz(n; scale=50) = begin
    x = zeros(n, 4)
    x[:, 1] = rand(n) .* scale
    x[:, 2] = zeros(n)
    x[:, 3] = rand(n) .* scale
    x[:, 4] = zeros(n)
    # ---------------------------
    y = zeros(n, 4)
    y[:, 1] = rand(n) .* x[:, 1]
    y[:, 2] = rand(n) .* x[:, 2]
    y[:, 3] = rand(n) .* x[:, 3]
    y[:, 4] = rand(n) .* x[:, 4]
    # ---------------------------
    μ = sum(y) / sum(x)
    return State(n, x, y, μ)
end

F(x, y, μ, data;
    τ=zeros(data[:n]), p=ones(data[:n])
) = begin
    n = data[:n]
    h₊ = data[:Fh](μ; p=p)
    φ₊ = data[:Fφ](h₊; p=p)

    # complementary of these fractions
    γ_c = 1.0 .- data[:γ]
    σ_c = 1.0 .- data[:σ]
    τ_c = 1.0 .- τ

    # treatment, and retention movements
    #    reshape the population and groups immediately.
    # p0
    xhalf1 = τ_c .* (x[:, 1] + data[:β])
    # f0
    xhalf2 = x[:, 2]
    # p1
    xhalf3 = τ .* (x[:, 1] + data[:β]) + x[:, 3]
    # f1
    xhalf4 = x[:, 4]

    # error = sum(xhalf1 + xhalf2 + xhalf3 + xhalf4) - sum(x)
    # @assert error < 1e-4
    # @info "error = $error"
    # @info "" γ_c σ_c τ_c
    # @info "" xhalf1 xhalf2 xhalf3 xhalf4
    # ---------------------------
    # p0, f0
    y₊1 = φ₊[0] .* xhalf1
    y₊2 = φ₊[0] .* xhalf2
    # p1, f1
    y₊3 = φ₊[1] .* xhalf3
    y₊4 = φ₊[1] .* xhalf4
    # ---------------------------
    # p0, f0

    x₊1 = data[:Px]' * (data[:γ] .* (xhalf1 - y₊1)) + data[:Py]' * (data[:γ] .* y₊1)
    x₊2 = (data[:Px]' * (data[:σ] .* (xhalf2 - y₊2))
           + data[:Px]' * (γ_c .* (xhalf1 - y₊1)) + data[:Py]' * (γ_c .* y₊1)
    )
    b0 = data[:Py]' * (data[:r₀] .* σ_c .* xhalf2 + data[:r₀] .* data[:σ] .* y₊2)
    x₊1 += b0
    # x₊1 .+= 0.0
    # @info "r = $(sum(x₊1 + x₊2 - xhalf1 - xhalf2))"
    # p1
    x₊3 = data[:Px]' * (data[:γ] .* (xhalf3 - y₊3)) + data[:Py]' * (data[:γ] .* y₊3)
    # f1
    x₊4 = (data[:Px]' * (data[:σ] .* (xhalf4 - y₊4))
           + data[:Px]' * (γ_c .* (xhalf3 - y₊3)) + data[:Py]' * (γ_c .* y₊3)
    )
    b1 = data[:Py]' * (data[:r₁] .* σ_c .* xhalf4 + data[:r₁] .* data[:σ] .* y₊4)
    # add back to x₊1 (p0), x₊3 (p1)
    x₊3 += b1
    # x₊3 .+= 0.0
    # ---------------------------
    total = sum(x₊1) + sum(x₊2) + sum(x₊3) + sum(x₊4)
    # @info "rr = $(total - sum(x))"
    # @info "x3 = $(sum(x₊3) - sum(x[:, 3]))"
    return x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4
end

Fc!(z, z₊, data; τ=zeros(data[:n]), p=ones(data[:n])) = begin
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4 = F(z.x, z.y, z.μ, data; τ=τ, p=p)
    z₊.x[:, 1] .= x₊1
    z₊.x[:, 2] .= x₊2
    z₊.x[:, 3] .= x₊3
    z₊.x[:, 4] .= x₊4
    z₊.y[:, 1] .= y₊1
    z₊.y[:, 2] .= y₊2
    z₊.y[:, 3] .= y₊3
    z₊.y[:, 4] .= y₊4
    z₊.μ = safe_ratio(sum(z₊.y), sum(z₊.x))
end


function __policy_opt_sd(z, data, C, ϕ, p; obj_style=1)
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
    # @objective(m, Min, sum(yv) * 0.5 + 0.5 * xv' * ϕ)
    # @objective(m, Min, λ' * ϕ)
    #     optimize!(m)
    #     τ₊ = value.(τv)
    #     z₊ = value.(zv)
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
    # return τ₊, y₊, z₊, nothing
end


function __policy_opt_myopic(z, data, C, ϕ, p; obj_style=1, verbose=false)
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
    # @objective(m, Min, sum(yv) * 0.5 + 0.5 * xv' * ϕ)
    # @objective(m, Min, λ' * ϕ)
    #     optimize!(m)
    #     τ₊ = value.(τv)
    #     z₊ = value.(zv)
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
        tol=1e-6
    )
    τ₊ = stats.solution[1:n]

    z₊ = copy(z)
    Fc!(z, z₊, data; τ=τ₊, p=p)

    x₊ = z₊.x
    y₊ = z₊.y
    μ₊ = z₊.μ
    return τ₊, y₊, z₊, nothing
end
