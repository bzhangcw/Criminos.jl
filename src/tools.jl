using LinearAlgebra
using JuMP, Gurobi
# import MathOptInterface as MOI


function find_x(size, _x, _y; x0=nothing)

    model = default_xinit_option.model
    empty!(model)

    if haskey(model, :ρ)
        delete(model, model[:ρ])
        unregister(model, :ρ)
    end
    _c = -rand(Float64, size)
    # if !default_xinit_option.is_setup
    if x0 |> isnothing
        @variable(model, x[1:size] .>= 0)
    else
        x = normalize(x0, 1)
        x *= (_x + _y)
    end
    @variable(model, 1 .>= ρ[1:size] .>= 0)
    c1 = @constraint(model, ρ' * x == _x)
    c2 = @constraint(model, (1 .- ρ)' * x == _y)
    @objective(model, Min, _c' * ρ)
    default_xinit_option.is_setup = true
    optimize!(model)
    xx, pp = value.(x), value.(ρ)
    delete(model, c1)
    delete(model, c2)
    return xx, pp
end