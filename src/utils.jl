using LinearAlgebra
using JuMP, Gurobi
using LaTeXStrings
import MathOptInterface as MOI


"""
locate y and x-y 
based on sum(y) and sum(x-y)
"""
function locate_y(size, _y, x_y; x0=nothing)

    model = default_xinit_option.model
    empty!(model)

    if haskey(model, :y)
        delete(model, model[:y])
        unregister(model, :y)
    end
    _c = -rand(Float64, size)
    if x0 |> isnothing
        @variable(model, x[1:size] .>= 0)
    else
        x = normalize(x0, 1)
        x *= (_y + x_y)
    end
    @variable(model, 1 >= ρ[1:size] .>= 0)
    c1 = @constraint(model, ρ' * x == _y)
    c2 = @constraint(model, x' * (1 .- ρ) == x_y)
    @objective(model, Min, _c' * ρ)
    default_xinit_option.is_setup = true
    optimize!(model)
    xx, yy = value.(x), x .* value.(ρ)
    delete(model, c1)
    delete(model, c2)
    return xx, yy
end
