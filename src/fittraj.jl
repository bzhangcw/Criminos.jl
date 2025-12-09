################################################################################
# Fitting Mixed-In Effect by Optimization 
# - trajectory based fitting
# @note:
#   the mixed-in effect is fitted by the trajectory data
################################################################################

using LinearAlgebra, SparseArrays
using JuMP, COPT

function fit_trajectory(
    data::D,
    vec_Ψ::Vector{Ψ};
    Tₘ=100,
    ϵₚ=1e1,
    opt_constraints=true,
    bool_H_var=true,
    H_val=nothing
) where {D,Ψ}
    μ = Criminos.default_barrier_option.μ

    T = min(data.Tₘ - 1, Tₘ)
    n = data.n

    # -----------------------------------------------------
    # set a lower bound matrix to allow strong deterrence
    _Γ = diagm(data.γ)
    _Γₕ = vec_Ψ[1].M * _Γ
    G = blockdiag(
        [sparse(_Γₕ' * inv(I - _Γ) * _Γₕ)
         for _Ψ in vec_Ψ]...
    )
    # -----------------------------------------------------

    md = Model(optimizer_with_attributes(
        () -> COPT.ConeOptimizer(),
        "RelGap" => 1e-2,
        "FeasTol" => 1e-4,
        "DualTol" => 1e-4,
    )
    )
    @variable(md, rv[1:n, 1:T])
    @variable(md, rabs[1:n, 1:T] .>= 0)
    @variable(md, yv[1:n, 1:T] .>= 0)
    @variable(md, gv[1:n])
    @variable(md, γv[1:n], lower_bound = 0.0, upper_bound = 1.0)
    Γv = diagm(γv)
    if bool_H_var
        @variable(md, Hv[1:n, 1:n], PSD)
        @constraint(md, Hv - G >= ϵₚ * I, PSDCone())
        set_upper_bound.(Hv, 1e3)
    elseif H_val !== nothing
        @info "using provided H"
        Hv = zeros(n, n)
        Hv .= H_val
    else
        # with fixed H
        Hv = G + ϵₚ * I
    end
    @constraint(md,
        [t in 2:T],
        0 .== begin
            Hv * data.traj_y[:, t] + gv +
            vec_Ψ[1].Γₕ' * (data.traj_x[:, t] + data.λ) +
            1 / μ .* (yv[:, t] - data.traj_y[:, t])
        end
    )

    @constraint(md,
        [t in 1:T],
        yv[:, t] - data.traj_y[:, t+1] .== rv[:, t]
    )

    # @constraint(md,
    #     [t in 1:T],
    #     yv[:, t] - 0.99 .* data.traj_x[:, t] .<= 0
    # )
    # optional
    opt_constraints && begin
        @info "apply optional constraint for stability"
        @constraint(md,
            [t in 1:T],
            yv[:, t] - 0.95 .* data.traj_x[:, t+1] .<= 0
        )
    end
    @constraint(
        md,
        [t in 1:T],
        rabs[:, t] .>= rv[:, t]
    )
    @constraint(
        md,
        [t in 1:T],
        rabs[:, t] .>= -rv[:, t]
    )
    @objective(md, Min, sum(rabs[:, T]))
    optimize!(md)
    mipar = MixedInParams(rand(n, n), rand(n, n), rand(n, n), rand(n, n))
    if termination_status(md) == MOI.OPTIMAL
        # prepare mixed-in effects
        _H = value.(Hv)
        _g = value.(gv)

        ω∇ω(y, τ, x) = begin
            _w = 1 / 2 * y' * _H * y + y' * _g
            ∇ = _H * y + _g
            _w, ∇
        end
        _args = (mipar, ω∇ω, G, md, _H, _g)
    else
        _args = (mipar, nothing, G, md, nothing, nothing)
        write_to_file(md, "md.cbf")
    end
    return _args
end