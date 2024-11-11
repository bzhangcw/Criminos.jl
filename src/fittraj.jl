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
    ϵₚ=1e1
) where {D,Ψ}
    μ = Criminos.default_barrier_option.μ
    bool_H_var = true
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
    ))
    @variable(md, rv[1:n, 1:T])
    @variable(md, rabs[1:n, 1:T] .>= 0)
    @variable(md, yv[1:n, 1:T] .>= 0)
    @variable(md, gv[1:n])
    if bool_H_var
        @variable(md, Hv[1:n, 1:n], PSD)
        @constraint(md, Hv - G >= ϵₚ * I, PSDCone())
        set_upper_bound.(Hv, 1e5)
        @constraint(md,
            [t in 1:T],
            0 .== begin
                Hv * data.traj_y[:, t+1] + gv +
                vec_Ψ[1].Γₕ' * data.traj_x[:, t] +
                1 / μ .* (yv[:, t] - data.traj_y[:, t])
            end
        )
    else
        # with fixed H
        Hv = G + ϵₚ * I
        @constraint(md,
            [t in 1:T],
            0 .== begin
                Hv * yv[:, t] + gv +
                vec_Ψ[1].Γₕ' * data.traj_x[:, t] +
                1 / μ .* (yv[:, t] - data.traj_y[:, t])
            end
        )
    end

    @constraint(md,
        [t in 1:T],
        yv[:, t] - data.traj_y[:, t+1] .== rv[:, t]
    )

    @constraint(md,
        [t in 1:T],
        yv[:, t] - 0.99 .* data.traj_x[:, t] .<= 0
    )
    # optional
    @constraint(md,
        [t in 1:T],
        yv[:, t] - 0.95 .* data.traj_x[:, t+1] .<= 0
    )
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
    # @objective(md, Min, sum(1/T * rabs * (1:T)))
    @objective(md, Min, sum(1 / T * rabs))
    optimize!(md)

    if termination_status(md) == MOI.OPTIMAL
        # prepare mixed-in effects
        _H = value.(Hv)
        _g = value.(gv)

        ω∇ω(y, τ) = begin
            _w = 1 / 2 * y' * _H * y + y' * _g
            ∇ = _H * y + _g
            _w, ∇
        end
        _args = (ω∇ω, G, md, _H, _g)
    else
        write_to_file(md, "md.cbf")
    end
    return _args
end