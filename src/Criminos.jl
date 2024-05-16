module Criminos

using JuMP, Gurobi
greet() = print("Hello World!")

EPS_FP = 1e-5

ℓ = 1
struct BarrierOption
    μ::Float64
end
mutable struct GNEPMixinOption
    y::Union{Nothing,Any}
    is_setup::Bool
    model::Union{Nothing,Model}
end
mutable struct XInitHelper
    x::Union{Nothing,Any}
    ρ::Union{Nothing,Any}
    is_setup::Bool
    model::Union{Nothing,Model}
    c1::Union{Nothing,Any}
    c2::Union{Nothing,Any}
end

const GRB_ENV = Ref{Gurobi.Env}()
default_gnep_mixin_option = Ref{GNEPMixinOption}()
default_xinit_option = Ref{XInitHelper}()
default_barrier_option = BarrierOption(0.01)
function __init__()
    global default_gnep_mixin_option, default_xinit_option
    GRB_ENV[] = Gurobi.Env()
    default_gnep_mixin_option = GNEPMixinOption(
        nothing,
        false,
        Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV[]),
            "NonConvex" => 2,
            "LogToConsole" => 0
        ))
    )
    default_xinit_option = XInitHelper(
        nothing,
        nothing,
        false,
        Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV[]),
            "NonConvex" => 2,
            "LogToConsole" => 0,
            "LogFile" => "grb.criminos.findx.log"
        )),
        nothing,
        nothing
    )
end

export default_barrier_option, default_gnep_mixin_option, default_xinit_option

##################################################
# sigfault when using constant env here.
# const GRB_ENV = Ref{Gurobi.Env}()
# GRB_ENV[] = Gurobi.Env()
##################################################

include("state.jl")
include("bidiag.jl")
include("fixpoint.jl")
include("mixin.jl")
include("potfunc.jl")
include("tools.jl")
include("decision.jl")

export find_x



function simulate(z₀::MarkovState, Ψ, Fp; K=10, metrics=[Lₓ, Lᵨ, ΔR, KL], bool_opt=true)
    z₊, bool_opt = Criminos.forward(z₀, Fp; K=K)
    ε = Dict()
    for (idx, (func, fname)) in enumerate(metrics[1])
        ε[fname] = zeros(z₊.k + 1)
    end

    z = copy(z₀)
    traj = [z]
    kₑ = 0
    for k in 1:z₊.k+1
        for (idx, (func, fname)) in enumerate(metrics[1])
            ε[fname][k] = func(z, z₊)
        end
        # move to next iterate
        _z = Fp[1](z)
        z₁ = MarkovState(k, _z, z.τ)
        # assign mixed-in function value
        z₁.f = z.f
        # copy previous recidivists
        z₁.y₋ = copy(z.y)

        kₑ = k
        push!(traj, z)
        if (z₁.z - z.z) |> norm < EPS_FP
            break
        end
        k += 1

        z = z₁
    end
    return kₑ, [ε], [traj], bool_opt
end

function simulate(
    zs::Vector{MarkovState{R,TR}}, Ψ, Fp;
    K=10000, metrics=[Lₓ, Lᵨ, ΔR, KL], bool_opt=true
) where {R,TR}
    ε = Dict()
    Vz = copy.(zs)
    traj = []
    r = length(Vz)
    kₑ = 0
    for k::Int in 1:K
        push!(traj, Vz)
        eps = zeros(r)
        _Vz = []
        for (id, z) in enumerate(Vz)
            # move to next iterate
            _z = Fp[id](z)
            z₁ = MarkovState(k, _z, z.τ)
            # assign mixed-in function value
            z₁.f = z.f
            # copy previous recidivists
            z₁.y₋ = copy(z.y)
            eps[id] = norm(z₁.z - z.z)
            z = z₁
            push!(_Vz, z)
        end
        kₑ = k
        if maximum(eps) < 1e-7
            @info "converged in $kₑ steps"
            break
        end
        k += 1
        Vz = _Vz
    end
    for (id, z) in enumerate(Vz)
        for (idx, (func, fname)) in enumerate(metrics[id])
            z₊ = traj[end][id]
            ε[id, fname] = [func(traj[j][id], z₊) for j in 1:kₑ]
        end
    end
    return kₑ, ε, traj, bool_opt
end


export MarkovState
export BidiagSys
export F, Φ, J
end # module Criminos
