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



function simulate(z₀, Ψ, Fp; K=10, metrics=[Lₓ, Lᵨ, ΔR, KL], bool_opt=true)

    z₊, bool_opt = Criminos.forward(z₀, Fp; K=K)
    ε = Dict()
    for (idx, (func, fname)) in enumerate(metrics)
        ε[fname] = zeros(z₊.k + 1)
    end

    z = copy(z₀)
    traj = [z]
    kₑ = 0
    for k in 1:z₊.k+1
        for (idx, (func, fname)) in enumerate(metrics)
            ε[fname][k] = func(z, z₊)
        end
        # move to next iterate
        _z, _τ = Fp(z)
        z₁ = MarkovState(k, _z, _τ)
        # assign mixed-in function value
        z₁.f = z.f
        # copy previous recidivists
        z₁.y₋ = copy(z.y)

        kₑ = k
        push!(traj, z)
        if (z₁.z - z.z) |> norm < 1e-7
            break
        end
        k += 1

        z = z₁
    end
    return kₑ, z₊, ε, traj, bool_opt
end


export MarkovState
export BidiagSys
export F, Φ, J
end # module Criminos
