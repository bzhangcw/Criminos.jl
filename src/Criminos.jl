module Criminos

using JuMP, Gurobi
greet() = print("Hello World!")

EPS_FP = 1e-5
USE_KL = false
ℓ = 1
struct BarrierOption
    μ::Float64
end
mutable struct GNEPMixinOption
    y::Union{Nothing,Any}
    ycon::Union{Nothing,Any}
    is_setup::Bool
    is_kl::Bool
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
    if !USE_KL
        _md = Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV[]),
            "NonConvex" => 2,
            "LogToConsole" => 0
        ))
        default_gnep_mixin_option = GNEPMixinOption(
            nothing,
            nothing,
            false,
            USE_KL,
            _md
        )
    else
        _md = Model(Ipopt.Optimizer)
        set_attribute(_md, "max_cpu_time", 60.0)
        # set_attribute(_md, "print_level", 0)
        default_gnep_mixin_option = GNEPMixinOption(
            nothing,
            nothing,
            false,
            USE_KL,
            _md
        )
    end

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
include("utils.jl")
include("decision.jl")

export find_x

function simulate(
    vector_ms::Vector{MarkovState{R,TR}},
    vec_Ψ,
    Fp;
    K=10000, metrics=[Lₓ, Lᵨ, ΔR, KL], bool_opt=true
) where {R,TR}
    ε = Dict()
    # --------------------------------------------------
    # !!! do not change the original
    # --------------------------------------------------
    Vz = copy.(vector_ms)
    traj = []
    r = length(Vz)
    kₑ = 0
    eps = zeros(r)
    for k::Int in 1:K
        push!(traj, Vz)
        _Vz = copy.(Vz)
        # FP iteration;
        Fp(_Vz)
        for (id, z) in enumerate(_Vz)
            # cal FP residual
            eps[id] = norm(Vz[id].z - z.z)
        end
        kₑ = k
        if maximum(eps) < 1e-7
            @info "converged in $kₑ steps"
            break
        end
        Vz = _Vz
        k += 1
    end
    for (id, z) in enumerate(Vz)
        for (func, fname) in metrics
            z₊ = traj[end][id]
            ε[id, fname] = [func(traj[j][id], z₊) for j in 1:kₑ]
        end
    end
    return kₑ, ε, traj, bool_opt
end


export MarkovState
export BidiagSys
export F
end # module Criminos
