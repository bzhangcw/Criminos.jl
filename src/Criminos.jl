module Criminos

using JuMP, Ipopt, COPT, MosekTools
using ProgressMeter
greet() = print("Hello World!")

EPS_FP = 1e-5
USE_KL = false
USE_GUROBI = false
USE_CONIC_DECISION = true

if USE_GUROBI
    using Gurobi
    const GRB_ENV = Ref{Gurobi.Env}()
    GRB_ENV[] = Gurobi.Env()
    create_default_gurobi_model(logf="/tmp/grb.log") = begin
        Model(optimizer_with_attributes(
            () -> Gurobi.Optimizer(GRB_ENV[]),
            "NonConvex" => 2,
            "LogToConsole" => 0,
            "LogFile" => logf
        ))
    end
end
if USE_KL
    using Ipopt
    create_default_ipopt_model() = begin
        Model(optimizer_with_attributes(
            () -> Ipopt.Optimizer(),
            "print_level" => 0
        ))
    end
end

create_default_copt_model() = begin
    Model(
        optimizer_with_attributes(
            () -> COPT.Optimizer(),
            "LogToConsole" => false
        ))
end

ℓ = 1
mutable struct BarrierOption
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

mutable struct DecisionOption
    τ::Union{Nothing,Any}
    ℓ::Union{Nothing,Any}
    h::Union{Nothing,Any}
    I::Union{Nothing,Any}
    τcon::Union{Nothing,Any}
    is_setup::Bool
    is_conic::Bool
    model::Union{Nothing,Model}
end


default_barrier_option = BarrierOption(5e-2)
default_gnep_mixin_option = Ref{GNEPMixinOption}()
default_xinit_option = Ref{XInitHelper}()
default_decision_option = Ref{DecisionOption}()
default_decision_conic_option = Ref{DecisionOption}()
function __init__()
    global default_gnep_mixin_option, default_xinit_option
    global default_decision_option, default_decision_conic_option
    if USE_GUROBI
        create_default_gurobi_model()
        default_gnep_mixin_option = GNEPMixinOption(
            nothing,
            nothing,
            false,
            USE_KL,
            _md
        )
    elseif USE_KL
        _md = create_default_ipopt_model()
    else
        _md = create_default_copt_model()
    end
    default_gnep_mixin_option = GNEPMixinOption(
        nothing,
        nothing,
        false,
        USE_KL,
        _md
    )
    default_xinit_option = XInitHelper(
        nothing,
        nothing,
        false,
        create_default_copt_model(),
        nothing,
        nothing
    )
    if USE_CONIC_DECISION
        _md_decision = Model(optimizer_with_attributes(
            () -> Mosek.Optimizer(),
            "MSK_IPAR_LOG" => 0
        ))
        default_decision_conic_option = DecisionOption(
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            false,
            USE_CONIC_DECISION,
            _md_decision
        )
    end
    default_decision_option = DecisionOption(
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        false,
        false,
        create_default_copt_model()
    )
end

export default_barrier_option
export default_gnep_mixin_option
export default_xinit_option
export default_decision_option, default_decision_conic_option

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
include("fit.jl")
include("fittraj.jl")

export find_x

function simulate(
    vector_ms::Vector{MarkovState{R,TR}},
    vec_Ψ,
    Fp;
    K=10000, metrics=[Lₓ, Lᵨ, ΔR, KL],
    bool_verbose=false,
    bool_opt=true,
    tol=1e-7
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
    p = Progress(K)
    for k::Int in 1:K
        push!(traj, Vz)
        _Vz = copy.(Vz)
        # FP iteration
        Fp(_Vz)
        for (id, z) in enumerate(_Vz)
            # cal FP residual
            eps[id] = (norm(Vz[id].x - z.x) + norm(Vz[id].y - z.y)) / maximum(z.x)
        end
        kₑ = k
        if maximum(eps) < tol
            @info "converged in $kₑ steps"
            break
        end
        Vz = _Vz
        k += 1
        next!(p)
    end
    finish!(p)
    @info "final eps: $(maximum(eps))"
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
export F!
export optalg, ν, σ
export tuning
export fit_trajectory
end # module Criminos
