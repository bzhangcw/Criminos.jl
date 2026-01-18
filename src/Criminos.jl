module Criminos
using LinearAlgebra, SparseArrays, SpecialFunctions
using ForwardDiff, LinearAlgebra, Random, Statistics
using Printf, LaTeXStrings, Plots, ProgressMeter, ColorSchemes
using CSV, Tables, DataFrames, YAML, JSON, PrettyTables, HDF5
using JuMP, UnoSolver

include("tools.jl")

include("discrete/state.jl")
include("discrete/glquad.jl")
include("discrete/routing.jl")
include("discrete/datarec.jl")
include("discrete/fixp2drec.jl")
include("continuous/fixp2cont.jl")

include("discrete/policies.jl")
include("discrete/policies.madnlp.jl")

export generate_random_data, compute_Psi_ct_composed
export F, Fc!
export State, randz, z_diff, get_x, get_y, safe_ratio
export visualize_results, visualize_state, visualize_matrix, visualize_vector


function run(z₀, data;
    fτ=nothing,
    p=zeros(data[:n]),
    max_iter=500,
    verbose=false,
    validate=false,
    func_traj=copy,
    keep_traj=false
)
    n = data[:n]
    z = copy(z₀)
    traj = keep_traj ? [func_traj(z)] : []
    for t in 1:500
        τ₊, _... = isnothing(fτ) ? (zeros(n), nothing) : fτ(z)
        z₊ = copy(z)
        Fc!(z, z₊, data; τ=τ₊, p=p, validate=validate)
        ϵ = z_diff(z₊, z)
        z = z₊
        # @info "" z.x z.y z.μ
        if (ϵ < 1e-6) || (t > max_iter)
            println("ϵ = $ϵ, convergence in $t steps")
            break
        end
        keep_traj && push!(traj, func_traj(z))
        if verbose
            println("ϵ = $ϵ, currently $t steps")
        end
    end
    return z, traj
end

export run

end # module Criminos
