module Criminos
using LinearAlgebra, SparseArrays, SpecialFunctions
using ForwardDiff, LinearAlgebra, Random, Statistics
using Printf, LaTeXStrings, Plots, ProgressMeter, ColorSchemes
using CSV, Tables, DataFrames, YAML, JSON, PrettyTables, HDF5
using JuMP, UnoSolver

include("tools.jl")

include("discrete/state.jl")
include("discrete/idiosyncrasy.jl")        # base interface + constant idiosyncrasy
include("discrete/idiosyncrasy_gamma.jl")  # Gamma frailty (GL quadrature)
include("discrete/routing.jl")
include("discrete/data.jl")
include("discrete/fixedpoint.jl")
include("continuous/fixp2cont.jl")

include("discrete/policies.jl")
include("discrete/policies.madnlp.jl")
include("discrete/policies.uno.jl")

export generate_random_data, compute_Psi_ct_composed
export F, Fc!
export State, randz, z_diff, get_x, get_y, safe_ratio
export visualize_results, visualize_state, visualize_matrix, visualize_vector


function run(z₀, data;
    fτ=nothing,
    p=zeros(data[:n]),
    mode=:existing,  # :existing, :new, or :both
    max_iter=5000,
    verbose=false,
    validate=false,
    func_traj=copy,
    keep_traj=false,
    verbose_interval=20
)
    n = data[:n]
    z = copy(z₀)
    traj = keep_traj ? [func_traj(z)] : []
    t = 0
    while true
        τ₊, _... = isnothing(fτ) ? (zeros(n), nothing) : fτ(z)
        z₊ = copy(z)
        Fc!(z, z₊, data; τ=τ₊, p=p, mode=mode, validate=validate)
        ϵ = z_diff(z₊, z)
        z = z₊
        # @info "" z.x z.y z.μ
        if (ϵ < 1e-6) || (t > max_iter)
            println("ϵ = $ϵ, convergence in $t steps")
            break
        end
        keep_traj && push!(traj, func_traj(z))
        if verbose && (t % verbose_interval == 0)
            println("ϵ = $ϵ, currently $t steps")
        end
        t += 1
    end
    return z, traj
end

export run

end # module Criminos
