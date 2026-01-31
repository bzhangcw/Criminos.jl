using DataStructures: SortedDict


# ------------------------------------------------------------
# these are default parameters for the data generation
# ------------------------------------------------------------
const a_risks = [0.0 -0.270 -0.456 -0.709 -1.282 -1.808]
const a_size = [2, 5, 5, 10, 10, 100]
const T_p = 1050
const κ = 0.7903 * 0.057
const β = 0.1883 * 1.0
const η_coeff = 0.00057

# ------------------------------------------------------------
# Data generation
# ------------------------------------------------------------

function generate_random_data(
    dims::Tuple{Int,Int},
    T_e::Float64=240.0;
    style_routing::Symbol=:age_and_reoffenses,
    δ_inc::Float64=0.0,  # incarceration probability upon reoffense
    T_f::Float64=2100.0, # length of follow-up period
    λ_coeff::Float64=0.05,
    style_λ::Symbol=:j_low_only,
    idiosyncrasy::AbstractIdiosyncrasy=GammaIdiosyncrasy(),  # default: Gamma frailty
)
    jₘ, aₘ = dims # number of max rearrests, number of age groups
    V = [(j, a) for j in 0:jₘ for a in 1:aₘ]
    n = length(V)
    e = ones(n)

    data = SortedDict{Symbol,Any}()

    state_to_idx = Dict(v => i for (i, v) in enumerate(V))

    scores_level = [β * j + a_risks[a] for (j, a) in V]
    # matrix representation of scores
    scores_mat = [β * j + a_risks[a] for a in 1:aₘ, j in 0:jₘ]

    η = η_coeff * T_e # not strictly matching the fitting
    # arrivals per episode (rename to beta to avoid confusion with recidivism hazards)
    if style_λ == :uniform
        β_arr = λ_coeff * ones(n) * T_e
    elseif style_λ == :j_low_only
        β_arr = λ_coeff * T_e .* [j <= 2 && a <= 3 ? 1.0 : 0.0 for (j, a) in V]
    elseif style_λ == :j_high_only
        β_arr = λ_coeff * T_e .* [j >= 5 ? 1.0 : 0.0 for (j, a) in V]
    else
        error("Unknown style_λ: $style_λ. Use :uniform or :decreasing.")
    end
    # retention rate
    γ = ones(n) .- T_e / T_p
    σ = ones(n) .- T_e / T_f

    # transition matrix
    if style_routing == :age_and_reoffenses
        Px, Py = _create_transition_matrix_by_age_and_reoffenses(V, T_e, a_size, jₘ, aₘ, state_to_idx)
    elseif style_routing == :random
        Px, Py = _create_random_transition_matrix(V, T_e, a_size, jₘ, aₘ, state_to_idx)
    else
        error("Unknown style_routing: $style_routing. Use :age_and_reoffenses or :age.")
    end

    # Hazard function h(μ) = score + κμ (± treatment effect)
    Fh(μ; p=ones(n)) = [
        (scores_level[1:n] .+ κ * μ); # untreated
        (scores_level[1:n] .+ κ * μ - p) # treated
    ]

    # Recidivism probability function (from idiosyncrasy.jl)
    Fφ = create_Fφ(idiosyncrasy, η, n)

    # Populate data dict
    data[:Δ] = T_e
    data[:aₘ] = aₘ
    data[:β] = β_arr
    data[:e] = e
    data[:Fh] = Fh
    data[:Fφ] = Fφ
    data[:idiosyncrasy] = idiosyncrasy
    data[:η] = η
    data[:jₘ] = jₘ
    data[:n] = n
    data[:Px] = Px
    data[:Py] = Py
    data[:scores_level] = scores_level
    data[:scores_mat] = scores_mat
    data[:γ] = γ
    data[:σ] = σ
    data[:state_to_idx] = state_to_idx
    data[:V] = V
    data[:δ_inc] = δ_inc  # incarceration probability upon reoffense

    # Compute return rates (dispatches on heterogeneity type via data[:idiosyncrasy])
    r = compute_return_rates(data, 0.0)
    data[:r₀] = r[0]
    data[:r₁] = r[1]
    return data
end
