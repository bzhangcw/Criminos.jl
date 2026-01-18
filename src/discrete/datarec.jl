using DataStructures: SortedDict


# ------------------------------------------------------------
# these are default parameters for the data generation
# ------------------------------------------------------------
a_risks = [0.0 -0.270 -0.456 -0.709 -1.282 -1.808]
a_size = [2, 5, 5, 10, 10, 100]
T_p = 1050
κ = 0.7903 * 0.057
β = 0.1883 * 1.0
hat_k, hat_θ = 3.97, 2.05
η_coeff = 0.00057

function generate_random_data(
    dims::Tuple{Int,Int},
    episode::Float64=240.0;
    λ_coeff::Float64=0.05,
    style_routing::Symbol=:age_and_reoffenses,
    δ_inc::Float64=0.0,  # incarceration probability upon reoffense
    T_f::Float64=2100.0, # length of follow-up period
)
    jₘ, aₘ = dims # number of max rearrests, number of age groups
    V = [(j, a) for j in 0:jₘ for a in 1:aₘ]
    n = length(V)
    e = ones(n)
    state_to_idx = Dict(v => i for (i, v) in enumerate(V))

    scores_level = [β * j + a_risks[a] for (j, a) in V]
    # matrix representation of scores
    scores_mat = [β * j + a_risks[a] for a in 1:aₘ, j in 0:jₘ]

    η = η_coeff * episode # not strictly matching the fitting
    # arrivals per episode (rename to beta to avoid confusion with recidivism hazards)
    β_arr = λ_coeff * ones(n) * episode
    # retention rate
    γ = ones(n) .- episode / T_p
    σ = ones(n) .- episode / T_f

    # transition matrix
    if style_routing == :age_and_reoffenses
        Px, Py = _create_transition_matrix_by_age_and_reoffenses(V, episode, a_size, jₘ, aₘ, state_to_idx)
    elseif style_routing == :random
        Px, Py = _create_random_transition_matrix(V, episode, a_size, jₘ, aₘ, state_to_idx)
    else
        error("Unknown style_routing: $style_routing. Use :age_and_reoffenses or :age.")
    end


    # get functions
    # @note: keep for reference.
    #  the analytic form of the recidivism probability
    #  by gamma distribution fit
    __Fφ(h) = 1 - (hat_θ / (hat_θ + η * exp(h)))^hat_k
    # --- style 1: decrease probability directly ---
    # @note: keep for reference.
    #   style 1: decrease probability directly
    #   Fh(μ; p=ones(n)) = [
    #       (scores_level[1:n] .+ κ * μ); # untreated
    #       (scores_level[1:n] .+ κ * μ) # treated
    #   ]
    #   Fφ(h; p=ones(n)) = begin
    #       ϕ = __Fφ.(h) .* [ones(n); 1.0 .- p]
    #       return Dict(0 => ϕ[1:n], 1 => ϕ[n+1:2n])
    #   end
    # --- style 2: decrease hazard rate ---
    Fh(μ; p=ones(n)) = [
        (scores_level[1:n] .+ κ * μ); # untreated
        (scores_level[1:n] .+ κ * μ - p) # treated
    ]
    Fφ(h; p=ones(n)) = begin
        ϕ = __Fφ.(h)
        return Dict(0 => ϕ[1:n], 1 => ϕ[n+1:2n])
    end

    # summary
    data = SortedDict{Symbol,Any}()
    data[:Δ] = episode
    data[:aₘ] = aₘ
    data[:β] = β_arr
    data[:e] = e
    data[:Fh] = Fh
    data[:Fφ] = Fφ
    data[:hat_k] = hat_k
    data[:hat_θ] = hat_θ
    data[:η] = η
    data[:jₘ] = jₘ
    data[:n] = n
    data[:Px] = Px
    data[:Py] = Py
    # placeholder return rates, should be estimated via `compute_return_rates`!
    data[:scores_level] = scores_level
    data[:scores_mat] = scores_mat
    data[:γ] = γ
    data[:σ] = σ
    data[:state_to_idx] = state_to_idx
    data[:V] = V
    data[:δ_inc] = δ_inc  # incarceration probability upon reoffense

    # update return rates
    r = compute_return_rates_gl16(data, 0.0; p=ones(n))
    r0 = r[0]
    r1 = r[1]
    data[:r₀] = r0
    data[:r₁] = r1
    return data
end

@doc raw"""
    compute_return_rates_gl16(data, μ; p=ones(n)) -> Dict(0=>r0, 1=>r1)
    @note: compute cohort-wise return rates
         r = ϕ_f / (1 - s_f) using GL-16 quadrature
        for the Gamma mixture integral. Does not modify `data`.
"""
function compute_return_rates_gl16(data::Union{SortedDict,Dict}, μ; p=ones(data[:n]))
    n = data[:n]
    η = data[:η]
    k = data[:hat_k]
    θ = data[:hat_θ]
    Ωf = @. -log(max(data[:σ], 1e-9))
    h = data[:Fh](μ; p=p)
    h0 = h[1:n]
    h1 = h[n+1:2n]
    A0 = η .* exp.(h0)
    A1 = η .* exp.(h1)
    ϕ0f, _χ0f, s0f = _gl16_cr_split_vec(A0, Ωf, k, θ)
    ϕ1f, _χ1f, s1f = _gl16_cr_split_vec(A1, Ωf, k, θ)
    r0 = ϕ0f ./ max.(1 .- s0f, 1e-12)
    r1 = ϕ1f ./ max.(1 .- s1f, 1e-12)
    return Dict(0 => r0, 1 => r1)
end