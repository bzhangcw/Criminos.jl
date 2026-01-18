# ------------------------------------------------------------
# Continuous-time generator construction (hazard-based)
#
# Builds a continuous-time generator Q (size (4n+1)×(4n+1)) directly from:
#   - term hazards ωᵖ, ωᶠ (derived from γ, σ over interval Δ),
#   - recidivism hazards λ = λ₀ * g * exp(h) with g ~ Gamma(k, θ),
#   - routing matrices Px, Py,
#   - arrivals are EXCLUDED here (Q models internal flows only).
# An augmented state is kept (last row/col) but with zero inflow, so:
#   exp(QΔ) = |Ψ  0|
#             |0  1|
# recovers the discrete-time linear map Ψ for one interval; arrivals b are handled in discrete code.
# 
# The competing-risk probabilities are averaged over the Gamma frailty using GL-16 quadrature.
# ------------------------------------------------------------

"""
    compute_Psi_ct_composed(data, μ; τ=zeros(n), p=ones(n)) -> (Ψ, exit)

Build per-episode Ψ by composing Gamma-averaged competing-risk probabilities 
with routing matrices. Uses GL-16 quadrature to average over frailty g ~ Gamma(k, θ).

## Probation arm (competing risk):
- s_p   = E[exp(-(λ + ωp)Δ)]                        (stay in p, no recid, no f-entry)
- m_pf  = E[ωp/(λ+ωp) (1 - e^{-(λ+ωp)Δ})]           (enter f: term ends first)
- y_p   = E[(1-δ_inc)λ/(λ+ωp) (1 - e^{-(λ+ωp)Δ})]   (recidivism, not incarcerated)
- inc_p = E[δ_inc*λ/(λ+ωp) (1 - e^{-(λ+ωp)Δ})]      (incarcerated from probation)
- Note: s_p + m_pf + y_p + inc_p = 1 (proper competing risk)

## Follow-up arm (competing risk):
- s_f   = E[exp(-(λ + ωf)Δ)]                        (stay in f)
- x_f   = E[ωf/(λ+ωf) (1 - e^{-(λ+ωf)Δ})]           (exit: complete follow-up)
- y_f   = E[(1-δ_inc)λ/(λ+ωf) (1 - e^{-(λ+ωf)Δ})]   (recidivism to p, not incarcerated)
- inc_f = E[δ_inc*λ/(λ+ωf) (1 - e^{-(λ+ωf)Δ})]      (incarcerated from follow-up)
- Note: s_f + x_f + y_f + inc_f = 1 (proper competing risk)

where λ = A * g with A = λ₀ * exp(h), δ_inc = data[:δ_inc] is incarceration probability.

## Contrast with `compute_Psi_discrete`:
The discrete version treats probation events (γ, φ) as INDEPENDENT, not competing.
This CT version is mathematically more accurate—it properly models that recidivism
and term-ending compete, so the probability of each depends on the other's hazard.
The difference is small when Δ is small or when hazards are low.

Routing:
  - survivors and f-entries use Px'; recidivists use Py'.
Policy note: τ only affects arrivals, so no mixing of existing p0 → p1 here.

Returns:
  - Ψ: 4n × 4n transition matrix (NOT column-stochastic due to exits and incarceration)
  - exit: 4n vector of exit probabilities per state (x_f for f-states, inc for all)
"""
function compute_Psi_ct_composed(data, μ; τ=zeros(data[:n]), p=ones(data[:n]))
  n = data[:n]
  Δ = data[:Δ]
  γ = data[:γ]
  σ = data[:σ]
  ωp = @. -log(max(γ, 1e-12)) / Δ
  ωf = @. -log(max(σ, 1e-12)) / Δ

  # Incarceration probability (default 0 if not specified)
  δ_inc = get(data, :δ_inc, 0.0)

  # Gamma frailty parameters
  k = data[:hat_k]
  θ = data[:hat_θ]

  # Cox model: A = λ₀ * exp(h), then λ = A * g where g ~ Gamma(k, θ)
  # η = λ₀ * Δ, so λ₀ = η / Δ
  η = data[:η]
  λ₀ = η / Δ
  h = data[:Fh](μ; p=p)
  h0 = h[1:n]       # untreated scores
  h1 = h[n+1:2n]    # treated scores
  A0 = λ₀ .* exp.(h0)  # A for untreated
  A1 = λ₀ .* exp.(h1)  # A for treated

  # Gamma-averaged probabilities via GL-16 quadrature (now includes incarceration)
  s0p, m0pf, y0p, s0f, x0f, y0f, inc0p, inc0f = _gl16_ct_probs_vec(A0, ωp, ωf, Δ, k, θ; δ_inc=δ_inc)
  s1p, m1pf, y1p, s1f, x1f, y1f, inc1p, inc1f = _gl16_ct_probs_vec(A1, ωp, ωf, Δ, k, θ; δ_inc=δ_inc)

  # routing
  PxT = data[:Px]'
  PyT = data[:Py]'
  D = LinearAlgebra.Diagonal
  Z = spzeros(n, n)

  # blocks
  L11 = PxT * D(s0p) + PyT * D(y0p)   # p0 <- p0
  L12 = PyT * D(y0f)                  # p0 <- f0
  L13 = Z
  L14 = Z

  L21 = PxT * D(m0pf)                 # f0 <- p0
  L22 = PxT * D(s0f)                  # f0 <- f0
  L23 = Z
  L24 = Z

  L31 = Z
  L32 = Z
  L33 = PxT * D(s1p) + PyT * D(y1p)   # p1 <- p1
  L34 = PyT * D(y1f)                  # p1 <- f1

  L41 = Z
  L42 = Z
  L43 = PxT * D(m1pf)                 # f1 <- p1
  L44 = PxT * D(s1f)                  # f1 <- f1

  row1 = hcat(L11, L12, L13, L14)
  row2 = hcat(L21, L22, L23, L24)
  row3 = hcat(L31, L32, L33, L34)
  row4 = hcat(L41, L42, L43, L44)
  Ψ = vcat(row1, row2, row3, row4)

  # Exit probabilities: incarceration for p-states, x_f + incarceration for f-states
  exit = vcat(inc0p, x0f .+ inc0f, inc1p, x1f .+ inc1f)

  return Ψ, exit
end