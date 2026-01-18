# ------------------------------------------------------------
# Generator-based utilities for continuous-time Markov chains
#
# Functions for:
#   - Computing routing generators Rx, Ry from discrete matrices Px, Py
#   - Matrix logarithm via eigendecomposition and Schur decomposition
#   - Computing Ψ via exp(Q*Δ) with absorbing states
#   - Validation utilities
#
# TODO/REMARK: The full generator approach (embedding Rx, Ry into a single Q)
# is NOT fully justified here. The main difficulty is that event-dependent
# routing (different routing for recidivism vs term-end) cannot be cleanly
# expressed in a single generator. The two-step approach (compute event
# probabilities first, then apply routing) is mathematically equivalent
# and avoids these complications. See sec.agent.tex for details.
# ------------------------------------------------------------

"""
    compute_routing_generators(Px, Py, Δ; method=:schur, ε=1e-24) -> (Rx, Ry)

Compute continuous-time routing generators from discrete-time routing matrices.

Given:
- Px: discrete-time routing matrix for survival/term-end (aging only)
- Py: discrete-time routing matrix for recidivism (offense count + aging)
- Δ: episode length
- method: :log, :eigen, or :schur (default, most stable)
- ε: small value to replace zero eigenvalues

Returns:
- Rx: continuous-time generator such that exp(Rx * Δ) ≈ Px
- Ry: continuous-time generator such that exp(Ry * Δ) ≈ Py

Methods:
- :log uses R = log(P) / Δ directly (fails if P has zeros)
- :eigen uses eigendecomposition: P = V*D*V⁻¹, R = V*log(D)*V⁻¹ / Δ
- :schur uses Schur decomposition: P = Q*T*Q', R = Q*log(T)*Q' / Δ
  (most numerically stable, Q is unitary so Q⁻¹ = Q')

Note: For reducible/sparse matrices, zero eigenvalues are replaced with ε.
The two-step approach (event probs + routing) avoids needing Rx, Ry entirely.

Warning: Py may have zero rows (e.g., max offense count states can't recidivate further).
These rows are filled with identity (stay in place) before computing log.
"""
function compute_routing_generators(Px::AbstractMatrix, Py::AbstractMatrix, Δ::Real;
    method::Symbol=:log, ε::Real=1e-24)
    # Convert to dense if sparse
    Px_dense = Matrix(Px)
    Py_dense = Matrix(Py)

    # Fix zero rows in Py (e.g., max offense count can't increment)
    # Fill with identity (stay in place) so matrix is proper stochastic
    n = size(Py_dense, 1)
    for i in 1:n
        if sum(Py_dense[i, :]) ≈ 0.0
            Py_dense[i, i] = 1.0
        end
    end

    if method == :log
        # Direct matrix logarithm
        if any(Px_dense .== 0) || any(Py_dense .== 0)
            @warn "Px or Py contains zero entries. Matrix logarithm may produce Inf/NaN. " *
                  "Consider using method=:eigen or the two-step approach."
        end
        Rx = real(log(Px_dense)) / Δ
        Ry = real(log(Py_dense)) / Δ

    elseif method == :eigen
        # Eigendecomposition method
        Rx = _matrix_log_eigen(Px_dense, ε) / Δ
        Ry = _matrix_log_eigen(Py_dense, ε) / Δ

    elseif method == :schur
        # Schur decomposition method (most stable)
        Rx = _matrix_log_schur(Px_dense, ε) / Δ
        Ry = _matrix_log_schur(Py_dense, ε) / Δ

    else
        error("Unknown method: $method. Use :log, :eigen, or :schur.")
    end

    # Check for Inf/NaN
    if any(isinf.(Rx)) || any(isnan.(Rx))
        @error "Rx contains Inf or NaN. Matrix logarithm failed for Px."
    end
    if any(isinf.(Ry)) || any(isnan.(Ry))
        @error "Ry contains Inf or NaN. Matrix logarithm failed for Py."
    end

    return Rx, Ry
end

"""
    _matrix_log_eigen(P, ε) -> log(P)

Compute matrix logarithm via eigendecomposition.
Zero eigenvalues are replaced with ε to avoid log(0).

For a general (non-symmetric) matrix: P = V * D * V⁻¹
So: log(P) = V * log(D) * V⁻¹

Note: V' = V⁻¹ only for symmetric/orthogonal matrices!
"""
function _matrix_log_eigen(P::Matrix, ε::Real)
    # Eigendecomposition: P = V * D * V⁻¹
    F = eigen(P)
    λ = F.values
    V = F.vectors

    # Replace zero/negative eigenvalues with ε
    λ_safe = map(x -> abs(x) < ε ? ε : x, λ)

    # Check for negative eigenvalues (would give complex log)
    if any(real.(λ_safe) .< 0)
        @warn "Matrix has negative eigenvalues. Using absolute value for log."
        λ_safe = abs.(λ_safe)
    end

    # log(P) = V * log(D) * V⁻¹  (NOT V' for non-symmetric matrices!)
    log_λ = log.(Complex.(λ_safe))
    V_inv = inv(V)
    log_P = V * Diagonal(log_λ) * V_inv

    return real(log_P)
end

"""
    _matrix_log_schur(P, ε) -> log(P)

Compute matrix logarithm via Schur decomposition (more stable than eigen).

Schur decomposition: P = Q * T * Q'  where Q is unitary, T is upper triangular.
Since Q is unitary: Q⁻¹ = Q', so log(P) = Q * log(T) * Q'

For upper triangular T, log(T) is computed via the Parlett recurrence.
Zero diagonal elements are replaced with ε to avoid log(0).
"""
function _matrix_log_schur(P::Matrix, ε::Real)
    n = size(P, 1)

    # Schur decomposition: P = Q * T * Q'
    F = schur(Complex.(P))
    T = F.T  # upper triangular (complex)
    Q = F.Z  # unitary matrix

    # Compute log(T) for upper triangular matrix using Parlett recurrence
    log_T = _log_triangular(T, ε)

    # log(P) = Q * log(T) * Q'  (Q is unitary, so Q⁻¹ = Q')
    log_P = Q * log_T * Q'

    return real(log_P)
end

"""
    _log_triangular(T, ε) -> log(T)

Compute logarithm of upper triangular matrix using Parlett recurrence.
"""
function _log_triangular(T::Matrix{<:Complex}, ε::Real)
    n = size(T, 1)
    log_T = zeros(Complex{Float64}, n, n)

    # Diagonal: log of eigenvalues (diagonal of T)
    for i in 1:n
        t_ii = T[i, i]
        if abs(t_ii) < ε
            log_T[i, i] = log(Complex(ε))
        elseif real(t_ii) < 0
            log_T[i, i] = log(Complex(abs(t_ii)))  # handle negative
        else
            log_T[i, i] = log(t_ii)
        end
    end

    # Off-diagonal: Parlett recurrence
    # log_T[i,j] = (T[i,j] + sum_{k=i+1}^{j-1} log_T[i,k]*T[k,j] - T[i,k]*log_T[k,j]) / (T[j,j] - T[i,i])
    # when T[i,i] ≠ T[j,j]
    for j in 2:n
        for i in j-1:-1:1
            if abs(T[j, j] - T[i, i]) > ε
                s = T[i, j]
                for k in i+1:j-1
                    s += log_T[i, k] * T[k, j] - T[i, k] * log_T[k, j]
                end
                log_T[i, j] = s / (T[j, j] - T[i, i])
            else
                # When eigenvalues are equal, use series expansion
                # For simplicity, use finite difference approximation
                log_T[i, j] = T[i, j] / T[i, i]
            end
        end
    end

    return log_T
end

"""
    validate_routing_generators(Px, Py, Rx, Ry, Δ; tol=1e-8) -> Bool

Validate that exp(R * Δ) ≈ P for the routing generators.
"""
function validate_routing_generators(Px, Py, Rx, Ry, Δ; tol=1e-8, verbose=true)
    # Check for Inf/NaN first
    if any(isinf.(Rx)) || any(isnan.(Rx)) || any(isinf.(Ry)) || any(isnan.(Ry))
        if verbose
            println("=== Routing Generator Validation ===")
            println("ERROR: Rx or Ry contains Inf/NaN values.")
            println("This happens when Px or Py have zero entries (sparse matrices).")
            println("The matrix logarithm log(P) is undefined when P has zeros.")
            println("\nSOLUTION: Use the two-step approach instead:")
            println("  1. Compute event probabilities via exp(Q*Δ) with per-cohort generator")
            println("  2. Apply routing matrices Px, Py post-hoc based on event type")
            println("  This avoids needing Rx, Ry entirely.")
        end
        return false
    end

    Px_reconstructed = exp(Rx * Δ)
    Py_reconstructed = exp(Ry * Δ)

    err_x = maximum(abs.(Matrix(Px) - Px_reconstructed))
    err_y = maximum(abs.(Matrix(Py) - Py_reconstructed))

    ok = err_x < tol && err_y < tol

    if verbose
        println("=== Routing Generator Validation ===")
        println("max|Px - exp(Rx*Δ)| = $err_x")
        println("max|Py - exp(Ry*Δ)| = $err_y")
        println("Tolerance: $tol")
        println("Valid: $ok")

        # Check generator properties
        println("\nGenerator properties:")
        println("  Rx row sums (should be ~0): ", extrema(sum(Rx, dims=2)))
        println("  Ry row sums (should be ~0): ", extrema(sum(Ry, dims=2)))
        println("  Rx off-diag min (should be ≥0): ", minimum(Rx - Diagonal(diag(Rx))))
        println("  Ry off-diag min (should be ≥0): ", minimum(Ry - Diagonal(diag(Ry))))
    end

    return ok
end

"""
    compute_Psi_ct_from_Q(data, μ; τ=zeros(n), p=ones(n)) -> (Ψ, exit)

Build per-episode Ψ via matrix exponential of generator Q, averaged over Gamma frailty.

## Method:
For each cohort, use 4×4 generators with absorbing states to compute event probabilities:
- Probation: Q_p gives [s_p, y_p, m_pf, inc_p] (stay, recidivate, term ends, incarcerated)
- Follow-up: Q_f gives [s_f, y_f, x_f, inc_f] (stay, recidivate, exit, incarcerated)

Then compose with routing matrices Px, Py based on event type.
Uses GL-16 quadrature to average over frailty g ~ Gamma(k, θ).

## Note on approximation:
This two-step approach (event probs then routing) is an APPROXIMATION that assumes
at most one event per episode. It is exact when:
1. Δ is small (low probability of multiple events)
2. Routing is identity (Px = Py = I)

For the full generator approach with continuous routing, use compute_Psi_ct_full_Q.

Returns:
  - Ψ: 4n × 4n transition matrix (NOT column-stochastic due to exits and incarceration)
  - exit: 4n vector of exit probabilities per state (incarceration + follow-up exit)
"""
function compute_Psi_ct_from_Q(data, μ; τ=zeros(data[:n]), p=ones(data[:n]))
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
    η = data[:η]
    λ₀ = η / Δ
    h = data[:Fh](μ; p=p)
    h0 = h[1:n]
    h1 = h[n+1:2n]
    A0 = λ₀ .* exp.(h0)
    A1 = λ₀ .* exp.(h1)

    # routing matrices
    Px = data[:Px]
    Py = data[:Py]

    # Routing matrices (convert to dense for composition)
    PxT_dense = Matrix(Px')
    PyT_dense = Matrix(Py')

    # Build transition matrix for a given frailty g using 4×4 absorbing-state generators
    function build_Psi_from_Q(g::Real)
        λ0 = A0 .* g
        λ1 = A1 .* g
        D = LinearAlgebra.Diagonal

        # Probation arm probabilities
        s0p = zeros(n)
        y0p = zeros(n)  # recidivism, not incarcerated
        m0pf = zeros(n)
        inc0p = zeros(n)  # incarcerated from probation
        s1p = zeros(n)
        y1p = zeros(n)
        m1pf = zeros(n)
        inc1p = zeros(n)

        # Follow-up arm probabilities
        s0f = zeros(n)
        y0f = zeros(n)  # recidivism, not incarcerated
        x0f = zeros(n)
        inc0f = zeros(n)  # incarcerated from follow-up
        s1f = zeros(n)
        y1f = zeros(n)
        x1f = zeros(n)
        inc1f = zeros(n)

        for i in 1:n
            # Untreated probation: 4×4 generator [stay, recid, term_end, incarcerated]
            # Q_p = | -(λ+ωp)   0   0   0 |
            #       | (1-δ)λ    0   0   0 |
            #       |   ωp      0   0   0 |
            #       |   δλ      0   0   0 |
            Q_p0 = [-(λ0[i] + ωp[i]) 0.0 0.0 0.0;
                (1-δ_inc)*λ0[i] 0.0 0.0 0.0;
                ωp[i] 0.0 0.0 0.0;
                δ_inc*λ0[i] 0.0 0.0 0.0]
            Ψ_p0 = exp(Q_p0 * Δ)
            s0p[i] = Ψ_p0[1, 1]
            y0p[i] = Ψ_p0[2, 1]
            m0pf[i] = Ψ_p0[3, 1]
            inc0p[i] = Ψ_p0[4, 1]

            # Untreated follow-up: 4×4 generator [stay, recid, exit, incarcerated]
            Q_f0 = [-(λ0[i] + ωf[i]) 0.0 0.0 0.0;
                (1-δ_inc)*λ0[i] 0.0 0.0 0.0;
                ωf[i] 0.0 0.0 0.0;
                δ_inc*λ0[i] 0.0 0.0 0.0]
            Ψ_f0 = exp(Q_f0 * Δ)
            s0f[i] = Ψ_f0[1, 1]
            y0f[i] = Ψ_f0[2, 1]
            x0f[i] = Ψ_f0[3, 1]
            inc0f[i] = Ψ_f0[4, 1]

            # Treated probation
            Q_p1 = [-(λ1[i] + ωp[i]) 0.0 0.0 0.0;
                (1-δ_inc)*λ1[i] 0.0 0.0 0.0;
                ωp[i] 0.0 0.0 0.0;
                δ_inc*λ1[i] 0.0 0.0 0.0]
            Ψ_p1 = exp(Q_p1 * Δ)
            s1p[i] = Ψ_p1[1, 1]
            y1p[i] = Ψ_p1[2, 1]
            m1pf[i] = Ψ_p1[3, 1]
            inc1p[i] = Ψ_p1[4, 1]

            # Treated follow-up
            Q_f1 = [-(λ1[i] + ωf[i]) 0.0 0.0 0.0;
                (1-δ_inc)*λ1[i] 0.0 0.0 0.0;
                ωf[i] 0.0 0.0 0.0;
                δ_inc*λ1[i] 0.0 0.0 0.0]
            Ψ_f1 = exp(Q_f1 * Δ)
            s1f[i] = Ψ_f1[1, 1]
            y1f[i] = Ψ_f1[2, 1]
            x1f[i] = Ψ_f1[3, 1]
            inc1f[i] = Ψ_f1[4, 1]
        end

        # Compose with routing
        Z = zeros(n, n)

        L11 = PxT_dense * D(s0p) + PyT_dense * D(y0p)   # p0 <- p0
        L12 = PyT_dense * D(y0f)                         # p0 <- f0
        L21 = PxT_dense * D(m0pf)                        # f0 <- p0
        L22 = PxT_dense * D(s0f)                         # f0 <- f0

        L33 = PxT_dense * D(s1p) + PyT_dense * D(y1p)   # p1 <- p1
        L34 = PyT_dense * D(y1f)                         # p1 <- f1
        L43 = PxT_dense * D(m1pf)                        # f1 <- p1
        L44 = PxT_dense * D(s1f)                         # f1 <- f1

        Ψ = [L11 L12 Z Z;
            L21 L22 Z Z;
            Z Z L33 L34;
            Z Z L43 L44]

        # Exit vector: incarceration for p-states, x_f + incarceration for f-states
        exit_vec = vcat(inc0p, x0f .+ inc0f, inc1p, x1f .+ inc1f)

        return Ψ, exit_vec
    end

    # Gamma-average via GL-16 quadrature
    g_max = _gamma_quantile_approx(0.9999, k, θ)

    Ψ_sum = zeros(4n, 4n)
    exit_sum = zeros(4n)

    for i in 1:8
        t1 = 0.5 * (1 + _GL16_x[i])
        t2 = 0.5 * (1 - _GL16_x[i])
        g1 = t1 * g_max
        g2 = t2 * g_max
        w = _GL16_w[i]

        pdf1 = _gamma_pdf(g1, k, θ) * g_max * 0.5
        pdf2 = _gamma_pdf(g2, k, θ) * g_max * 0.5

        Ψ1, exit1 = build_Psi_from_Q(g1)
        Ψ2, exit2 = build_Psi_from_Q(g2)

        Ψ_sum .+= w * pdf1 * Ψ1
        Ψ_sum .+= w * pdf2 * Ψ2
        exit_sum .+= w * pdf1 * exit1
        exit_sum .+= w * pdf2 * exit2
    end

    return Ψ_sum, exit_sum
end

"""
    validate_cr_probs_vs_expQ(λ, ωp, ωf, Δ; verbose=true) -> Bool

Validate that the closed-form competing-risk probabilities match exp(Q*Δ).

For a single cohort with hazards λ (recidivism), ωp (prob term-end), ωf (follow-up term-end):

Closed-form probabilities (per-arm, assuming one interval):
- s_p  = exp(-(λ + ωp)Δ)                    (stay in p)
- y_p  = λ/(λ+ωp) * (1 - exp(-(λ+ωp)Δ))     (recidivate in p → stays in p)
- m_pf = ωp/(λ+ωp) * (1 - exp(-(λ+ωp)Δ))    (p → f, term ends)
- s_f  = exp(-(λ + ωf)Δ)                    (stay in f)
- y_f  = λ/(λ+ωf) * (1 - exp(-(λ+ωf)Δ))     (recidivate from f → goes to p)
- x_f  = ωf/(λ+ωf) * (1 - exp(-(λ+ωf)Δ))    (exit from f)

We use SEPARATE generators for p-arm and f-arm with absorbing states.

Returns true if all probabilities match within tolerance.
"""
function validate_cr_probs_vs_expQ(λ::Real, ωp::Real, ωf::Real, Δ::Real; verbose=true, tol=1e-10)
    # Closed-form competing-risk probabilities
    denom_p = max(λ + ωp, 1e-12)
    denom_f = max(λ + ωf, 1e-12)

    s_p = exp(-denom_p * Δ)
    y_p = (λ / denom_p) * (1 - exp(-denom_p * Δ))
    m_pf = (ωp / denom_p) * (1 - exp(-denom_p * Δ))

    s_f = exp(-denom_f * Δ)
    y_f = (λ / denom_f) * (1 - exp(-denom_f * Δ))
    x_f = (ωf / denom_f) * (1 - exp(-denom_f * Δ))

    # Verify they sum to 1
    sum_p = s_p + y_p + m_pf
    sum_f = s_f + y_f + x_f

    # For probation arm: 3 states = [p (transient), recid (absorbing), term_end (absorbing)]
    Q_p = [-(λ + ωp) 0.0 0.0;
        λ 0.0 0.0;
        ωp 0.0 0.0]
    Ψ_p = exp(Q_p * Δ)
    Ψ_stay_p = Ψ_p[1, 1]
    Ψ_recid_p = Ψ_p[2, 1]
    Ψ_term_p = Ψ_p[3, 1]

    # For follow-up arm: 3 states = [f (transient), recid (absorbing), exit (absorbing)]
    Q_f = [-(λ + ωf) 0.0 0.0;
        λ 0.0 0.0;
        ωf 0.0 0.0]
    Ψ_f = exp(Q_f * Δ)
    Ψ_stay_f = Ψ_f[1, 1]
    Ψ_recid_f = Ψ_f[2, 1]
    Ψ_exit_f = Ψ_f[3, 1]

    # Check matches
    err_sp = abs(Ψ_stay_p - s_p)
    err_yp = abs(Ψ_recid_p - y_p)
    err_mp = abs(Ψ_term_p - m_pf)
    err_sf = abs(Ψ_stay_f - s_f)
    err_yf = abs(Ψ_recid_f - y_f)
    err_xf = abs(Ψ_exit_f - x_f)

    all_ok = all([err_sp, err_yp, err_mp, err_sf, err_yf, err_xf] .< tol)

    if verbose
        println("=== Competing-Risk Validation ===")
        println("λ=$λ, ωp=$ωp, ωf=$ωf, Δ=$Δ")
        println()
        println("Closed-form probabilities:")
        println("  s_p=$s_p, y_p=$y_p, m_pf=$m_pf (sum=$(sum_p))")
        println("  s_f=$s_f, y_f=$y_f, x_f=$x_f (sum=$(sum_f))")
        println()
        println("From exp(Q*Δ) with absorbing states:")
        println("  Probation arm (3×3 with absorbing recid/term):")
        println("    Ψ[stay,p]=$(Ψ_stay_p) (expect s_p=$s_p, err=$err_sp)")
        println("    Ψ[recid,p]=$(Ψ_recid_p) (expect y_p=$y_p, err=$err_yp)")
        println("    Ψ[term,p]=$(Ψ_term_p) (expect m_pf=$m_pf, err=$err_mp)")
        println()
        println("  Follow-up arm (3×3 with absorbing recid/exit):")
        println("    Ψ[stay,f]=$(Ψ_stay_f) (expect s_f=$s_f, err=$err_sf)")
        println("    Ψ[recid,f]=$(Ψ_recid_f) (expect y_f=$y_f, err=$err_yf)")
        println("    Ψ[exit,f]=$(Ψ_exit_f) (expect x_f=$x_f, err=$err_xf)")
        println()
        println("All OK: $all_ok")
    end

    return all_ok
end
