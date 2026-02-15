

column_names = [:p0, :f0, :p1, :f1]

@doc """
    One step update of the state.

    # Arguments
    - `x`: the current state of the system (n×4 matrix: [p0, f0, p1, f1])
    - `y`: the current recidivism counts
    - `μ`: the current mean recidivate rate
    - `data`: the data of the system
    - `τ`: treatment probability per cohort (n-vector)
    - `p`: treatment effectiveness parameter
    - `b`: untreated returns from previous episode (n-vector, used by :uponentry)

    # Treatment model:
    - `mode`: :new, :existing, :both, or :uponentry
    - `mode` = :new: Treatment applies to NEW ARRIVALS ONLY (β):
        - τ[v] = probability that new arrivals in cohort v receive treatment
        - Existing p0: ALL stay in p0 (no treatment applied to existing)
        - New arrivals β: τ fraction goes to p1, (1-τ) goes to p0
    - `mode` = :existing: Treatment applies to existing p0 ONLY:
        - τ[v] = probability that someone in cohort v receives treatment
        - Existing p0: τ fraction moves to p1, (1-τ) stays in p0
        - New arrivals β: ALL go to p0
    - `mode` = :both: Treatment applies to BOTH existing p0 AND new arrivals
    - `mode` = :uponentry: Treatment upon entry to probation only (Appendix ref):
        - b already part of x[:,1] from previous period's dynamics
        - xhalf1 = x[:,1] + (1-τ)·β - τ·b  (add untreated arrivals, divert treated returns)
        - xhalf3 = x[:,3] + τ·(β + b)       (add treated eligible inflow)
        - Existing p0/p1 continue; only entering individuals subject to treatment

    # Incarceration model:
    When someone reoffends (y), with probability δ_inc they are incarcerated and exit.
    Only (1-δ_inc) fraction of recidivists stay in the system and are routed via Py.
"""
F(x, y, b, μ, data;
    τ=zeros(data[:n]),
    p=ones(data[:n]),
    mode=:existing
) = begin
    n = data[:n]
    h₊ = data[:Fh](μ; p=p)
    φ₊ = data[:Fφ](h₊; p=p)
    # return rates from function call (depends on μ via h₊)
    r = compute_return_rates(data, μ; p=p)
    r0 = r[0]
    r1 = r[1]
    # r0 = data[:r₀]
    # r1 = data[:r₁]

    # incarceration probability (default 0 if not specified)
    δ_inc = get(data, :δ_inc, 0.0)
    stay_frac = 1.0 - δ_inc  # fraction of recidivists who stay in system

    # complementary of these fractions
    γ_c = 1.0 .- data[:γ]
    σ_c = 1.0 .- data[:σ]
    τ_c = 1.0 .- τ

    # Treatment assignment based on mode
    if mode == :new
        # Treatment applies to new arrivals ONLY
        # Existing p0: ALL stay in p0 (untreated)
        # New arrivals: τ fraction goes to p1, (1-τ) goes to p0
        xhalf1 = x[:, 1] + τ_c .* data[:β]           # all existing p0 + untreated new arrivals
        xhalf3 = x[:, 3] + τ .* data[:β]             # existing p1 + treated new arrivals
    elseif mode == :existing
        # Treatment applies to existing ONLY
        xhalf1 = τ_c .* x[:, 1] + data[:β]  # untreated p0 + untreated new arrivals
        xhalf3 = x[:, 3] + τ .* x[:, 1] # existing p1 + treated p0
    elseif mode == :both
        # Treatment applies to BOTH existing p0 (AND) new arrivals
        # Existing p0: τ fraction moves to p1, (1-τ) stays in p0
        # New arrivals: τ fraction goes to p1, (1-τ) goes to p0
        xhalf1 = τ_c .* x[:, 1] + τ_c .* data[:β]    # untreated p0 + untreated new arrivals
        xhalf3 = x[:, 3] + τ .* x[:, 1] + τ .* data[:β]  # existing p1 + treated p0 + treated new arrivals
    elseif mode == :uponentry
        # Treatment upon entry (eq.entry.assign)
        # b (untreated returns) is already in x[:,1] from previous dynamics
        # Divert τ fraction of b to treatment; add (1-τ) fraction of new arrivals
        e = data[:β] + b                               # eligible inflow (eq.entry.inflow)
        xhalf1 = x[:, 1] + τ_c .* data[:β] - τ .* b   # + (1-τ)ξ - τb
        xhalf3 = x[:, 3] + τ .* e                      # + τ(ξ + b)
    else
        throw(ArgumentError("mode must be :new, :existing, :both, or :uponentry, got $mode"))
    end
    # f0: unchanged (treatment only applies to probation)
    xhalf2 = x[:, 2]
    # f1: unchanged
    xhalf4 = x[:, 4]

    # error = sum(xhalf1 + xhalf2 + xhalf3 + xhalf4) - sum(x)
    # @assert error < 1e-4
    # @info "error = $error"
    # @info "" γ_c σ_c τ_c
    # @info "" xhalf1 xhalf2 xhalf3 xhalf4
    # ---------------------------
    # Total recidivism (before incarceration split)
    # p0, f0
    y₊1 = φ₊[0] .* xhalf1
    y₊2 = φ₊[0] .* xhalf2
    # p1, f1
    y₊3 = φ₊[1] .* xhalf3
    y₊4 = φ₊[1] .* xhalf4
    # ---------------------------
    # Recidivists who stay in system (not incarcerated)
    y_stay1 = stay_frac .* y₊1
    y_stay2 = stay_frac .* y₊2
    y_stay3 = stay_frac .* y₊3
    y_stay4 = stay_frac .* y₊4
    # ---------------------------
    # p0
    # Survivors route via Px, recidivists (who stay) route via Py
    x₊1 = data[:Px]' * (data[:γ] .* (xhalf1 - y₊1)) + data[:Py]' * (data[:γ] .* y_stay1)
    # f0
    x₊2 = (data[:Px]' * (data[:σ] .* (xhalf2 - y₊2))
           + data[:Px]' * (γ_c .* (xhalf1 - y₊1)) + data[:Py]' * (γ_c .* y_stay1)
    )
    # Return from follow-up: r already accounts for competing risk, apply stay_frac
    b0 = data[:Py]' * (stay_frac .* r0 .* σ_c .* xhalf2 + stay_frac .* r0 .* data[:σ] .* y₊2)
    x₊1 += b0
    # p1
    x₊3 = data[:Px]' * (data[:γ] .* (xhalf3 - y₊3)) + data[:Py]' * (data[:γ] .* y_stay3)
    # f1
    x₊4 = (data[:Px]' * (data[:σ] .* (xhalf4 - y₊4))
           + data[:Px]' * (γ_c .* (xhalf3 - y₊3)) + data[:Py]' * (γ_c .* y_stay3)
    )
    b1 = data[:Py]' * (stay_frac .* r1 .* σ_c .* xhalf4 + stay_frac .* r1 .* data[:σ] .* y₊4)
    # add back to x₊3 (p1)
    x₊3 += b1
    # x₊3 .+= 0.0
    # ---------------------------
    total = sum(x₊1) + sum(x₊2) + sum(x₊3) + sum(x₊4)
    # @info "rr = $(total - sum(x))"
    # @info "x3 = $(sum(x₊3) - sum(x[:, 3]))"

    # Construct full xhalf matrix (x̃)
    xhalf = hcat(xhalf1, xhalf2, xhalf3, xhalf4)
    
    return x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, b0, xhalf
end

@doc """
    compute_Psi_discrete(data, μ; τ=zeros(n), p=ones(n), mode=:existing) -> (Ψ, b, exit)

Build per-episode transition matrix Ψ (4n×4n), arrival contribution b (4n), and exit vector (4n).

## Treatment policy:
- `mode` = :new: Treatment applies to NEW ARRIVALS ONLY (β):
    - Existing p0: ALL stay in p0 (Ψ matrix has no τ-dependent terms for p0→p1)
    - New arrivals β: τ fraction goes to p1, (1-τ) goes to p0
- `mode` = :existing: Treatment applies to existing p0 ONLY:
    - Existing p0: τ fraction moves to p1 (gets treated), (1-τ) stays in p0
    - New arrivals β: ALL go to p0
- `mode` = :uponentry: Treatment upon entry only (Appendix ref):
    - Existing p0: ALL stay in p0 (same as :new)
    - f0 returns split by τ: L12 scaled by (1-τ), L32 gets τ fraction
    - Arrivals (b vector): added post-dynamics, no recidivism/routing applied

## Incarceration model:
When someone reoffends, with probability δ_inc they are incarcerated and exit the system.
Only (1-δ_inc) fraction of recidivists stay and are routed via Py.

## Probation arm (INDEPENDENT events, not competing risk):
- `γ` = prob of staying in probation (term doesn't end)
- `φ` = prob of recidivism (independent of term ending)
- `φ * (1-δ_inc)` = recidivism that stays in system (routed via Py)
- `φ * δ_inc` = recidivism leading to incarceration (exit)

This simplification is acceptable because in probation, if recidivism happens first,
the term resets anyway; if term ends first, person moves to follow-up. The order
doesn't affect the final routing—only one outcome matters per episode.

## Follow-up arm (competing risk via GL-16):
- Uses `r = φ/(1-s)` computed via Gamma-averaged competing risk
- Proper competing risk because exit vs recidivism are mutually exclusive outcomes
- `r * (1-δ_inc)` = return to probation (not incarcerated)
- `r * δ_inc` = incarcerated from follow-up

## Contrast with `compute_Psi_ct_composed`:
The continuous-time version uses proper competing risk for BOTH arms, which is
mathematically more accurate but gives similar results when Δ is small.

Returns:
- Ψ: 4n×4n transition matrix (NOT column-stochastic due to exits and incarceration)
- b: 4n vector of contributions from new arrivals (β) after treatment split and routing
- exit: 4n exit probability vector (incarceration from p-states, exit + incarceration from f-states)
"""
function compute_Psi_discrete(data, μ; τ=zeros(data[:n]), p=ones(data[:n]), mode=:existing)
    n = data[:n]
    # rates and complements
    γ = data[:γ]
    σ = data[:σ]
    γ_c = 1.0 .- γ
    σ_c = 1.0 .- σ
    # treatment split
    τ_c = 1.0 .- τ
    # incarceration probability (default 0 if not specified)
    δ_inc = get(data, :δ_inc, 0.0)
    stay_frac = 1.0 - δ_inc  # fraction of recidivists who stay in system

    # hazards and return rates at current μ
    h₊ = data[:Fh](μ; p=p)
    φ₊ = data[:Fφ](h₊; p=p)
    φ0 = φ₊[0]
    φ1 = φ₊[1]
    r = compute_return_rates_gl16(data, μ; p=p)
    r0 = r[0]
    r1 = r[1]

    # cached matrices
    Px = data[:Px]
    Py = data[:Py]
    TpX = Px'
    TpY = Py'

    # diagonal helpers (fully qualified to avoid extra imports)
    Dγ = LinearAlgebra.Diagonal(γ)
    Dγc = LinearAlgebra.Diagonal(γ_c)
    Dσ = LinearAlgebra.Diagonal(σ)
    Dσc = LinearAlgebra.Diagonal(σ_c)
    Dτ = LinearAlgebra.Diagonal(τ)
    Dτc = LinearAlgebra.Diagonal(τ_c)
    Dφ0 = LinearAlgebra.Diagonal(φ0)
    Dφ1 = LinearAlgebra.Diagonal(φ1)
    # Recidivists who stay in system (scaled by stay_frac)
    Dφ0_stay = LinearAlgebra.Diagonal(stay_frac .* φ0)
    Dφ1_stay = LinearAlgebra.Diagonal(stay_frac .* φ1)
    Dr0_stay = LinearAlgebra.Diagonal(stay_frac .* r0)
    Dr1_stay = LinearAlgebra.Diagonal(stay_frac .* r1)

    # zero block
    Z = spzeros(n, n)

    # contributions to x₊ from current x
    # Treatment policy depends on mode:
    # - :new: only new arrivals get treated, existing p0 stays in p0
    # - :existing: τ fraction of p0 moves to p1, (1-τ) stays in p0
    #
    # Survivors (1-φ) route via Px, recidivists who stay (φ*stay_frac) route via Py

    if mode == :new
        # mode = :new: Treatment applies to NEW ARRIVALS ONLY
        # Existing p0 ALL stays in p0 (no treatment applied)
        # x₊1 blocks (p0 → p0): ALL p0 stays (no τ factor)
        L11 = TpX * (Dγ * (I - Dφ0)) + TpY * (Dγ * Dφ0_stay)
        L12 = TpY * (Dr0_stay * Dσc + Dr0_stay * (Dσ * Dφ0))
        L13 = Z
        L14 = Z

        # x₊2 blocks (→ f0)
        L21 = TpX * (Dγc * (I - Dφ0)) + TpY * (Dγc * Dφ0_stay)
        L22 = TpX * (Dσ - Dσ * Dφ0)
        L23 = Z
        L24 = Z

        # x₊3 blocks (→ p1): NO p0 → p1 transition (only new arrivals get treated)
        L31 = Z
        L32 = Z
        L33 = TpX * (Dγ - Dγ * Dφ1) + TpY * (Dγ * Dφ1_stay)
        L34 = TpY * (Dr1_stay * Dσc + Dr1_stay * (Dσ * Dφ1))

        # x₊4 blocks (→ f1)
        L41 = Z
        L42 = Z
        L43 = TpX * (Dγc - Dγc * Dφ1) + TpY * (Dγc * Dφ1_stay)
        L44 = TpX * (Dσ - Dσ * Dφ1)

    elseif mode == :existing
        # mode = :existing: Treatment applies to BOTH existing p0 AND new arrivals
        # x₊1 blocks (p0 → p0): only τ_c fraction of p0 stays untreated
        L11 = Dτc * (TpX * (Dγ * (I - Dφ0)) + TpY * (Dγ * Dφ0_stay))
        L12 = TpY * (Dr0_stay * Dσc + Dr0_stay * (Dσ * Dφ0))
        L13 = Z
        L14 = Z

        # x₊2 blocks (→ f0)
        L21 = Dτc * (TpX * (Dγc * (I - Dφ0)) + TpY * (Dγc * Dφ0_stay))
        L22 = TpX * (Dσ - Dσ * Dφ0)
        L23 = Z
        L24 = Z

        # x₊3 blocks (→ p1): includes τ fraction of p0 that gets treated
        # From p0: τ fraction gets treated, then goes through p1 dynamics (using φ1)
        L31 = Dτ * (TpX * (Dγ * (I - Dφ1)) + TpY * (Dγ * Dφ1_stay))
        L32 = Z
        L33 = TpX * (Dγ - Dγ * Dφ1) + TpY * (Dγ * Dφ1_stay)
        L34 = TpY * (Dr1_stay * Dσc + Dr1_stay * (Dσ * Dφ1))

        # x₊4 blocks (→ f1)
        L41 = Dτ * (TpX * (Dγc * (I - Dφ1)) + TpY * (Dγc * Dφ1_stay))
        L42 = Z
        L43 = TpX * (Dγc - Dγc * Dφ1) + TpY * (Dγc * Dφ1_stay)
        L44 = TpX * (Dσ - Dσ * Dφ1)

    elseif mode == :uponentry
        # mode = :uponentry: Treatment upon entry (new arrivals + returns from follow-up)
        # Existing p0 stays in p0 (same as :new for L11, L21)
        # f0 returns split: (1-τ) → p0, τ → p1
        L11 = TpX * (Dγ * (I - Dφ0)) + TpY * (Dγ * Dφ0_stay)
        L12 = Dτc * TpY * (Dr0_stay * Dσc + Dr0_stay * (Dσ * Dφ0))  # (1-τ) fraction of f0 returns → p0
        L13 = Z
        L14 = Z

        # x₊2 blocks (→ f0): same as :new
        L21 = TpX * (Dγc * (I - Dφ0)) + TpY * (Dγc * Dφ0_stay)
        L22 = TpX * (Dσ - Dσ * Dφ0)
        L23 = Z
        L24 = Z

        # x₊3 blocks (→ p1): NO existing p0 → p1, but τ fraction of f0 returns → p1
        L31 = Z
        L32 = Dτ * TpY * (Dr0_stay * Dσc + Dr0_stay * (Dσ * Dφ0))   # τ fraction of f0 returns → p1
        L33 = TpX * (Dγ - Dγ * Dφ1) + TpY * (Dγ * Dφ1_stay)
        L34 = TpY * (Dr1_stay * Dσc + Dr1_stay * (Dσ * Dφ1))

        # x₊4 blocks (→ f1): same as :new
        L41 = Z
        L42 = Z
        L43 = TpX * (Dγc - Dγc * Dφ1) + TpY * (Dγc * Dφ1_stay)
        L44 = TpX * (Dσ - Dσ * Dφ1)

    else
        throw(ArgumentError("mode must be :new, :existing, :both, or :uponentry, got $mode"))
    end

    # assemble 4n x 4n sparse block matrix
    row1 = hcat(L11, L12, L13, L14)
    row2 = hcat(L21, L22, L23, L24)
    row3 = hcat(L31, L32, L33, L34)
    row4 = hcat(L41, L42, L43, L44)
    Ψ = vcat(row1, row2, row3, row4)

    # Affine term: where new arrivals β end up after one episode
    β = data[:β]
    if mode == :uponentry
        # Arrivals added post-dynamics (eq.entry.assign): no recidivism/routing applied
        # Split by τ directly into p0/p1
        b1 = Dτc * β                    # untreated arrivals → p0
        b2 = zeros(n)                   # no direct arrivals to f0
        b3 = Dτ * β                     # treated arrivals → p1
        b4 = zeros(n)                   # no direct arrivals to f1
    else
        # New arrivals enter probation, then go through recidivism/term-ending/routing
        # Untreated arrivals (τ_c fraction of β → p0)
        t0 = Dτc * β                    # untreated arrivals entering p0
        y0 = Dφ0 * t0                   # of those, who recidivate
        y0_stay = stay_frac .* y0       # recidivists who stay (not incarcerated)
        xmy0 = t0 - y0                  # survivors (no recidivism)
        b1 = TpX * (Dγ * xmy0) + TpY * (Dγ * y0_stay)    # stay in p0
        b2 = TpX * (Dγc * xmy0) + TpY * (Dγc * y0_stay)  # term ends → f0
        # Treated arrivals (τ fraction of β → p1)
        t1 = Dτ * β                     # treated arrivals entering p1
        y1 = Dφ1 * t1                   # of those, who recidivate
        y1_stay = stay_frac .* y1       # recidivists who stay (not incarcerated)
        xmy1 = t1 - y1                  # survivors (no recidivism)
        b3 = TpX * (Dγ * xmy1) + TpY * (Dγ * y1_stay)    # stay in p1
        b4 = TpX * (Dγc * xmy1) + TpY * (Dγc * y1_stay)  # term ends → f1
    end
    b = vcat(b1, b2, b3, b4)

    # Exit probabilities per state:
    # - p0: incarceration from recidivism = φ0 * δ_inc (both γ and γ_c cases)
    # - f0: exit from follow-up completion + incarceration from recidivism
    # - p1, f1: same pattern
    inc0_p = δ_inc .* φ0  # incarceration rate from p0
    inc1_p = δ_inc .* φ1  # incarceration rate from p1
    # For follow-up: exit = σ_c * (1-r) + incarceration from recidivism
    # Actually simpler: exit = 1 - (what stays in system)
    # From f0: stays = σ*(1-φ0) + r0_stay*σ_c + r0_stay*σ*φ0 (goes to p0)
    # Exit from f0 = σ_c*(1-r0) + δ_inc*r0*... this is complex
    # Simpler: compute exit as 1 - column_sum(Ψ) for each column
    # But for now, approximate:
    inc0_f = δ_inc .* r0  # incarceration from f0 recidivism
    exit0_f = σ_c .* (1.0 .- r0)  # successful exit from f0
    inc1_f = δ_inc .* r1
    exit1_f = σ_c .* (1.0 .- r1)

    exit = vcat(inc0_p, exit0_f .+ inc0_f, inc1_p, exit1_f .+ inc1_f)

    return Ψ, b, exit
end

Fc!(z, z₊, data; τ=zeros(data[:n]), p=ones(data[:n]), mode=:existing, validate=false) = begin
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, b₀, xhalf = F(z.x, z.y, z.b, z.μ, data; τ=τ, p=p, mode=mode)
    z₊.x[:, 1] .= x₊1
    z₊.x[:, 2] .= x₊2
    z₊.x[:, 3] .= x₊3
    z₊.x[:, 4] .= x₊4
    z₊.y[:, 1] .= y₊1
    z₊.y[:, 2] .= y₊2
    z₊.y[:, 3] .= y₊3
    z₊.y[:, 4] .= y₊4
    z₊.b .= b₀
    z₊.tlx .= xhalf
    z₊.μ = safe_ratio(sum(z₊.y), sum(z₊.x))
    # validate this using Ψ
    if validate
        x_vec = [z.x[:, 1]; z.x[:, 2]; z.x[:, 3]; z.x[:, 4]]
        _Ψ, b, _exit = compute_Psi_discrete(data, z.μ; τ=τ, p=p, mode=mode)
        xΨ_vec = _Ψ * x_vec + b
        n = data[:n]
        xΨ = [xΨ_vec[1:n] xΨ_vec[n+1:2n] xΨ_vec[2n+1:3n] xΨ_vec[3n+1:4n]]
        Xplus = [x₊1 x₊2 x₊3 x₊4]
        abs_err = LinearAlgebra.norm(xΨ - Xplus)
        rel_err = abs_err / max(1e-12, LinearAlgebra.norm(Xplus))
        if rel_err > 1e-4
            @error "Ψ validation failed" abs_err rel_err
        end
    end
end

