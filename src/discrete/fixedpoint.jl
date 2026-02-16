

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
    q = zeros(n)

    # Treatment assignment based on mode
    if mode == :new
        # Treatment applies to new arrivals ONLY
        # Existing p0: ALL stay in p0 (untreated)
        # New arrivals: τ fraction goes to p1, (1-τ) goes to p0
        q .= data[:e]
        xhalf1 = x[:, 1] + τ_c .* data[:e]           # all existing p0 + untreated new arrivals
        xhalf3 = x[:, 3] + τ .* data[:e]             # existing p1 + treated new arrivals
    # ------------------------------------------------------------
    # @deprecated: will not use them, they allow treatment in mid-of-probation
    # ------------------------------------------------------------
    # elseif mode == :existing
    #     # Treatment applies to existing ONLY
    #     xhalf1 = τ_c .* x[:, 1] + data[:e]  # untreated p0 + untreated new arrivals
    #     xhalf3 = x[:, 3] + τ .* x[:, 1] # existing p1 + treated p0
    # elseif mode == :both
    #     # Treatment applies to BOTH existing p0 (AND) new arrivals
    #     # Existing p0: τ fraction moves to p1, (1-τ) stays in p0
    #     # New arrivals: τ fraction goes to p1, (1-τ) goes to p0
    #     xhalf1 = τ_c .* x[:, 1] + τ_c .* data[:e]    # untreated p0 + untreated new arrivals
    #     xhalf3 = x[:, 3] + τ .* x[:, 1] + τ .* data[:e]  # existing p1 + treated p0 + treated new arrivals
    elseif mode == :uponentry
        # Treatment upon entry (eq.entry.assign)
        # b (untreated returns) is already in x[:,1] from previous dynamics
        # Divert τ fraction of b to treatment; add (1-τ) fraction of new arrivals
        q = data[:e] + b                               # eligible inflow (eq.entry.inflow)
        xhalf1 = x[:, 1] + τ_c .* data[:e] - τ .* b   # + (1-τ)ξ - τb
        xhalf3 = x[:, 3] + τ .* q                      # + τ(ξ + b)
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

    return x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, b0, xhalf, q
end

@doc """
    compute_Psi_discrete(data, μ; τ=zeros(n), p=ones(n)) -> (Ψ, exit)

Build per-episode transition matrix Ψ (4n×4n) mapping tlx (x̃^k) to x^{k+1}, and exit vector (4n).
Ψ captures the linear step from xhalf to x⁺ only (no new-arrival term).

## Treatment policy (embedded in caller; Ψ is independent of mode):
- Ψ describes dynamics from state (p0, f0, p1, f1) after treatment split:
    - Existing p0: ALL stay in p0 (no τ on p0→p0 blocks)
    - No p0→p1 in this step (treatment split is applied before this map).

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
- exit: 4n exit probability vector (incarceration from p-states, exit + incarceration from f-states)
"""
function compute_Psi_discrete(data, μ; τ=zeros(data[:n]), p=ones(data[:n]))
    n = data[:n]
    # rates and complements
    γ = data[:γ]
    σ = data[:σ]
    γ_c = 1.0 .- γ
    σ_c = 1.0 .- σ
    # incarceration probability (default 0 if not specified)
    δ_inc = get(data, :δ_inc, 0.0)
    stay_frac = 1.0 - δ_inc  # fraction of recidivists who stay in system

    # hazards and return rates at current μ
    h₊ = data[:Fh](μ; p=p)
    φ₊ = data[:Fφ](h₊; p=p)
    φ0 = φ₊[0]
    φ1 = φ₊[1]
    r = compute_return_rates(data, μ; p=p)
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
    Dφ0 = LinearAlgebra.Diagonal(φ0)
    Dφ1 = LinearAlgebra.Diagonal(φ1)
    # Recidivists who stay in system (scaled by stay_frac)
    Dφ0_stay = LinearAlgebra.Diagonal(stay_frac .* φ0)
    Dφ1_stay = LinearAlgebra.Diagonal(stay_frac .* φ1)
    Dr0_stay = LinearAlgebra.Diagonal(stay_frac .* r0)
    Dr1_stay = LinearAlgebra.Diagonal(stay_frac .* r1)

    # zero block
    Z = spzeros(n, n)

    # contributions to x₊ from xhalf (tlx): x⁺ = Ψ * xhalf
    # Survivors (1-φ) route via Px, recidivists who stay (φ*stay_frac) route via Py
    # x₊1 blocks (p0 → p0)
    L11 = TpX * (Dγ * (I - Dφ0)) + TpY * (Dγ * Dφ0_stay)
    L12 = TpY * (Dr0_stay * Dσc + Dr0_stay * (Dσ * Dφ0))
    L13 = Z
    L14 = Z

    # x₊2 blocks (→ f0)
    L21 = TpX * (Dγc * (I - Dφ0)) + TpY * (Dγc * Dφ0_stay)
    L22 = TpX * (Dσ - Dσ * Dφ0)
    L23 = Z
    L24 = Z

    # x₊3 blocks (→ p1)
    L31 = Z
    L32 = Z
    L33 = TpX * (Dγ - Dγ * Dφ1) + TpY * (Dγ * Dφ1_stay)
    L34 = TpY * (Dr1_stay * Dσc + Dr1_stay * (Dσ * Dφ1))

    # x₊4 blocks (→ f1)
    L41 = Z
    L42 = Z
    L43 = TpX * (Dγc - Dγc * Dφ1) + TpY * (Dγc * Dφ1_stay)
    L44 = TpX * (Dσ - Dσ * Dφ1)

    # assemble 4n x 4n sparse block matrix
    row1 = hcat(L11, L12, L13, L14)
    row2 = hcat(L21, L22, L23, L24)
    row3 = hcat(L31, L32, L33, L34)
    row4 = hcat(L41, L42, L43, L44)
    Ψ = vcat(row1, row2, row3, row4)

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

    return Ψ, exit
end

Fc!(z, z₊, data; τ=zeros(data[:n]), p=ones(data[:n]), mode=:existing, validate=false) = begin
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4, b₀, xhalf, q₊ = F(z.x, z.y, z.b, z.μ, data; τ=τ, p=p, mode=mode)
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
    z₊.q .= q₊
    z₊.μ = safe_ratio(sum(z₊.y), sum(z₊.x))
    # validate: x⁺ = Ψ * xhalf (tlx)
    if validate
        n = data[:n]
        _Ψ, _exit = compute_Psi_discrete(data, z.μ; τ=τ, p=p)
        xhalf_vec = [xhalf[:, 1]; xhalf[:, 2]; xhalf[:, 3]; xhalf[:, 4]]
        xplus_vec = _Ψ * xhalf_vec
        xΨ = [xplus_vec[1:n] xplus_vec[n+1:2n] xplus_vec[2n+1:3n] xplus_vec[3n+1:4n]]
        Xplus = [x₊1 x₊2 x₊3 x₊4]
        abs_err = LinearAlgebra.norm(xΨ - Xplus)
        rel_err = abs_err / max(1e-12, LinearAlgebra.norm(Xplus))
        if rel_err > 1e-4
            @error "Ψ validation failed" abs_err rel_err
        end
    end
end

