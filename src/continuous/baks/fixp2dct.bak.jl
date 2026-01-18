# ------------------------------------------------------------
# Continuous-time fixed-point and policy mapping
# Reuse discrete structures; replace φ with CT competing-risk ϕ
# ------------------------------------------------------------
include(joinpath(@__DIR__, "datact.jl"))

column_names = [:p0, :f0, :p1, :f1]

randz_ct(n; scale=50) = begin
    x = zeros(n, 4)
    x[:, 1] = rand(n) .* scale
    x[:, 2] = zeros(n)
    x[:, 3] = rand(n) .* scale
    x[:, 4] = zeros(n)
    y = zeros(n, 4)
    μ = safe_ratio(sum(y), max(sum(x), 1e-9))
    return State(n, x, y, μ)
end

"""
    compute_Psi_ct(data, μ; τ=zeros(n), p=ones(n), compat_mode=get(data,:compat_mode,true))

Return the 4n×4n transition kernel Ψ that maps vec(x) at the start of the interval
to vec(x⁺) at the end, with arrivals excluded (β=0). Ordering is [:p0,:f0,:p1,:f1].
"""
function compute_Psi_ct(data::Dict, μ; τ=zeros(data[:n]), p=ones(data[:n]), compat_mode=get(data, :compat_mode, true))
    n = data[:n]
    PxT = data[:Px]'
    PyT = data[:Py]'
    # local step with no arrivals
    function step_no_arrivals(a1, a2, a3, a4)
        # assignment
        xh1 = (1 .- τ) .* a1
        xh2 = a2
        xh3 = τ .* a1 .+ a3
        xh4 = a4
        # splits
        if compat_mode
            # use φ from mixture but original γ/σ routing
            ϕ = compute_phi_ct(data, μ; p=p)
            y1 = ϕ[0][:p] .* xh1
            y2 = ϕ[0][:f] .* xh2
            y3 = ϕ[1][:p] .* xh3
            y4 = ϕ[1][:f] .* xh4
            γ = data[:γ]
            σ = data[:σ]
            γ_c = 1.0 .- γ
            σ_c = 1.0 .- σ
            x1 = PxT * (γ .* (xh1 .- y1)) + PyT * (γ .* y1) +
                 PyT * (data[:r₀] .* σ_c .* xh2 .+ data[:r₀] .* σ .* y2)
            x2 = PxT * (σ .* (xh2 .- y2)) + PxT * (γ_c .* (xh1 .- y1)) + PyT * (γ_c .* y1)
            x3 = PxT * (γ .* (xh3 .- y3)) + PyT * (γ .* y3) +
                 PyT * (data[:r₁] .* σ_c .* xh4 .+ data[:r₁] .* σ .* y4)
            x4 = PxT * (σ .* (xh4 .- y4)) + PxT * (γ_c .* (xh3 .- y3)) + PyT * (γ_c .* y3)
            return x1, x2, x3, x4
        else
            # mixture-consistent competing-risk splits
            k = get(data, :hat_k, 3.97)
            θ = get(data, :hat_θ, 2.05)
            η = data[:η]
            λ = data[:λ_from_h](μ; p=p)
            h0 = log.(max.(λ[0] ./ max(get(data, :λ0_ct, 1 / data[:Δ]), 1e-12), 1e-12))
            h1 = log.(max.(λ[1] ./ max(get(data, :λ0_ct, 1 / data[:Δ]), 1e-12), 1e-12))
            A0 = η .* exp.(h0)
            A1 = η .* exp.(h1)
            Ωp = data[:ωp] .* data[:Δ]
            Ωf = data[:ωf] .* data[:Δ]
            ϕ0p, χ0p, sp0 = cr_split_vec(A0, Ωp, k, θ)
            ϕ0f, χ0f, sf0 = cr_split_vec(A0, Ωf, k, θ)
            ϕ1p, χ1p, sp1 = cr_split_vec(A1, Ωp, k, θ)
            ϕ1f, χ1f, sf1 = cr_split_vec(A1, Ωf, k, θ)
            y1 = ϕ0p .* xh1
            y2 = ϕ0f .* xh2
            y3 = ϕ1p .* xh3
            y4 = ϕ1f .* xh4
            x1 = PxT * (sp0 .* xh1) + PyT * y1
            x2 = PxT * (sf0 .* xh2) + PxT * (χ0p .* xh1)
            x3 = PxT * (sp1 .* xh3) + PyT * y3
            x4 = PxT * (sf1 .* xh4) + PxT * (χ1p .* xh3)
            return x1, x2, x3, x4
        end
    end
    # build Ψ by columns
    m = 4n
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for j in 1:m
        a1 = zeros(n)
        a2 = zeros(n)
        a3 = zeros(n)
        a4 = zeros(n)
        if j <= n
            a1[j] = 1.0
        elseif j <= 2n
            a2[j-n] = 1.0
        elseif j <= 3n
            a3[j-2n] = 1.0
        else
            a4[j-3n] = 1.0
        end
        x1, x2, x3, x4 = step_no_arrivals(a1, a2, a3, a4)
        # append nonzeros for this column
        for (block, vec) in enumerate((x1, x2, x3, x4))
            base = (block - 1) * n
            for i in 1:n
                v = vec[i]
                if v != 0.0
                    push!(rows, base + i)
                    push!(cols, j)
                    push!(vals, v)
                end
            end
        end
    end
    return sparse(rows, cols, vals, m, m)
end

"""
    psi_block_counts(Ψ, n)
Return a 4×4 matrix of nnz per block for quick inspection of block structure,
with ordering [:p0,:f0,:p1,:f1].
"""
function psi_block_counts(Ψ::SparseArrays.AbstractSparseMatrixCSC, n::Int)
    counts = fill(0, 4, 4)
    for rb in 1:4, cb in 1:4
        r1 = (rb - 1) * n + 1
        r2 = rb * n
        c1 = (cb - 1) * n + 1
        c2 = cb * n
        # take a sparse slice (not a view) to keep type as SparseMatrixCSC
        counts[rb, cb] = nnz(Ψ[r1:r2, c1:c2])
    end
    counts
end

F_ct(x, y, μ, data;
    τ=zeros(data[:n]), p=ones(data[:n])
) = begin
    n = data[:n]
    # arrivals and assignment at the start of the interval (only new arrivals get treatment)
    τ_c = 1.0 .- τ
    xhalf1 = τ_c .* (x[:, 1] .+ data[:β])      # p0
    xhalf2 = x[:, 2]                           # f0
    xhalf3 = τ .* (x[:, 1] .+ data[:β]) .+ x[:, 3]  # p1
    xhalf4 = x[:, 4]                           # f1

    # per-interval CT probabilities under competing risks (Gamma mixture)
    k = get(data, :hat_k, 3.97)
    θ = get(data, :hat_θ, 2.05)
    η = data[:η]
    λ = data[:λ_from_h](μ; p=p)
    h0 = log.(max.(λ[0] ./ max(get(data, :λ0_ct, 1 / data[:Δ]), 1e-12), 1e-12))
    h1 = log.(max.(λ[1] ./ max(get(data, :λ0_ct, 1 / data[:Δ]), 1e-12), 1e-12))
    A0 = η .* exp.(h0)
    A1 = η .* exp.(h1)
    Ωp = data[:ωp] .* data[:Δ]
    Ωf = data[:ωf] .* data[:Δ]
    ϕ0p, χ0p, sp0 = cr_split_vec(A0, Ωp, k, θ)
    ϕ0f, χ0f, sf0 = cr_split_vec(A0, Ωf, k, θ)
    ϕ1p, χ1p, sp1 = cr_split_vec(A1, Ωp, k, θ)
    ϕ1f, χ1f, sf1 = cr_split_vec(A1, Ωf, k, θ)

    y₊1 = ϕ0p .* xhalf1
    y₊2 = ϕ0f .* xhalf2
    y₊3 = ϕ1p .* xhalf3
    y₊4 = ϕ1f .* xhalf4

    PxT = data[:Px]'
    PyT = data[:Py]'

    if get(data, :compat_mode, true)
        # Compatibility: mimic prior discrete mapping (γ, σ applied to non-recidivists), include returns b0/b1
        γ = data[:γ]
        σ = data[:σ]
        γ_c = 1.0 .- γ
        σ_c = 1.0 .- σ
        # untreated blocks
        x₊1 = PxT * (γ .* (xhalf1 .- y₊1)) + PyT * (γ .* y₊1)
        x₊2 = (PxT * (σ .* (xhalf2 .- y₊2))
               + PxT * (γ_c .* (xhalf1 .- y₊1)) + PyT * (γ_c .* y₊1))
        b0 = PyT * (data[:r₀] .* σ_c .* xhalf2 .+ data[:r₀] .* σ .* y₊2)
        x₊1 .+= b0
        # treated blocks
        x₊3 = PxT * (γ .* (xhalf3 .- y₊3)) + PyT * (γ .* y₊3)
        x₊4 = (PxT * (σ .* (xhalf4 .- y₊4))
               + PxT * (γ_c .* (xhalf3 .- y₊3)) + PyT * (γ_c .* y₊3))
        b1 = PyT * (data[:r₁] .* σ_c .* xhalf4 .+ data[:r₁] .* σ .* y₊4)
        x₊3 .+= b1
    else
        # Competing-risk consistent splits (include routing-back of follow-up recidivists)
        x₊1 = PxT * (sp0 .* xhalf1) + PyT * (y₊1 .+ y₊2)
        x₊2 = PxT * (sf0 .* xhalf2) + PxT * (χ0p .* xhalf1)
        x₊3 = PxT * (sp1 .* xhalf3) + PyT * (y₊3 .+ y₊4)
        x₊4 = PxT * (sf1 .* xhalf4) + PxT * (χ1p .* xhalf3)
    end

    return x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4
end

Fc_ct!(z, z₊, data; τ=zeros(data[:n]), p=ones(data[:n])) = begin
    x₊1, x₊2, x₊3, x₊4, y₊1, y₊2, y₊3, y₊4 = F_ct(z.x, z.y, z.μ, data; τ=τ, p=p)
    z₊.x[:, 1] .= x₊1
    z₊.x[:, 2] .= x₊2
    z₊.x[:, 3] .= x₊3
    z₊.x[:, 4] .= x₊4
    z₊.y[:, 1] .= y₊1
    z₊.y[:, 2] .= y₊2
    z₊.y[:, 3] .= y₊3
    z₊.y[:, 4] .= y₊4
    z₊.μ = safe_ratio(sum(z₊.y), sum(z₊.x))
end


