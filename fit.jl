using ForwardDiff
using LinearAlgebra
using Random
using Printf
using LaTeXStrings
using JuMP
using Criminos
using Plots
using ProgressMeter
using ColorSchemes
using SCS, COPT

using CSV, Tables, DataFrames

ϵₜ = 0.0
expt_inc(τ) = 1 .- exp.(-τ) .+ ϵₜ
expt_dec(τ) = exp.(-2τ) .+ ϵₜ
inv_expt_inc(z) = -log.((1 + ϵₜ) .- z)
inv_expt_dec(z) = -0.5 * log.(z - ϵₜ)

@doc raw"""
old style randomized approach
"""
function generate_random_Ω(N, n, ℜ; kwargs...)
    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec_Ψ]...)
    D, _ = eigs(G)
    D = real(D)
    _c = vcat([_Ψ.Q * _Ψ.λ for _Ψ in vec_Ψ]...)
    if cc.style_correlation == :uppertriangular
        ∇₀ = Matrix(G)
        ∇ₜ = UpperTriangular(cc.style_correlation_seed(N, N))
        Hₜ = Symmetric(cc.style_correlation_seed(N, N))
        Hₜ = Hₜ' * Hₜ
        # this is the lower bound to
        #   guarantee the uniqueness of NE
        H₀ = (
            cc.style_correlation_psd ? G + 1e-3 * I(N) : zeros(N, N)
        )
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    else
        throw(ErrorException("not implemented"))
    end
    if !cc.style_correlation_subp
        Hₜ = blockdiag([sparse(Hₜ[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
        Ω = (∇₀, H₀, ∇ₜ, Hₜ)
    end
    ∇₀, H₀, ∇ₜ, Hₜ, _... = Ω
    ω∇ω(y, τ) = begin
        _t = τ .+ 1
        _T = diagm(τ)
        # Cₜ = diagm(τ) * Hₜ * diagm(τ) + H₀
        # cₜ = (∇ₜ*τ)[:] - (∇₀*_c)[:]
        Cₜ = _T * (Hₜ) * _T + H₀
        cₜ = -(∇ₜ*_t)[:] - (Cₜ*∇₀*_c)[:]
        _w = (
            y' * Cₜ * y / 2 +    # second-order
            y' * cₜ             # first-order
        )
        ∇ = Cₜ * y + cₜ
        return _w, ∇
    end
    return ω∇ω, G, cc.ι
end


function generate_fitting_Ω(N, n, ℜ;
    ρₛ=rand(N),
    τₛ=rand(N),
    Σ₁=nothing,
    Σ₂=nothing,
    yₛ=nothing,
    V=nothing,
    style_mixin_monotonicity=2
)
    # shifting parameter

    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec_Ψ]...)

    T_inc = diagm(expt_inc(τₛ))
    invT_inc = inv(T_inc)
    T_dec = diagm(expt_dec(τₛ))
    invT_dec = inv(T_dec)

    md = Model(optimizer_with_attributes(
        () -> COPT.Optimizer(),
    ))

    # ι must exceed maximum ρₛ
    M₁, M₂, b₁, θ, ι = [], [], [], [], []
    for (id, _Ψ) in enumerate(vec_Ψ)
        _ρ = ρₛ[(id-1)*n+1:id*n]
        push!(ι, max(cc.ι, _ρ...))
        push!(M₁, -_Ψ.Γ + I + diagm(_ρ) * _Ψ.Γₕ)
        push!(M₂, _Ψ.Γ + I + cc.ι .* _Ψ.Γₕ)
        push!(b₁, _ρ .* _Ψ.λ)
        push!(θ, _Ψ.Q * _Ψ.λ)
    end

    @variable(md, yv[1:N] .>= 0)
    @variable(md, _bv[1:N] .>= 0)
    @variable(md, _Bv[1:N, 1:N] .>= 0)
    set_upper_bound.(_bv, 1e4)
    @constraint(md, _Bv[1:N, 1:N] - diagm(_bv) .== 0)
    if style_mixin_monotonicity != 2
        # decreasing
        @error("not implemented")
    else
        # U-shaped
        @constraint(md, (_Bv * Σ₁ * T_inc * ones(N) + _Bv * Σ₂ * T_dec * ones(N)) .== yv)
    end
    for (id, _Ψ) in enumerate(vec_Ψ)
        @constraint(md, M₁[id] * yv[(id-1)*n+1:id*n] .== b₁[id])
    end
    for (id, _Ψ) in enumerate(vec_Ψ)
        # @constraint(md, M₂[id] * yv[(id-1)*n+1:id*n] .<= ι[id] .* _Ψ.λ)
        # @constraint(md, M₂[id] * yv[(id-1)*n+1:id*n] .<= _Ψ.λ)
    end
    @objective(md, Max, tr(_Bv))
    optimize!(md)

    if termination_status(md) != MOI.OPTIMAL
        @warn "Optimizer did not converge"
        @info "" termination_status(md)
        return md
    end
    y = value.(yv)
    x = vcat([1 ./ (1 .- _Ψ.γ) .* (_Ψ.λ - _Ψ.Γₕ * y[(id-1)*n+1:id*n])
              for (id, _Ψ) in enumerate(vec_Ψ)]...)
    _B = value.(_Bv)
    _A = inv(_B)

    # --------------------------------------------------
    # scaling to fit the real population size
    # --------------------------------------------------
    # compute the scaler 
    if yₛ === nothing
        β = ones(N)
    else
        β = (yₛ ./ y)
    end
    @info "scaling" β
    Hₕ(τ) = _A * diagm(expt_inc(τ)) * _A'
    gₕ(τ) = β .* (
        _A * diagm(expt_inc(τ)) * Σ₁ * diagm(expt_inc(τ)) * ones(N) +
        _A * diagm(expt_inc(τ)) * Σ₂ * diagm(expt_dec(τ)) * ones(N)
    )
    yₕ(τ) = Hₕ(τ) \ gₕ(τ)

    # rescale λ
    for _Ψ in vec_Ψ
        _Ψ.λ .= β .* _Ψ.λ
    end
    y₊ = yₕ(τₛ)
    x₊ = vcat([1 ./ (1 .- _Ψ.γ) .* (_Ψ.λ - _Ψ.Γₕ * y₊[(id-1)*n+1:id*n])
               for (id, _Ψ) in enumerate(vec_Ψ)]...)
    @info "" maximum(abs.(y ./ x - ρₛ))
    @info "" maximum(abs.(y₊ ./ x₊ - ρₛ))
    @info "" maximum(abs.(_A * _B - I)) < 1e-5
    @info "" maximum(abs.(y₊ ./ β .- y))
    @info "" (yₕ(τₛ .* 1000)) |> sum

    ω∇ω(y, τ) = begin
        _H = Hₕ(τ) + G
        # _g = -gₕ(τ) - _Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ
        _g = -gₕ(τ) - vcat([_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ for _Ψ in vec_Ψ]...)
        _w = 1 / 2 * y' * _H * y + y' * _g
        ∇ = _H * y + _g
        _w, ∇
    end
    return ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B, md
end



function generate_fitting_Ωxy(N, n, ℜ;
    xₛ=nothing,
    yₛ=nothing,
    τₛ=rand(N),
    Σ₁=nothing,
    Σ₂=nothing,
    V=nothing,
    style_mixin_monotonicity=2
)
    @info "fitting with Ωxy"
    T_inc = diagm(expt_inc(τₛ))
    invT_inc = inv(T_inc)
    T_dec = diagm(expt_dec(τₛ))
    invT_dec = inv(T_dec)

    md = Model(optimizer_with_attributes(
        () -> COPT.ConeOptimizer(),
    ))
    ρₛ = yₛ ./ xₛ
    @variable(md, yv[1:N] .>= 0)
    @variable(md, λ[1:N] .>= 0)
    @variable(md, γ[1:N] .>= 0.1)
    set_upper_bound.(γ[1:N], 0.95)
    # γ = diag([_Ψ.Γ for _Ψ in vec_Ψ]...)
    set_upper_bound.(λ[1:N], xₛ ./ 3)
    M₁, M₂, b₁, θ, ι = [], [], [], [], []
    for (id, _Ψ) in enumerate(vec_Ψ)
        _ρ = ρₛ[(id-1)*n+1:id*n]
        push!(ι, max(cc.ι, _ρ...))
        # _λ = λ[(id-1)*n+1:id*n]
        # push!(M₁, -_Ψ.Γ + I + diagm(_ρ) * _Ψ.Γₕ)
        # push!(M₂, _Ψ.Γ + I + cc.ι .* _Ψ.Γₕ)
        # push!(b₁, _ρ .* _λ)

    end


    @constraint(md, yv .== yₛ)
    @constraint(md, sum(γ) .== 0.781 * N)
    @variable(md, _bv[1:N] .>= 0.0)
    set_upper_bound.(_bv, 1e4)
    @variable(md, bm >= 0)
    @constraint(md, bm .>= _bv)
    if true
        _Bv = diagm(_bv)
    else
        # bounded PSD mode
        @variable(md, _Bv[1:N, 1:N], PSD)
        set_upper_bound.(_Bv, 1e4)
        @constraint(md, _Bv <= 1e3I, PSDCone())
        @constraint(md, _Bv >= 0.1I, PSDCone())
    end

    if style_mixin_monotonicity != 2
        # decreasing
        @error("not implemented")
    else
        # U-shaped
        @constraint(md, (_Bv * Σ₁ * T_inc * ones(N) + _Bv * Σ₂ * T_dec * ones(N)) .== yv)
    end
    for (id, _Ψ) in enumerate(vec_Ψ)
        rg = (id-1)*n+1:id*n
        Γ = diagm(γ[rg])
        Γₕ = _Ψ.M * Γ
        @constraint(
            md, λ[rg] - Γₕ * yₛ[rg] .== (I - Γ) * xₛ[rg]
        )
    end
    @objective(md, Min, bm)
    # @objective(md, Max, tr(_Bv) + 10 * sum(λ))
    optimize!(md)

    if termination_status(md) ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL]
    else
        @warn "Optimizer did not converge"
        @info "" termination_status(md)
        return md
    end

    # we retrieve λ from the optimization problem
    for (id, _Ψ) in enumerate(vec_Ψ)
        rg = (id-1)*n+1:id*n
        # fit lambda
        _λ = value.(λ[rg])
        # fit gamma
        _γ = value.(γ[rg])
        Γ = diagm(_γ)
        Γₕ = _Ψ.M * Γ
        _Ψ.λ .= _λ
        _Ψ.γ .= _γ
        _Ψ.Γ .= Γ
        _Ψ.Γₕ .= Γₕ
    end

    # shifting parameter
    G = blockdiag(
        [sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ)
         for _Ψ in vec_Ψ]...
    )

    y = value.(yv)
    x = vcat([1 ./ (1 .- _Ψ.γ) .* (_Ψ.λ - _Ψ.Γₕ * y[(id-1)*n+1:id*n])
              for (id, _Ψ) in enumerate(vec_Ψ)]...)
    _B = value.(_Bv)
    _A = inv(_B)

    Hₕ(τ) = _A * diagm(expt_inc(τ)) * _A'
    gₕ(τ) = (
        _A * diagm(expt_inc(τ)) * Σ₁ * diagm(expt_inc(τ)) * ones(N) +
        _A * diagm(expt_inc(τ)) * Σ₂ * diagm(expt_dec(τ)) * ones(N)
    )
    yₕ(τ) = Hₕ(τ) \ gₕ(τ)

    @info "" maximum(abs.(y - yₛ))
    @info "" maximum(abs.(x - xₛ))
    @info "" maximum(abs.(_A * _B - I)) < 1e-5
    @info "" maximum(abs.(yₕ(τₛ) .- yₛ) / norm(yₛ, 1))
    @info "" (yₕ(τₛ .* 1000)) |> sum

    ω∇ω(y, τ) = begin
        _H = Hₕ(τ) + G
        _g = -gₕ(τ) - vcat(
            [_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ for _Ψ in vec_Ψ]...
        )
        _w = 1 / 2 * y' * _H * y + y' * _g
        ∇ = _H * y + _g
        _w, ∇
    end

    return ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B, md
end

function tuning(
    n, Σ, Σ₁, Σ₂, V;
    style_mixin_monotonicity=2,
    ρₛ=nothing,
    τₛ=nothing,
    xₛ=nothing,
    yₛ=nothing,
)
    if cc.style_correlation == :uppertriangular
        Σ₁ .= UpperTriangular(Σ₁)
        Σ₂ .= UpperTriangular(Σ₂)
    elseif cc.style_correlation == :diagonal
        Σ₁ .= Diagonal(Σ₁)
        Σ₂ .= Diagonal(Σ₂)
    elseif cc.style_correlation == :lowertriangular
        Σ₁ .= LowerTriangular(Σ₁)
        Σ₂ .= LowerTriangular(Σ₂)
    else
    end
    if !cc.style_correlation_subp
        # remove off-diagonal part
        Σ₁ .= blockdiag([sparse(Σ₁[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
        Σ₂ .= blockdiag([sparse(Σ₂[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:ℜ]...)
    end
    Σ₁ .= round.(Σ₁, digits=3)
    Σ₂ .= round.(Σ₂, digits=3)
    if cc.style_mixin_parameterization == :random
        _args = generate_random_Ω(N, n, ℜ)
    elseif cc.style_mixin_parameterization == :fitting
        _args = generate_fitting_Ω(
            N, n, ℜ;
            ρₛ=ρₛ,
            yₛ=nothing,
            τₛ=τₛ,
            Σ₁=Σ₁,
            Σ₂=Σ₂,
            style_mixin_monotonicity=style_mixin_monotonicity
        )
    elseif cc.style_mixin_parameterization == :fittingxy
        _args = generate_fitting_Ωxy(
            N, n, ℜ;
            xₛ=xₛ,
            yₛ=yₛ,
            τₛ=τₛ,
            Σ₁=Σ₁,
            Σ₂=Σ₂,
            style_mixin_monotonicity=style_mixin_monotonicity
        )
    end
    return _args
end