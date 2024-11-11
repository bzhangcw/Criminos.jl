################################################################################
# Fitting Mixed-In Effect by Optimization 
# - equilibria based fitting
################################################################################

using Random, ForwardDiff, LinearAlgebra
using Printf, LaTeXStrings
using JuMP, COPT, MosekTools
import MathOptInterface as MOI
ϵₜ = 0.0

# increase faster than decrease
upscaler = 2.0
downscaler = 4.0
expt_inc(τ) = 1 .- exp.(-upscaler * τ) .+ ϵₜ
expt_dec(τ) = exp.(-downscaler * τ) .+ ϵₜ
inv_expt_inc(z) = -log.((1 + ϵₜ) .- z) ./ upscaler
inv_expt_dec(z) = -log.(z - ϵₜ) ./ downscaler
# --------------------------------------------------
# the exponential cone representation
# --------------------------------------------------
# to do this, you should add an auxillary variable: r
# r - ϵₜ ≥ exp(-τ) => [-τ, 1, r - ϵₜ] in EC
expt_inc_cone!(τ, r, model) = @constraint(
    model, [-downscaler * τ, 1, r - ϵₜ] in MOI.ExponentialCone()
)

mutable struct MixedInParams{Tm}
    Σ₁::Tm
    Σ₂::Tm
    B::Tm
    A::Tm
end

gety(mipar::MixedInParams, θ₁, θ₂) = mipar.B * mipar.Σ₁ * θ₁ + mipar.B * mipar.Σ₂ * θ₂


function tuning(
    n, Σ, Σ₁, Σ₂, V;
    style_mixin_monotonicity=2,
    ρₛ=nothing,
    τₛ=nothing,
    xₛ=nothing,
    yₛ=nothing,
    cc=nothing,
    vec_Ψ=nothing,
    bool_H_diag=true,
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
        @info "keep correlation matrix as it is"
    end
    if !cc.style_correlation_subp
        # remove off-diagonal part
        Σ₁ .= blockdiag([sparse(Σ₁[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:cc.R]...)
        Σ₂ .= blockdiag([sparse(Σ₂[(id-1)*n+1:id*n, (id-1)*n+1:id*n]) for id in 1:cc.R]...)
    end
    Σ₁ .= round.(Σ₁, digits=3)
    Σ₂ .= round.(Σ₂, digits=3)
    if cc.style_mixin_parameterization == :random
        _args = generate_random(
            cc.N, cc.n, cc.R;
            vec_Ψ=vec_Ψ
        )
    elseif cc.style_mixin_parameterization == :fitting
        _args = generate_fitting_ρ(
            cc.N, cc.n, cc.R;
            ρₛ=ρₛ,
            yₛ=nothing,
            τₛ=τₛ,
            Σ₁=Σ₁,
            Σ₂=Σ₂,
            style_mixin_monotonicity=style_mixin_monotonicity,
            cc=cc,
            vec_Ψ=vec_Ψ,
        )
    elseif cc.style_mixin_parameterization == :fittingxy
        _args = generate_fitting_xy(
            cc.N, cc.n, cc.R;
            xₛ=xₛ,
            yₛ=yₛ,
            τₛ=τₛ,
            Σ₁=Σ₁,
            Σ₂=Σ₂,
            style_mixin_monotonicity=style_mixin_monotonicity,
            cc=cc,
            vec_Ψ=vec_Ψ,
            bool_H_diag=bool_H_diag
        )
    end
    return _args
end

@doc raw"""
generate a random mixed-in effect
"""
function generate_random(
    N, n, ℜ;
    vec_Ψ=nothing,
    kwargs...
)
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

@doc """
fitting mixed-in effect by optimization 
 over the equilibria depicted by xₛ, yₛ (the population and the offenders)
"""
function generate_fitting_xy(
    N, n, ℜ;
    # -------------------------------------
    # equilibria information
    xₛ=nothing, # must provide
    yₛ=nothing, # must provide
    τₛ=rand(N), # the static τ at default
    λ=nothing, # optional 
    γ=nothing, # optional 
    # -------------------------------------
    cc=nothing,
    vec_Ψ=nothing,
    Σ₁=nothing,
    Σ₂=nothing,
    style_mixin_monotonicity=2,
    # the functional form
    style_function=1,
    # the lower bound PSD matrix; 
    # strongly convex is desired
    bool_H_diag=true,
    ϵₕ=5e-1,
    # very likely to be infeasible
    bool_γ_same=false,
)
    @info """
    fitting at equilibria by (x, y)
    """
    θ₁ = expt_inc(τₛ)
    θ₂ = expt_dec(τₛ)

    md = Model(optimizer_with_attributes(
        () -> COPT.ConeOptimizer(),
    ))
    @variable(md, yv[1:N] .>= 0)

    # --------------------------------------------------
    # if there is no external λ and γ
    # set these to be optimized
    # set bounds to γ and λ: these are myoptic
    if isnothing(λ)
        @variable(md, λ[1:N] .>= 0)
        set_upper_bound.(λ[1:N], xₛ ./ 2)
    end
    if isnothing(γ)
        @variable(md, γ[1:N] .>= 0.02)
        set_upper_bound.(γ[1:N], 0.8)
        begin
            @variable(md, γ₀)
            bool_γ_same && @constraint(md, γ[1:N] .== γ₀)
        end
    end
    # --------------------------------------------------

    @constraint(md, yv .== yₛ)
    @variable(md, _bv[1:N] .>= ϵₕ)
    set_upper_bound.(_bv, 1e3)
    if bool_H_diag
        _Bv = diagm(_bv)
    else
        # bounded full rank PSD mode
        @variable(md, _Bv[1:N, 1:N], PSD)
        set_upper_bound.(_Bv, 1e4)
        @constraint(md, _Bv <= 1e3I, PSDCone())
        @constraint(md, _Bv >= 0.1I, PSDCone())
    end

    if style_mixin_monotonicity != 2
        @error("not implemented")
    else
        # U-shaped: increasing and decreasing
        # - increasing controlled by Σ₁
        # - decreasing controlled by Σ₂
        @constraint(md, (_Bv * Σ₁ * θ₁ + _Bv * Σ₂ * θ₂) .== yv)
    end
    for (id, _Ψ) in enumerate(vec_Ψ)
        rg = (id-1)*n+1:id*n
        Γ = diagm(γ[rg])
        Γₕ = _Ψ.M * Γ
        @constraint(
            md, λ[rg] - Γₕ * yₛ[rg] .== (I - Γ) * xₛ[rg]
        )
    end
    @objective(md, Max, sum(_bv))
    optimize!(md)

    if termination_status(md) ∈ [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL, MOI.SLOW_PROGRESS]
    else
        @warn "Optimizer did not converge"
        write_to_file(md, "model.lp")
        @info "" termination_status(md)
        return md
    end

    # we retrieve λ from the optimization problem
    for (id, _Ψ) in enumerate(vec_Ψ)
        rg = (id-1)*n+1:id*n
        _λ = value.(λ[rg])
        _γ = value.(γ[rg])
        _Γ = diagm(_γ)
        _Γₕ = _Ψ.M * _Γ
        _Ψ.λ .= _λ
        _Ψ.γ .= _γ
        _Ψ.Γ .= _Γ
        _Ψ.Γₕ .= _Γₕ
    end

    # shifting parameter
    G = blockdiag(
        [sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ)
         for _Ψ in vec_Ψ]...
    )

    # --------------------------------------------------
    # get values and produce mixed-in effect params
    # --------------------------------------------------
    y = value.(yv)
    x = vcat([1 ./ (1 .- _Ψ.γ) .* (_Ψ.λ - _Ψ.Γₕ * y[(id-1)*n+1:id*n])
              for (id, _Ψ) in enumerate(vec_Ψ)]...)
    _B = value.(_Bv)
    _A = inv(_B)

    Hₕ(τ) = style_function == 1 ? _A : _A * diagm(expt_inc(τ)) * _A'
    gₕ(τ) = style_function == 1 ? (
        Σ₁ * expt_inc(τ) +
        Σ₂ * expt_dec(τ)
    ) : (
        _A * diagm(expt_inc(τ)) * Σ₁ * expt_inc(τ) +
        _A * diagm(expt_inc(τ)) * Σ₂ * expt_dec(τ)
    )
    (style_function != 1) && @warn("use old style, which is hard to explain...")
    yₕ(τ) = Hₕ(τ) \ gₕ(τ)

    # get mixed-in effect parameters struct
    mipar = MixedInParams(Σ₁, Σ₂, _B, _A)

    @info "" maximum(abs.(y - yₛ))
    @info "" maximum(abs.(x - xₛ))
    @info "" maximum(abs.(_A * _B - I)) < 1e-5
    @info "" maximum(abs.(yₕ(τₛ) .- yₛ) / norm(yₛ, 1))
    @info "" maximum(abs.(gety(mipar, expt_inc(τₛ), expt_dec(τₛ)) .- yₛ) / norm(yₛ, 1))
    τᵢ = similar(τₛ)
    τᵢ .= 1.0
    τ₀ = similar(τₛ)
    τ₀ .= 0.01
    @info "" (yₛ |> sum) sum(yₕ(τᵢ)) sum(yₕ(τ₀))

    ω∇ω(y, τ) = begin
        _H = Hₕ(τ) + G
        _g = -gₕ(τ) - vcat(
            [_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ for _Ψ in vec_Ψ]...
        )
        _w = 1 / 2 * y' * _H * y + y' * _g
        ∇ = _H * y + _g
        _w, ∇
    end

    return mipar, ω∇ω, G, y, x, gₕ, Hₕ, yₕ, _A, _B, md
end


include("fit.depre.jl")