@doc """
fitting mixed-in effect by optimization 
 over the equilibria depicted by ρₛ (the offending rate)
"""
function generate_fitting_ρ(N, n, ℜ;
    ρₛ=rand(N),
    τₛ=rand(N),
    Σ₁=nothing,
    Σ₂=nothing,
    yₛ=nothing,
    V=nothing,
    style_mixin_monotonicity=2,
    vec_Ψ=nothing,
    cc=nothing,
)
    @warn "fitting with offending rate, please change to the method using (x,y)"
    throw(ErrorException("change to the method using (x,y)"))
    # shifting parameter
    G = blockdiag([sparse(_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.Γₕ) for _Ψ in vec_Ψ]...)

    T_inc = diagm(expt_inc(τₛ))
    T_dec = diagm(expt_dec(τₛ))

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

    ω∇ω(y, τ, x) = begin
        _H = Hₕ(τ) + G
        # _g = -gₕ(τ) - _Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ
        _g = -gₕ(τ) - vcat([_Ψ.Γₕ' * inv(I - _Ψ.Γ) * _Ψ.λ for _Ψ in vec_Ψ]...)
        _w = 1 / 2 * y' * _H * y + y' * _g
        ∇ = _H * y + _g
        _w, ∇
    end
    return ω∇ω, G, ι, y, x, gₕ, Hₕ, yₕ, _A, _B, md
end
