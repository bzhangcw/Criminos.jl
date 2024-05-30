################################################################################
# Decision rules and wrappers
################################################################################
# the decision is made as a centralized party

function decision_identity(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    # do nothing
end

# matching rule
function decision_matching(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    _x = z.z[1:z.n]
    _ρ = z.z[z.n+1:2z.n]
    c, _... = args
    τ₊ = _ρ - c
    τ = max.(τ₊, 0)
    return τ
end

# matching rule
function decision_matching_lh(
    vector_ms::Vector{MarkovState{R,Tx}},
    vec_Ψ::Vector{BidiagSys{Tx,Tm}};
    args=nothing,
    kwargs...
) where {R,Tx,Tm}
    _x = z.z[1:z.n]
    _ρ = z.z[z.n+1:2z.n]
    c, _... = args
    τ₊ = _ρ - c
    xₙ = Int(z.n // 2)
    τ = similar(τ₊)
    τ[1:xₙ] .= max(0, sum(τ₊[1:xₙ]) / xₙ)
    τ[xₙ+1:end] .= max(0, sum(τ₊[xₙ+1:end]) / (z.n - xₙ))
    return τ
end