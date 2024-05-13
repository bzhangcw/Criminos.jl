################################################################################
# Decision rules and wrappers
################################################################################



function c_linearquad(
    _τ, args;
    kwargs...
)

end

function decision_identity(z; args=nothing, kwargs...)
    # do nothing
    return z.τ
end

# matching rule
function decision_matching(z; args=nothing, kwargs...)
    _x = z.z[1:z.n]
    _ρ = z.z[z.n+1:2z.n]
    c, _... = args
    τ₊ = _ρ - c
    τ = max.(τ₊, 0)
    return τ
end

# matching rule
function decision_matching_lh(z; args=nothing, kwargs...)
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