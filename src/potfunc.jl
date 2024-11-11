############################################
# Ordinary Potential/Progress Functions
############################################

function Lᵨ(z, z₊; p=1)
    return LinearAlgebra.norm(z.y - z₊.y, p) + 1e-9
end

function Lₓ(z, z₊; p=1)
    return LinearAlgebra.norm(z.x - z₊.x, p) + 1e-9
end

function ∑y(z, z₊; p=1)
    return sum(z.y)
end

function ∑τ(z, z₊; p=1)
    return sum(z.τ)
end

function ∑x(z, z₊; p=1)
    return sum(z.x)
end

function θ(z, z₊; p=1)
    return z.θ
end

function fpr(z, z₊; p=1)
    return z.fpr
end


function ρ(z, z₊; p=1)
    return sum(z.y) / sum(z.x)
end


