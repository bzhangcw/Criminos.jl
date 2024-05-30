

############################################
# ordinary potential functions
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
