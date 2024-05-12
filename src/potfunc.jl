############################################
# potential functions
############################################

############################################
# ordinary potential functions
############################################
function Lᵨ(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return LinearAlgebra.norm(_r - _r₊, p) + 1e-9
end

function Lₓ(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    _x₊ = z₊.x
    _r₊ = z₊.ρ
    return LinearAlgebra.norm(_x - _x₊, p) + 1e-9
end

function ∑y(z, z₊; p=1)
    _x = z.x
    _r = z.ρ
    return sum(_x .* _r)
end
