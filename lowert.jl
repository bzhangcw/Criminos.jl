using ForwardDiff
using LinearAlgebra
using Random
using Printf
using LaTeXStrings

# using Criminos

n = 6
K = 25
specr(x) = maximum(abs.(eigvals(x)))
# Random.seed!(0)
# generate to julia
# M = np.zeros((n, n))
# for i in range(1, n):
#     M[i, i - 1] = 1
M = zeros(n, n)
for i in 1:n
    if i > 1
        M[i, i-1] = 1
    end
end
M = I - M
γ = Random.rand(Float64, n)
λ = Random.rand(Float64, n)
q = Random.rand(Float64, n)

trace = zeros(5, n, K)

x₀ = Random.rand(Float64, n)
ρ₀ = Random.rand(Float64, n)
y₀ = x₀ .* ρ₀

x = copy(x₀)
ρ = copy(ρ₀)
y = copy(y₀)
σ = 1 ./ (1 .+ exp.(-ρ₀))

# matrices
Q = Diagonal(q)
Γ = Diagonal(γ)
O = zeros(n, n)
Ja = zeros(2n, 2n);
J = zeros(2n, 2n);

# linear mix-in effect
z₀ = [x; y]
z = copy(z₀)
α = 0.8
τ = 0.5 * I
Z = ones(n, n) * 0.1
ε = zeros(K)
Δ = zeros(K)
kl = zeros(K)


function f1(z)
    _x = z[1:n]
    _y = z[n+1:2n]
    _r = _y ./ _x
    _Φ = Γ - M * Γ * Diagonal(_r)
    _x₊ = _Φ * _x + λ
    return _x₊
end

function f2(z)
    _x = z[1:n]
    _y = z[n+1:2n]
    _r = _y ./ _x
    # _Φ = Γ - M * Γ * Diagonal(_r)
    # _x₊ = _Φ * _x + λ
    # _y₊ = _Φ * _x .* _r + Q * λ
    # _A = α * τ * Z * (_x .* _r)
    # # _r₊ = sigmoid.(log.(_y₊ ./ _x₊ ./ (1 .- _y₊ ./ _x₊)) + _A)
    # _r₊ = _y₊ ./ _x₊
    _y₊ = Γ * _y - M * Γ * Diagonal(1 ./ _x) * (_y .^ 2) + Q * λ
    return _y₊
end

function fixpoint(z)
    return [f1(z); f2(z)]
end

function sigmoid(z)
    return 1 ./ (1 .+ exp.(-z))
end

# for k in 1:K
#     global z, z₊
#     # finding the equilibrium
#     z₊ = fixpoint(z)
#     if (z₊ - z) |> norm < 1e-8
#         break
#     end
#     z = z₊
# end


z = copy(z₀)
for k in 1:K
    global x, ρ, y, σ, z, J, Ja
    global f1x, f1x, f1p, f1y, f1s, f2x, f2p, f2y, f2s, f3x, f3p, f3y, f3s, f4x, f4p, f4y, f4s
    global R, Y, X, Φ

    x = z[1:n]
    y = z[n+1:2n]
    ρ = y ./ x

    Φ = Γ - M * Γ * Diagonal(ρ)
    z₁ = fixpoint(z)



    J = ForwardDiff.jacobian(fixpoint, z)
    X = Diagonal(x)
    Y = Diagonal(y)
    Dx = M * Γ * Y^2 * (X^(-2))
    Dy = Γ - 2 * M * Γ * (X^(-1)) * Y
    Ja = [
        Γ (-M*Γ);
        Dx Dy
    ]
    if k == 1
        @printf(
            # "%3s | %31s | %7s | %4s | %4s | %4s | %4s   \n",
            # "%3s | %7s | %7s | %4s | %4s | %4s | %4s   \n",
            "%3s | %4s | %4s | %4s | %4s   \n",
            "k",
            # "jac diff (ad/analytical)",
            # "Δ", "ε", "ρ(Φ)", "|Φ|", "ρ(J)", "|J|",
            "ρ(Φ)", "|Φ|", "ρ(J)", "|J|",
        )
    end
    # Δ[k] = (z - z₊) |> norm
    # Δ[k] = lyapunov_euclidean(z, z₊)
    # ε[k] = lyapunov(z, z₊, Q, λ)
    @printf(
        # "%3d | %.1e %.1e %.1e %.1e | %.1e | %.2f | %.2f | %.2f | %.2f\n",
        # "%3d | %.1e | %.1e | %.2f | %.2f | %.2f | %.2f\n",
        "%3d | %.2f | %.2f | %.2f | %.2f\n",
        k,
        # abs.(J1 - Jₐ[1:n, :]) |> maximum,
        # abs.(J2 - Jₐ[n+1:2n, :]) |> maximum,
        # abs.(J3 - Jₐ[2n+1:3n, :]) |> maximum,
        # abs.(J4 - Jₐ[3n+1:4n, :]) |> maximum,
        # Δ[k],
        # ε[k],
        specr(Φ),
        opnorm(Φ),
        specr(J),
        opnorm(J),
    )

    # proceed k+1
    # x = x₁
    # σ = log.(y ./ (x₁ - y)) + A
    # ρ = 1 ./ (1 .+ exp.(-σ))
    copyto!(z, z₁)

end


############################################################
# check block Gaussian elimination preserve the determinant
# Q = [-f3p θ*I; -f4p -f4y]
# Q1 = [O θ*I(n);
#     (-f4p+1/θ*(-f4y)*f3p) -f4y]

# Aₛ = [
#     (θ*I-f1x) -f1p;
#     O θ*I
# ]
# L = [O O; -R*(I-R)*(-f4x-1/θ*f4y*f3x) -R*(I-R)*(-f4p-1/θ*f4y*f3p)]

# v = (θ^2)^n * det(Aₛ - 1 / θ * L)

# # 
# Lₛ = [
#     (θ*I-f1x) (-f1p);
#     1/θ*R*(I-R)*(-f4x-1/θ*f4y*f3x) θ*I+1/θ*R*(I-R)*(-f4p-1/θ*f4y*f3p)
# ];

# v1 = det(Lₛ) * (θ^2)^n