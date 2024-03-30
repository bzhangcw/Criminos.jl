using ForwardDiff
using LinearAlgebra
using Random
using Printf

n = 6
K = 30
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
γ = Random.rand(Float64, n)
λ = Random.rand(Float64, n)
q = Random.rand(Float64, n)

tracex = zeros(n, K)
tracer = zeros(n, K)
x = Random.rand(Float64, n)
ρ = Random.rand(Float64, n)
y = zeros(n)
σ = zeros(n)

# matrices
Q = Diagonal(q)
Γ = Diagonal(γ)
O = zeros(n, n)
J = zeros(4n, 4n);

z = [x ρ y σ]
α = 0.8
τ = 0.5 * I
Z = ones(n, n)
ε = zeros(K)
for k in 1:K
    global x, ρ, y, σ, z, J
    global f1x, f1x, f1p, f1y, f1s, f2x, f2p, f2y, f2s, f3x, f3p, f3y, f3s, f4x, f4p, f4y, f4s
    global R, Y, X, Φ
    σ = log.(ρ ./ (1 .- ρ))
    X = Diagonal(x)
    R = Diagonal(ρ)
    Φ = Γ - (I - M) * Γ * R

    x₁ = Φ * x + λ
    y = Φ * X * ρ + Q * λ
    X₁ = Diagonal(x₁)
    Y = Diagonal(y)

    z = [x; ρ; y; σ]

    tracex[:, k] = x
    tracer[:, k] = ρ

    # treatment
    A = α * τ * Z * X * ρ
    function f1(z)
        _x = z[1:n]
        _r = z[n+1:2n]
        _y = z[2n+1:3n]
        _s = z[3n+1:4n]
        _Φ = Γ - (I - M) * Γ * Diagonal(_r)
        _f1 = _Φ * _x + λ
        return _f1
    end

    function f2(z)
        _x = z[1:n]
        _r = z[n+1:2n]
        _y = z[2n+1:3n]
        _s = z[3n+1:4n]
        _f2 = 1 ./ (1 .+ exp.(-_s))
        return _f2
    end
    function f3(z)
        _x = z[1:n]
        _r = z[n+1:2n]
        _y = z[2n+1:3n]
        _s = z[3n+1:4n]

        _Φ = Γ - (I - M) * Γ * Diagonal(_r)
        _f3 = _Φ * (_x .* _r) + Q * λ
        return _f3
    end
    function f4(z)
        _x = z[1:n]
        _r = z[n+1:2n]
        _y = z[2n+1:3n]
        _s = z[3n+1:4n]
        _Φ = Γ - (I - M) * Γ * Diagonal(_r)
        _f1 = _Φ * _x + λ
        _A = α * τ * Z * (_x .* _r)
        _f4 = log.(_y ./ (_f1 .- _y)) + _A
        return _f4
    end
    # try jacobian analytically
    f1x = Γ - (I - M) * Γ * R
    f1p = -(I - M) * Γ * X
    f1y = O
    f1s = O

    f2x = O
    f2p = O
    f2y = O
    f2s = R * (I - R)

    f3x = Γ * R - 1 * (I - M) * Γ * R .^ 2
    f3p = Γ * X - 2 * (I - M) * Γ * X * R
    f3y = O
    f3s = O

    f4x = Diagonal(-1 ./ (x₁ - y)) * f1x + α * τ * Z * R
    f4p = Diagonal(-1 ./ (x₁ - y)) * f1p + α * τ * Z * X
    f4y = Diagonal(1 ./ y) + Diagonal(1 ./ (x₁ - y))
    f4s = O

    J = [
        f1x f1p f1y f1s;
        f2x f2p f2y f2s;
        f3x f3p f3y f3s;
        f4x f4p f4y f4s
    ]


    # AD mode Jacobian
    J1 = ForwardDiff.jacobian(f1, z)
    J2 = ForwardDiff.jacobian(f2, z)
    J3 = ForwardDiff.jacobian(f3, z)
    J4 = ForwardDiff.jacobian(f4, z)
    if k == 1
        @printf(
            "%3s | %31s | %7s | %4s | %4s | %4s | %4s   \n", "k",
            "jac diff (ad/analytical)", "|Δx|", "ρ(Φ)", "|Φ|", "ρ(J)", "|J|",
        )
    end
    ε[k] = (x₁ - x) |> norm
    @printf(
        "%3d | %.1e %.1e %.1e %.1e | %.1e | %.2f | %.2f | %.2f | %.2f\n",
        k,
        abs.(J1 - J[1:n, :]) |> maximum,
        abs.(J2 - J[n+1:2n, :]) |> maximum,
        abs.(J3 - J[2n+1:3n, :]) |> maximum,
        abs.(J4 - J[3n+1:4n, :]) |> maximum,
        ε[k],
        specr(Φ),
        opnorm(Φ),
        specr(J),
        opnorm(J),
    )

    global θ, C, C0, C1, C2
    θ = 1.5
    C = θ * I(4n) - J
    C0 = [
        θ*I(n)-f1x -f1p O O;
        O θ*I(n) O -R*(I-R)
        -f3x -f3p θ*I(n) O;
        -f4x -f4p -f4y θ*I(n)
    ]

    C1 = [
        θ*I(n)-f1x -f1p O O;
        O θ*I(n) O -R*(I-R)
        -f3x O θ*I(n) O;
        -f4x (-f4p+1/θ*(-f4y)*f3p) -f4y θ*I(n)
    ]

    C2 = [
        θ*I(n)-f1x -f1p O O;
        O θ*I(n) O -R*(I-R)
        O O θ*I(n) O;
        (-f4x-1/θ*f4y*f3x) (-f4p-1/θ*f4y*f3p) O θ*I(n)
    ]
    # proceed k+1
    x = x₁
    σ = log.(y ./ (x₁ - y)) + A
    ρ = 1 ./ (1 .+ exp.(-σ))

end

# using Plots
# plot(1:K, ε, yscale=:log10)

# check block Gaussian elimination preserve the determinant
Q = [-f3p θ*I; -f4p -f4y]
Q1 = [O θ*I(n);
    (-f4p+1/θ*(-f4y)*f3p) -f4y]

Aₛ = [
    (θ*I-f1x) -f1p;
    O θ*I
]
L = [O O; -R*(I-R)*(-f4x-1/θ*f4y*f3x) -R*(I-R)*(-f4p-1/θ*f4y*f3p)]

v = (θ^2)^n * det(Aₛ - 1 / θ * L)

# 
Lₛ = [
    (θ*I-f1x) (-f1p);
    1/θ*R*(I-R)*(-f4x-1/θ*f4y*f3x) θ*I+1/θ*R*(I-R)*(-f4p-1/θ*f4y*f3p)
];

v1 = det(Lₛ) * (θ^2)^n