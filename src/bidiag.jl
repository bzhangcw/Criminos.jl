using Random
using LinearAlgebra
using Printf

"""
mutable struct BidiagSys{Tx,Tm}

Bidiagonal system in the paper.

## Fields
- `n::Int`: Number of states.
- `γ::Tx`: Rate retention.
- `λ::Tx`: Rate arrival.
- `q::Tx`: Probability.
- `M::Tm`: Helper matrix.
- `Q::Tm`: Helper matrix.
- `Γ::Tm`: Helper matrix.
- `Γₕ::Tm`: Helper matrix.
- `style::Symbol`: Random or known.

## Constructors
- `BidiagSys(n::Int; style=:rand)`: Constructs a `BidiagSys` object with the specified number of states `n` and style `style`.

"""
Base.@kwdef mutable struct BidiagSys{Tx,Tm}
    n::Int = 0        # n states
    γ::Tx             # rate retention 
    λ::Tx             # rate arrival 
    q::Tx             # probability
    M::Tm             # helper matrix
    Q::Tm             # helper matrix
    Γ::Tm             # helper matrix
    Γₕ::Tm            # helper matrix
    style::Symbol     # random or known

    BidiagSys(n::Int; style=:rand) = (
        this = new{Vector{Float64},Matrix{Float64}}();
        this.n = n;
        this.style = style;
        _construct_callback(n, this; style=style)
    )
end

Base.show(io::IO, ::MIME"text/plain", z::BidiagSys{Tx,Tm}) where {Tx,Tm} =
    print(io, """@bidiagonal system with $(z.n) states  """)

Base.show(stdout, ::MIME"text/plain", z::BidiagSys{Tx,Tm}) where {Tx,Tm} =
    print(stdout, """@bidiagonal system with $(z.n) states  """)

function _construct_callback(n, this::BidiagSys; style=:rand)
    M = zeros(n, n)
    for i in 1:n
        if i > 1
            M[i, i-1] = 1
        end
    end
    this.M = I - M
    if style == :rand
        this.λ = Random.rand(Float64, n)
        this.γ = Random.rand(Float64, n)
        this.q = Random.rand(Float64, n)
    elseif style == :full
        this.γ = ones(n)
        this.λ = zeros(Float64, n)
        this.λ[1] = 0.1
        this.q = zeros(Float64, n)
        this.q[1] = 0.1
    elseif style == :fullrg
        this.γ = ones(n)
        this.λ = Random.rand(Float64, n)
        this.q = Random.rand(Float64, n)
    else
        ErrorException("Unknown style: $style") |> throw
    end
    this.Γ = Diagonal(this.γ)
    this.Q = Diagonal(this.q)
    this.Γₕ = this.M * this.Γ
    return this
end

