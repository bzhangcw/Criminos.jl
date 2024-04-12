using Random
using LinearAlgebra

Base.@kwdef mutable struct MarkovState{R,Tx}
    k::Int = 0          # iteration
    n::Int = 0          # n states
    z::Tx               # fix-point iterate
    x::Tx               # iterate
    ρ::Tx               # probability
    y::Tx               # auxiliary variable
    τ::Tx               # treatment probability

    # use known initial condition
    MarkovState(k, z0::Vector{Float64}, τ::Vector{Float64}) = (
        this = new{Float64,Vector{Float64}}();
        this.k = k;
        this.z = copy(z0);
        this.n = n = length(this.z) ÷ 2;
        this.x = this.z[1:n];
        this.ρ = this.z[n+1:2n];
        this.y = this.x .* this.ρ;
        this.τ = τ;
        return this
    )

    # use random initial condition
    MarkovState(k, n::Int) = (
        this = new{Float64,Vector{Float64}}();
        this.k = k;
        this.n = n;
        this.x = Random.rand(Float64, n);
        this.ρ = Random.rand(Float64, n);
        this.y = this.x .* this.ρ;
        this.z = [this.x; this.ρ];
        this.τ = rand(Float64, n) / 10;
        return this
    )
end

Base.show(io::IO, ::MIME"text/plain", z::MarkovState{R,Tx}) where {R,Tx} =
    print(
        io,
        """@current iterate $(z.n) states:
           x: $(round.(z.x;digits=2))
           ρ: $(round.(z.ρ;digits=2))
        """
    )

Base.copy(z::MarkovState{R,Tx}) where {R,Tx} =
    MarkovState(z.k, z.z, z.τ)