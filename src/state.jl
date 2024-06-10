using Random
using LinearAlgebra

function copy_fields(this, z0)
    for field in fieldnames(typeof(z0))
        setfield!(this, field, copy(getfield(z0, field)))
    end
end

"""
struct MarkovState{R,Tx}

A mutable struct representing the state of a Markov-type Dynamical system

# Fields
- `k::Int`: The iteration number.
- `n::Int`: The number of states.
- `z::Tx`: The fix-point iterate.
- `x::Tx`: The iterate.
- `x₋::Tx`: The previous iterate.
- `ρ::Tx`: The probability.
- `y::Tx`: The recidivists.
- `y₋::Tx`: The previous recidivists.
- `τ::Tx`: The treatment probability.
- `f::Real`: The objective value of the mixed-in function.

# Constructors
- `MarkovState(k, n::Int; z=[Random.rand(Float64, n); Random.rand(Float64, n)], τ=Random.rand(Float64, n), β::Real=1.0)`: Constructs a `MarkovState` object with the given parameters.

"""
Base.@kwdef mutable struct MarkovState{R,Tx}
    k::Int = 0          # iteration
    n::Int = 0          # n states
    z::Tx               # fix-point iterate
    x::Tx               # iterate
    x₋::Tx              # previous iterate
    ρ::Tx               # probability
    y::Tx               # recivists
    y₋::Tx              # previous recivists
    τ::Tx               # treatment probability
    f::Real             # objective value of the mixed-in function
    β::Real             # group size
    θ::Real             # cutoff risk value
    fpr::Real           # false positive rate, if applicable
    # use random initial condition
    MarkovState(
        k, n::Int;
        z=[Random.rand(Float64, n); Random.rand(Float64, n)],
        τ=Random.rand(Float64, n),
        β::Real=1.0
    ) = (
        this = new{Float64,Vector{Float64}}();
        this.k = k;
        this.n = n;
        this.ρ = z[n+1:2n];
        this.x₋ = z[1:n] * β;
        this.y₋ = this.x₋ .* this.ρ;
        this.y = this.x₋ .* this.ρ;
        this.x = z[1:n] * β;
        this.β = β;
        this.z = [this.x; this.ρ];
        this.τ = copy(τ);
        this.f = 1e4;
        this.θ = 0.0;
        this.fpr = 0.0;
        return this
    )
end

Base.show(io::IO, ::MIME"text/plain", z::MarkovState{R,Tx}) where {R,Tx} =
    print(
        io,
        """@current iterate $(z.n) states:
           x: $(round.(z.x;digits=4))
           x₋: $(round.(z.x₋;digits=4))
           ρ: $(round.(z.ρ;digits=4))
           y: $(round.(z.y;digits=4))
           ∑: $(round.(z.y|>sum;digits=4))/$(round.(z.x|>sum;digits=4))
           τ: $(round.(z.τ;digits=4))
        """
    )

Base.copy(z::MarkovState{R,Tx}) where {R,Tx} = begin
    this = MarkovState(z.k, z.n)
    copy_fields(this, z)
    return this
end