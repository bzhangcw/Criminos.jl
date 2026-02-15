# ----------------------------------------------------------------------------
# state struct for discrete-time dynamical system
# ----------------------------------------------------------------------------
Base.@kwdef mutable struct State{R}
    x::Matrix{R}
    y::Matrix{R}
    b::Vector{R}  # untreated returns (b₀)
    tlx::Matrix{R}  # intermediate population after treatment (x̃)
    μ::R
    n::Int
    function State(
        n::Int, x::Matrix{R}, y::Matrix{R}, μ::R;
        b::Vector{R}=zeros(R, n),
        tlx::Matrix{R}=similar(x)
    ) where {R}
        this = new{R}()
        this.n = n
        this.x = x
        this.y = y
        this.b = b
        this.tlx = tlx
        this.μ = μ
        return this
    end
end

function copy_fields(this, z0)
    for field in fieldnames(typeof(z0))
        setfield!(this, field, copy(getfield(z0, field)))
    end
end
Base.copy(z::State) = begin
    this = State(z.n, similar(z.x), similar(z.y), z.μ)
    copy_fields(this, z)
    return this
end
z_diff(z::State, z₊::State) = begin
    return norm(z.x - z₊.x) + norm(z.y - z₊.y)
end

function get_x(z::State)
    return [z.x[:, 1]; z.x[:, 2]; z.x[:, 3]; z.x[:, 4]]
end

function get_y(z::State)
    return [z.y[:, 1]; z.y[:, 2]; z.y[:, 3]; z.y[:, 4]]
end

randz(n; scale=50) = begin
    x = zeros(n, 4)
    x[:, 1] = rand(n) .* scale
    x[:, 2] = zeros(n)
    x[:, 3] = rand(n) .* scale
    x[:, 4] = zeros(n)
    # ---------------------------
    y = zeros(n, 4)
    y[:, 1] = rand(n) .* x[:, 1]
    y[:, 2] = rand(n) .* x[:, 2]
    y[:, 3] = rand(n) .* x[:, 3]
    y[:, 4] = rand(n) .* x[:, 4]
    # ---------------------------
    μ = sum(y) / sum(x)
    return State(n, x, y, μ)
end



# ------------------------------------------------------------
# Visualization utilities for 4n vectors and matrices
# ------------------------------------------------------------
using DataFrames

"""
    visualize_vector(x::Vector, data; name="x") -> DataFrame

Convert a 4n vector to a DataFrame with columns:
- v: cohort tuple (j, a)
- term: "p" (probation) or "f" (follow-up)  
- treated: 0 or 1
- value: the vector value

The 4n vector is ordered as [p0; f0; p1; f1] where each block has n elements.
"""
function visualize_vector(x::AbstractVector, data; name::String="value")
    n = data[:n]
    V = data[:V]
    @assert length(x) == 4n "Expected 4n=$(4n) elements, got $(length(x))"

    rows = []
    labels = [("p", 0), ("f", 0), ("p", 1), ("f", 1)]
    for (block_idx, (term, treated)) in enumerate(labels)
        offset = (block_idx - 1) * n
        for i in 1:n
            push!(rows, (
                v=V[i],
                j=V[i][1],
                a=V[i][2],
                term=term,
                treated=treated,
                Symbol(name) => x[offset+i]
            ))
        end
    end
    return DataFrame(rows)
end

"""
    visualize_state(z::State, data) -> DataFrame

Convert a State struct to a DataFrame with columns for x, y values.
"""
function visualize_state(z::State, data)
    n = data[:n]
    V = data[:V]

    x_vec = get_x(z)
    y_vec = get_y(z)

    rows = []
    labels = [("p", 0), ("f", 0), ("p", 1), ("f", 1)]
    for (block_idx, (term, treated)) in enumerate(labels)
        offset = (block_idx - 1) * n
        for i in 1:n
            push!(rows, (
                v=V[i],
                j=V[i][1],
                a=V[i][2],
                term=term,
                treated=treated,
                x=x_vec[offset+i],
                y=y_vec[offset+i],
            ))
        end
    end
    df = DataFrame(rows)
    df.recid_rate = df.y ./ max.(df.x, 1e-12)
    return df
end

"""
    visualize_matrix(M::AbstractMatrix, data; threshold=1e-8, style=:full) -> DataFrame

Convert a transition matrix to a DataFrame.

# Arguments
- `M`: transition matrix (4n×4n if style=:full, n×n if style=:cohort)
- `data`: data dictionary with :n and :V
- `threshold`: only show entries above this value
- `style`: 
  - `:full` (default): 4n×4n matrix with columns from_v, from_term, from_treated, to_v, to_term, to_treated, value
  - `:cohort`: n×n matrix (e.g., Px, Py) with columns from_v, from_j, from_a, to_v, to_j, to_a, value

Only shows entries above threshold.
"""
function visualize_matrix(M::AbstractMatrix, data; threshold::Real=1e-8, style::Symbol=:full)
    n = data[:n]
    V = data[:V]

    if style == :cohort
        # n×n cohort routing matrix (Px, Py)
        @assert size(M) == (n, n) "Expected (n, n)=$((n, n)), got $(size(M)) for style=:cohort"

        rows = []
        for col in 1:n
            for row in 1:n
                val = M[row, col]
                if abs(val) > threshold
                    from_v = V[col]
                    to_v = V[row]
                    push!(rows, (
                        from_v=from_v,
                        from_j=from_v[1],
                        from_a=from_v[2],
                        to_v=to_v,
                        to_j=to_v[1],
                        to_a=to_v[2],
                        value=val
                    ))
                end
            end
        end
        return DataFrame(rows)

    elseif style == :full
        # 4n×4n full transition matrix
        @assert size(M) == (4n, 4n) "Expected (4n, 4n)=$((4n, 4n)), got $(size(M)) for style=:full"

        labels = [("p", 0), ("f", 0), ("p", 1), ("f", 1)]

        function idx_to_state(idx)
            block = div(idx - 1, n) + 1
            i = mod(idx - 1, n) + 1
            term, treated = labels[block]
            return (v=V[i], j=V[i][1], a=V[i][2], term=term, treated=treated)
        end

        rows = []
        for col in 1:4n
            for row in 1:4n
                val = M[row, col]
                if abs(val) > threshold
                    from_state = idx_to_state(col)
                    to_state = idx_to_state(row)
                    push!(rows, (
                        from_v=from_state.v,
                        from_j=from_state.j,
                        from_a=from_state.a,
                        from_term=from_state.term,
                        from_treated=from_state.treated,
                        to_v=to_state.v,
                        to_j=to_state.j,
                        to_a=to_state.a,
                        to_term=to_state.term,
                        to_treated=to_state.treated,
                        value=val
                    ))
                end
            end
        end
        return DataFrame(rows)

    else
        throw(ArgumentError("style must be :full or :cohort, got $style"))
    end
end

"""
    summarize_matrix(M::AbstractMatrix, data) -> DataFrame

Summarize a 4n×4n transition matrix by block (p0, f0, p1, f1).
Shows row sums and column sums for each block.
"""
function summarize_matrix(M::AbstractMatrix, data)
    n = data[:n]
    @assert size(M) == (4n, 4n) "Expected (4n, 4n)=$((4n, 4n)), got $(size(M))"

    labels = ["p0", "f0", "p1", "f1"]
    block_sums = zeros(4, 4)

    for i in 1:4
        for j in 1:4
            row_range = ((i-1)*n+1):(i*n)
            col_range = ((j-1)*n+1):(j*n)
            block_sums[i, j] = sum(M[row_range, col_range])
        end
    end

    df = DataFrame(block_sums, Symbol.(labels))
    insertcols!(df, 1, :to => labels)
    return df
end