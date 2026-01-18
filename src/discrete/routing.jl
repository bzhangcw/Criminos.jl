# ------------------------------------------------------------
# Routing matrix construction utilities
#
# Routing matrices Px, Py are row-stochastic (rows sum to 1):
#   - Px: routing for survival/term-end events (aging only)
#   - Py: routing for recidivism events (offense count + aging)
#
# Convention: Px[i,j] = P(transition to state j | currently in state i)
# ------------------------------------------------------------

function _create_transition_matrix_by_age_and_reoffenses(V, episode, a_size, jₘ, aₘ, state_to_idx)
    Px = spzeros(length(V), length(V))
    for j in 0:jₘ
        for a in 1:aₘ
            _v = (j, a)
            if a < aₘ
                _m = episode / 365 / a_size[a]
                Px[state_to_idx[_v], state_to_idx[(j, a + 1)]] = _m
                Px[state_to_idx[_v], state_to_idx[_v]] = 1 - _m
            else
                Px[state_to_idx[_v], state_to_idx[_v]] = 1.0
            end
        end
    end
    Py = spzeros(length(V), length(V))
    for j in 0:jₘ
        for a in 1:aₘ
            _v = (j, a)
            j_next = min(j + 1, jₘ)  # cap at max offense count
            if a < aₘ
                _m = episode / 365 / a_size[a]
                Py[state_to_idx[_v], state_to_idx[(j_next, a + 1)]] = _m
                Py[state_to_idx[_v], state_to_idx[(j_next, a)]] = 1 - _m
            else
                Py[state_to_idx[_v], state_to_idx[(j_next, a)]] = 1.0
            end
        end
    end
    return Px, Py
end

"""
    _create_random_transition_matrix(V, episode, a_size, jₘ, aₘ, state_to_idx)

Create purely random row-stochastic routing matrices Px, Py.
Each row sums to 1 (random transitions from each state).
"""
function _create_random_transition_matrix(V, episode, a_size, jₘ, aₘ, state_to_idx)
    n = length(V)
    # Generate random matrices and normalize rows to be stochastic
    Px_raw = rand(n, n)
    Py_raw = rand(n, n)
    # Normalize each row to sum to 1
    Px = sparse(Px_raw ./ sum(Px_raw, dims=2))
    Py = sparse(Py_raw ./ sum(Py_raw, dims=2))
    return Px, Py
end
