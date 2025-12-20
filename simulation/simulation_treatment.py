# survival function fitters
import pandas as pd
from autograd import numpy as np
from functools import wraps
from lifelines import CoxPHFitter
from lifelines.fitters import ParametricRegressionFitter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# ------------------------------------------------------------------------------
# treatment rules
# ------------------------------------------------------------------------------
def treatment_null(dfi, **kwargs):
    dfi["bool_treat"] = 0.0
    return []


def batch_treatment_rule(func):
    """
    Decorator that handles common batch treatment rule logic.
    Applied periodically to select and enroll individuals into treatment.

    - Get qualifying and in-treatment populations
    - Calculate remaining capacity
    - Apply treatment to selected indices
    - Return selected indices

    The wrapped function should return the selected indices from candidates.
    """

    @wraps(func)
    def wrapper(
        dfi,
        capacity=100,
        str_qualifying="",
        str_current_enroll="",
        t=0.0,
        **kwargs,
    ):
        people_qualifying = dfi.query(str_qualifying)
        people_intreatment = dfi.query(str_current_enroll)
        remaining_capacity = capacity - people_intreatment.shape[0]

        if remaining_capacity <= 0:
            return []

        # Get candidates (qualifying people not yet in treatment)
        candidates = people_qualifying.query("bool_treat == 0")

        if candidates.empty:
            return []

        # Let the wrapped function select from candidates
        idx_selected = func(
            candidates=candidates,
            remaining_capacity=remaining_capacity,
            **kwargs,
        )

        if len(idx_selected) == 0:
            return []

        return idx_selected

    return wrapper


@batch_treatment_rule
def treatment_rule_priority(
    candidates, remaining_capacity, key=None, ascending=False, exclude=None, **kwargs
):
    """
    Select individuals by priority (sorted by key).

    @note:
        - if `key` is None, select by highest `score`
        - if `ascending`, choose the top-C-smallest individuals
            else choose the top-C-largest individuals
        - if `exclude` is provided, filter out individuals where exclude(row) is True
    """
    cc = candidates.copy()

    # Apply exclusion filter if provided
    if exclude is not None:
        mask = cc.apply(lambda row: not exclude(row), axis=1)
        cc = cc[mask]

    key = "score" if key is None else key
    to_compute = kwargs.get("to_compute", None)
    if to_compute is not None:
        cc[key] = to_compute(cc)
    return cc.sort_values(key, ascending=ascending).index[:remaining_capacity]


@batch_treatment_rule
def treatment_rule_random(candidates, remaining_capacity, **kwargs):
    """
    Select individuals randomly.
    """
    n = min(remaining_capacity, len(candidates))
    return candidates.sample(n).index


@batch_treatment_rule
def treatment_rule_priority_fluid(
    candidates, remaining_capacity, prob_vector, state_column="state", **kwargs
):
    """
    Priority policy, but may not exhaust the capacity.
        The policy is motivated by the fluid model, where we preset a probability vector for
        prob_vector = {state s: probability p_s}, s \in S
    and each person (in state s) is assigned according to a coin flip with the probability vector.

    Args:
        candidates: DataFrame of qualifying individuals not yet in treatment
        remaining_capacity: maximum number of individuals to select
        prob_vector: dict mapping state tuple -> probability (0 to 1)
        state_column: column name containing the state (default "state")

    Returns:
        indices of selected individuals (may be fewer than remaining_capacity)
    """
    selected = []

    for idx, row in candidates.iterrows():
        if len(selected) >= remaining_capacity:
            break

        # Get state as tuple
        state = row[state_column]
        if hasattr(state, "value"):
            state_key = tuple(state.value)
        else:
            state_key = tuple(state) if hasattr(state, "__iter__") else (state,)

        # Look up probability for this state (default 0 if not in prob_vector)
        prob = prob_vector.get(state_key, 0.0)

        # Coin flip with probability
        if np.random.random() < prob:
            selected.append(idx)

    return selected
