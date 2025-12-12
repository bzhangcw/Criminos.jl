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
    dfi["score_with_treat"] = dfi["score"]
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
        effect=0.15,
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

        # Apply treatment to selected
        dfi.loc[idx_selected, "bool_treat"] = 1.0
        dfi.loc[idx_selected, "score_with_treat"] = (
            dfi.loc[idx_selected, "score"]
            - dfi.loc[idx_selected, "bool_treat"] * effect
        )
        dfi.loc[idx_selected, "treat_start"] = t
        dfi.loc[idx_selected, "has_been_treated"] = 1.0
        dfi.loc[idx_selected, "num_treated"] += 1

        return idx_selected

    return wrapper


@batch_treatment_rule
def treatment_rule_priority(
    candidates, remaining_capacity, key=None, ascending=False, **kwargs
):
    """
    Select individuals by priority (sorted by key).

    @note:
        - if `key` is None, select by highest `score`
        - if `ascending`, choose the top-C-smallest individuals
            else choose the top-C-largest individuals
    """
    cc = candidates.copy()
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
