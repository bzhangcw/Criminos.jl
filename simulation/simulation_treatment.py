# survival function fitters
import pandas as pd
from autograd import numpy as np
from lifelines import CoxPHFitter
from lifelines.fitters import ParametricRegressionFitter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# ------------------------------------------------------------------------------
# treatment rules
# ------------------------------------------------------------------------------
def treatment_null(dfi, **kwargs):
    dfi["treat"] = 0.0
    dfi["score_with_treat"] = dfi["score"]


def treatment_rule_priority(
    dfi, key=None, capacity=100, ascending=False, effect=0.05, **kwargs
):
    """
    Select the individuals with some priority
        defined by `key` up to the `capacity`
    @note:
        - only select the individuals who have not left yet
        - if `key` is None, select the individuals with the highest `score`;
            the `key` is some column in `dfi` (current individuals dataframe)
        - if `ascending`, choose the top-C-smallest individuals
            else choose the top-C-largest individuals
    """
    key = "score" if key is None else key
    people_staying = dfi.query("bool_left == 0 & treat_made == 0")
    people_intreatment = people_staying.query("treat == 1")
    remaining_capacity = capacity - people_intreatment.shape[0]
    if remaining_capacity > 0:
        idx_selected = (
            people_staying.query("treat == 0")
            .sort_values(key, ascending=ascending)
            .index[:remaining_capacity]
        )
        dfi.loc[idx_selected, "treat"] = 1.0
        dfi.loc[idx_selected, "score_with_treat"] = (
            dfi.loc[idx_selected, "score"] - dfi.loc[idx_selected, "treat"] * 1.0
        )


def treatment_rule_random(dfi, capacity=100, effect=0.05, **kwargs):
    """
    Select the individuals with randomly.
    """
    people_staying = dfi.query("bool_left == 0 & treat_made == 0")
    people_intreatment = people_staying.query("treat == 1")
    people_notreatment = people_staying.query("treat == 0")
    remaining_capacity = np.round(
        min(capacity - people_intreatment.shape[0], people_notreatment.shape[0])
    ).astype(int)
    if remaining_capacity > 0:
        idx_selected = people_notreatment.sample(remaining_capacity).index
        dfi.loc[idx_selected, "treat"] = 1.0
        dfi["score_with_treat"] = dfi["score"] - dfi["treat"] * effect
