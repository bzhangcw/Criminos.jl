# survival function fitters
import pandas as pd
from autograd import numpy as np
from lifelines import CoxPHFitter
from lifelines.fitters import ParametricRegressionFitter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


# this is the default scoring weights
# @note: this is the default scoring weights for the original model
# these weights are obtained from the following AFTFitter.
DEFAULT_SCORING_PARAMS = {
    "score_fixed": 0.7904,
    "score_comm": 0.7904,
    "score_age_dist": 0.7904,
    "score_felony_arrest": 0.1884,
}


# without refitting, just use the original covariates
def scoring_null(idx, dfi, **kwargs):
    return dfi.loc[idx, "offset"]


def scoring_all(idx, dfi, scoring_weights=DEFAULT_SCORING_PARAMS, **kwargs):

    # this part is already computed in the `input.py`
    return (
        dfi.loc[idx, "score_fixed"] * scoring_weights["score_fixed"]
        + dfi.loc[idx, "score_comm"] * scoring_weights["score_comm"]
        + dfi.loc[idx, "score_state"]
        # note the individual state is weighted already.
    )


class ExponentialAFTFitter(ParametricRegressionFitter):

    # this class property is necessary, and should always be a non-empty list of strings.
    _fitted_parameter_names = ["lambda_"]

    # the below variables maps {dataframe columns, formulas} to parameters
    def _cumulative_hazard(self, params, t, Xs):
        # params is a dictionary that maps unknown parameters to a numpy vector.
        # Xs is a dictionary that maps unknown parameters to a numpy 2d array
        beta = params["lambda_"]
        X = Xs["lambda_"]
        lambda_ = np.exp(np.dot(X, beta))
        return lambda_ * t


"""
Refit the baseline hazard after adding a new covariate `new_col`.
    if refit, then the `produce_score` function will be updated.
"""


def refit_baseline(
    self, df_individual, new_col=None, bool_use_cph=False, baseline="breslow"
):
    if new_col is None:
        refit_baseline_no_extra(self, df_individual, baseline=baseline)
    else:
        refit_baseline_extra(
            self, df_individual, new_col, bool_use_cph, baseline=baseline
        )


def refit_baseline_no_extra(self, df_individual):
    df_sorted = df_individual.sort_values("time")
    times = df_sorted["time"].unique()
    baseline_cumhaz = []

    # Breslow estimate for baseline cumulative hazard
    for t in times:
        # Deaths at time t
        n_deaths = (
            (df_individual["time"] == t) & (df_individual["observed"] == 1)
        ).sum()
        # Risk set at time t
        risk_set = df_individual["time"] >= t
        sum_risk = np.sum(np.exp(df_individual.loc[risk_set, "score"]))
        dLambda = n_deaths / sum_risk if sum_risk > 0 else 0
        baseline_cumhaz.append((t, dLambda))

    self.cumhaz_df = cumhaz_df = pd.DataFrame(
        baseline_cumhaz, columns=["time", "dLambda"]
    )
    cumhaz_df["Lambda"] = cumhaz_df["dLambda"].cumsum()
    cumhaz_df["S0"] = np.exp(-cumhaz_df["Lambda"])

    # Step 3: create baseline survival interpolator
    self.s0_with_interpolation = S0_interp = interp1d(
        cumhaz_df["time"],
        cumhaz_df["S0"],
        kind="previous",
        fill_value="extrapolate",
    )

    self.produce_score = scoring_null


def refit_baseline_extra(
    self, df_individual, new_col, bool_use_cph=False, baseline="breslow"
):
    """
    Refit baseline hazard after adding one extra covariate `new_col`.
    Keeps `survival_function` unchanged by integrating the extra effect
    into each subject's 'score' before recomputing the baseline.
    """
    df = df_individual
    print(df_individual.columns)

    # 2) Fit a Cox model with only new_col as a covariate, old_score as offset
    if baseline in ["gompertz", "breslow"]:
        self.cph = cph = CoxPHFitter(
            penalizer=0.15,
        )
        cph.fit(
            df,
            duration_col="time",
            event_col="observed",
            formula=f"offset + {new_col}",
            show_progress=True,
        )
    else:
        print("Using exponential AFT")
        __refit_baseline_extra_aft_exponential(self, df_individual, new_col)
        return

    # 4) Compute a new linear predictor: score = α * offset + β * new_col
    # @note: the original score is kept in the `offset`
    df_individual["score"] = (
        cph.params_["offset"] * df_individual["offset"]
        + cph.params_[new_col] * df_individual[new_col]
    )

    # 5) Recompute baseline cumulative hazard (Breslow) using combined_score
    df_sorted = df_individual.sort_values("time")
    times = df_sorted["time"].unique()
    baseline_cumhaz = []

    for t in times:
        n_deaths = (
            (df_individual["time"] == t) & (df_individual["observed"] == 1)
        ).sum()
        risk_set = df_individual["time"] >= t
        sum_risk = np.sum(np.exp(df_individual.loc[risk_set, "score"]))
        dLambda = n_deaths / sum_risk if sum_risk > 0 else 0
        baseline_cumhaz.append((t, dLambda))

    cumhaz_df = pd.DataFrame(baseline_cumhaz, columns=["time", "dLambda"])
    cumhaz_df["Lambda"] = cumhaz_df["dLambda"].cumsum()
    cumhaz_df["S0"] = np.exp(-cumhaz_df["Lambda"])

    self.cumhaz_df = cumhaz_df
    self.cumhaz_df_by_cph = cph.baseline_survival_.copy()

    if bool_use_cph:
        self.s0_with_interpolation = interp1d(
            self.cumhaz_df_by_cph.index.values,
            self.cumhaz_df_by_cph["baseline survival"].values,
            kind="previous",
            fill_value="extrapolate",
        )
    else:
        self.s0_with_interpolation = interp1d(
            cumhaz_df["time"].values,
            cumhaz_df["S0"].values,
            kind="previous",
            fill_value="extrapolate",
        )

    self.s0 = self.s0_with_interpolation
    self._scoring_weights = _scoring_weights = {
        "score_fixed": cph.params_["lambda_"]["offset"],
        "score_comm": cph.params_["lambda_"]["offset"],
        "score_age_dist": cph.params_["lambda_"]["offset"],
        f"score_{new_col}": cph.params_["lambda_"][new_col],
    }
    print(_scoring_weights)
    self.produce_score = lambda idx, dfi: scoring_all(idx, dfi, _scoring_weights)

    if baseline == "gompertz":
        __override_Gompertz(self)


def __override_Gompertz(self):
    # 1) sample times
    t_min, t_max = self.cumhaz_df["time"].min(), self.cumhaz_df["time"].max()
    t_obs = np.linspace(t_min, t_max, 200)
    S_obs = self.s0_with_interpolation(t_obs)

    def S_gompertz(t, a, b):
        return np.exp(-(a / b) * (np.exp(b * t) - 1))

    # 3) fit params
    (a_hat, b_hat), _ = curve_fit(S_gompertz, t_obs, S_obs, p0=[0.1, 0.01])
    print(a_hat, b_hat)

    # 4) override the survival function
    self.s0 = lambda t: S_gompertz(t, a_hat, b_hat)


def __refit_baseline_extra_aft_exponential(self, df_individual, new_col):
    df = df_individual.copy()
    df["time"] += 1.0

    # 2) Fit a Cox model with only new_col as a covariate, old_score as offset
    self.cph = cph = ExponentialAFTFitter()
    cph.fit(
        df,
        duration_col="time",
        event_col="observed",
        regressors={"lambda_": f"offset + {new_col}"},
        show_progress=True,
    )

    # 4) Compute a new linear predictor: score = α * offset + β * new_col
    _params = cph.params_["lambda_"]
    df_individual["score"] = (
        _params["offset"] * df_individual["offset"]
        + _params[new_col] * df_individual[new_col]
    )

    λ0 = np.exp(_params["Intercept"])

    # 5) Recompute baseline cumulative hazard (Breslow) using combined_score
    # But this is only for comparison use only.
    df_sorted = df_individual.sort_values("time")
    times = df_sorted["time"].unique()
    baseline_cumhaz = []

    for t in times:
        n_deaths = (
            (df_individual["time"] == t) & (df_individual["observed"] == 1)
        ).sum()
        risk_set = df_individual["time"] >= t
        sum_risk = np.sum(np.exp(df_individual.loc[risk_set, "score"]))
        dLambda = n_deaths / sum_risk if sum_risk > 0 else 0
        baseline_cumhaz.append((t, dLambda))

    cumhaz_df = pd.DataFrame(baseline_cumhaz, columns=["time", "dLambda"])
    cumhaz_df["Lambda"] = cumhaz_df["dLambda"].cumsum()
    cumhaz_df["S0"] = np.exp(-cumhaz_df["Lambda"])

    self.cumhaz_df = cumhaz_df

    self.s0_with_interpolation = lambda t: np.exp(-λ0 * t)
    self.s0 = self.s0_with_interpolation
    self.produce_score = lambda idx, dfi: scoring_all(idx, dfi, new_col, _params)
    self._scoring_weights = _scoring_weights = {
        "score_fixed": cph.params_["lambda_"]["offset"],
        "score_comm": cph.params_["lambda_"]["offset"],
        "score_age_dist": cph.params_["lambda_"]["offset"],
        f"score_{new_col}": cph.params_["lambda_"][new_col],
    }
    print(_scoring_weights)
    self.produce_score = lambda idx, dfi: scoring_all(idx, dfi, _scoring_weights)
