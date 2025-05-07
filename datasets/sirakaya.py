import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use serif font family
        "font.serif": [
            "Computer Modern Roman"
        ],  # Specify specific serif font (default LaTeX font)
        # 'axes.labelsize': 12,  # Font size for labels
        "font.size": 24,  # General font size
        "legend.fontsize": 22,  # Font size for legend
        "xtick.labelsize": 22,  # Font size for x-axis ticks
        "ytick.labelsize": 22,  # Font size for y-axis ticks
    }
)

ucr_to_category = {
    100: 1,
    110: 1,
    120: 1,
    130: 1,
    140: 1,
    200: 2,
    210: 2,
    220: 2,
    300: 3,
    310: 3,
    320: 3,
    400: 4,
    430: 4,
    500: 5,
    510: 5,
    520: 5,
    530: 5,
    600: 6,
    700: 7,
    710: 7,
    720: 7,
    730: 7,
    800: 8,
    810: 8,
    820: 8,
    830: 8,
    840: 8,
    850: 8,
    860: 8,
    870: 8,
    880: 8,
    998: 8,
}


def score_levels(v, vals):
    n = len(vals)
    for i in range(n):
        if v <= i + 1.0:
            return vals[i]
    return vals[n - 1]


def eval_score_fixed(row):
    score = 0.0
    # sex
    score += score_levels(row["sex"], [0.0, -0.534])
    # employment
    score += score_levels(row["employ"], [0.0, -0.206, -0.488])
    # education
    # not significant in the thesis
    # score += score_levels(row["education"], [0.0, 0.57, -0.086, -0.188, -0.301])
    # drug
    score += score_levels(row["drug-abuse"], [0.0, 0.039, 0.403])
    # felony-prior-conviction
    score += score_levels(row["felony-prior-conviction"], [0.0, 0.369, 0.463])
    # age
    score += score_levels(
        row["age-dist"], [0.0, -0.270, -0.456, -0.709, -1.282, -1.808]
    )
    # race
    score += score_levels(row["race"], [0.0, 0.593, 0.245, -0.537, -0.534])
    # ethnicity
    score += score_levels(row["ethnicity"], [0.0, -0.336])
    # offense-type
    score += score_levels(
        row["offense-type"], [0.0, -0.220, 0.739, 0.386, 0.868, 0.936, 0.606, 0.776]
    )
    # term
    score += row["probation-term"] * -0.003
    # supervision-level
    score += score_levels(row["supervision"], [0.0, 0.220, -0.172, 0.238, 0.586, 0.697])
    # ------------------------------------------------------------
    # fixed community level features
    # ------------------------------------------------------------
    score += row["hispanic"] * 0.006
    # poverty-level
    score += row["persons_poverty"] * -0.001
    # ------------------------------------------------------------
    return score


def eval_score_comm(row):
    score = 0.0
    score += row["mean_re_time"] * -0.003
    score += row["percent_re"] * 0.057
    # ------------------------------------------------------------
    return score


def fill_default(df):
    df["survival"] = df["survival"].fillna(df["survival"].max())
    df["supervision"] = df["supervision-level"].fillna(6).apply(lambda x: max(6 - x, 1))
    df["offense-type"] = df["offense-type"].map(ucr_to_category).fillna(8)
    cols_fill_1 = [
        "age-dist",
        "education",
        "sex",
        "race",
        "drug-abuse",
        "felony-prior-conviction",
        "ethnicity",
        "employ",
    ]
    df[cols_fill_1] = df[cols_fill_1].fillna(1)
    return df


def fill_emis_impute_df(
    df,
    cols_fill,
    n_imputations=5,
    n_components=3,
    max_iter=50,
    random_state=0,
):
    """
    EMis-like imputation on selected columns of a DataFrame.

    Args:
        df: pandas DataFrame
        cols_fill: list of column names to impute
        n_imputations: number of complete DataFrames to return
        n_components: number of GMM components
        max_iter: EM iterations
        random_state: random seed

    Returns:
        List of DataFrames, each with selected columns imputed
    """
    np.random.seed(random_state)
    # Subset the columns to fill
    X = df[cols_fill].to_numpy(dtype=float)

    # Initial mean imputation
    X_obs = np.copy(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        mean_val = np.nanmean(col)
        col[np.isnan(col)] = mean_val
        X_obs[:, j] = col

    # Fit Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=n_components, max_iter=max_iter, random_state=random_state
    )
    gmm.fit(X_obs)

    imputations = []
    for _ in range(n_imputations):
        X_imp = np.copy(X)
        for i in range(X.shape[0]):
            missing = np.isnan(X[i])
            if np.any(missing):
                mu = gmm.means_
                sigma = gmm.covariances_
                weights = gmm.predict_proba(X_obs[i].reshape(1, -1)).flatten()
                component = np.random.choice(np.arange(n_components), p=weights)

                mu_i = mu[component]
                sigma_i = sigma[component]
                sampled = np.random.multivariate_normal(mu_i, sigma_i)

                X_imp[i, missing] = sampled[missing]

        # Replace only the selected columns
        df_new = df.copy()
        df_new[cols_fill] = X_imp
        imputations.append(df_new)

    return imputations, df_new
