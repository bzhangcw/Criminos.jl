import queue
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class Simulator(object):

    def __init__(self, eval_score_fixed, eval_score_comm):
        self.eval_score_fixed = eval_score_fixed
        self.eval_score_comm = eval_score_comm

    def refit_baseline(self, df_individual):
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

        # Step 3: Create baseline survival interpolator
        self.S0_interp = S0_interp = interp1d(
            cumhaz_df["time"],
            cumhaz_df["S0"],
            kind="previous",
            fill_value="extrapolate",
        )

    # ------------------------------------------------------------------------------
    # simulation utilities
    # ------------------------------------------------------------------------------
    def survival_function(self, t, score):
        """Return estimated survival probability at time t for a given score."""
        S0_t = self.S0_interp(t)
        return S0_t ** np.exp(score)

    def sample_survival_time(self, score):
        """Sample a survival time given a score using inverse transform sampling."""
        u = np.random.uniform()
        S_target = u ** (1 / np.exp(score))  # invert S(t | x)
        # Solve S0(t) = S_target => t = S0^{-1}(S_target)
        t_sampled = float(
            np.interp(
                S_target, self.cumhaz_df["S0"][::-1], self.cumhaz_df["time"][::-1]
            )
        )
        return t_sampled

    def get_new_event(self, row, t_now):
        _tau = self.sample_survival_time(row["score"])
        event = (
            np.round((t_now + _tau) * 100).astype(int),
            (  # sort by event time
                t_now + _tau,
                _tau,
                row.name,
                "r",
            ),
        )
        return event

    def run(
        self,
        dfi,  # individual dataframe, will modify inplace
        dfc,  # community dataframe, will modify inplace
        fo=open("log.txt", "w"),
        p_length=360,
        T_max=1095,
        opt_verbosity=2,
    ):
        """
        Run the simulation for a given population.

        Parameters
        ----------
        dfi : pandas.DataFrame
            Individual dataframe containing individual-level data. Will be modified inplace.
        dfc : pandas.DataFrame
            Community dataframe containing community-level data. Will be modified inplace.
        fo : file object, optional
            File object to write logs to, by default open("log.txt", "w")
        p_length : int, optional
            Length of each period in days, by default 360
        T_max : int, optional
            Maximum simulation time in days, by default 1095
        opt_verbosity : int, optional
            Verbosity level for logging (0-2), by default 2

        Notes
        -----
        The simulation runs for T_max days, with events being processed in chronological order.
        For each individual:
        - Initial events are generated based on their score
        - Events are added to a priority queue
        - When an event occurs, new events may be generated
        - Statistics are tracked for various metrics

        The simulation maintains several counters:
        - num_x: Counts by individual characteristics
        - num_y: Counts by outcome
        - num_n_x: Counts by community characteristics
        - num_n_y: Counts by community outcomes
        - num_n_time: Time-based statistics
        - num_n_mu: Mean statistics
        - num_n_mean_arrest: Mean arrest statistics
        """
        # create an empty priority queue
        self.event_queue = event_queue = queue.PriorityQueue()
        t = 0.0
        p = 0
        # initialize the events
        dfi["current"] = dfi.apply(lambda row: self.get_new_event(row, 0.0), axis=1)
        dfi["acc_survival"] = 0.0

        # statistics
        num_events = 0
        self.num_x = num_x = defaultdict(int)
        self.num_y = num_y = defaultdict(int)
        self.num_n_y = num_n_y = defaultdict(float)
        self.num_n_x = num_n_x = defaultdict(float)
        self.num_n_time = num_n_time = defaultdict(float)
        self.num_n_mu = num_n_mu = defaultdict(float)
        self.num_n_mean_arrest = num_n_mean_arrest = defaultdict(float)
        self.t_dfc = t_dfc = []

        # county ids
        id_cc = dfc.index.to_list()

        # Push an event (time, id, info), using -time for max-heap
        for e in dfi["current"].to_list():
            event_queue.put(e)

        for k, v in dfi.groupby("j")["observed"].apply("count").to_dict().items():
            num_x[k, 0] = v

        for k, v in (
            dfi.groupby("code-county")["observed"].apply("count").to_dict().items()
        ):
            num_n_x[k] = v

        opt_verbosity >= 1 and print("event:happ/  ready to go@", file=fo)
        print(self.log_line(), file=fo, flush=True)
        # ------------------------------------------------------------
        # simulation start
        # ------------------------------------------------------------
        while t < T_max:
            if event_queue.empty():
                break
            # ------------------------------------------------------------
            # update the score
            # ------------------------------------------------------------
            # Pop the newest event (highest time)
            _, (_ac, tau, idx, info) = event = event_queue.get()
            if np.isnan(tau):
                continue
            opt_verbosity >= 2 and print(self.log_event(event), file=fo)
            t = _ac
            num_events += 1
            _episode = np.floor(t / p_length).astype(int)
            _row = dfi.iloc[idx]
            # ------------------------------------------------------------
            dfi.loc[idx, "acc_survival"] += tau
            dfi.loc[idx, "j"] = dfi.loc[idx, "j"] + 1
            num_y[_row["j"], _episode + 1] += 1
            num_n_y[_row["code-county"]] += 1
            num_n_time[_row["code-county"]] += tau
            if _episode > p:
                opt_verbosity >= 1 and print(
                    f"event:happ/\t{_episode}: update score", file=fo
                )
                for _cid in id_cc:
                    dfc.loc[_cid, "percent_re"] = num_n_mu[_cid, _episode] = num_n_y[
                        _cid
                    ] / (num_n_x[_cid] * _episode)
                    dfc.loc[_cid, "mean_re_time"] = num_n_mean_arrest[
                        _cid, _episode
                    ] = num_n_time[_cid] / (num_n_y[_cid] + 1e-10)

                # re-calculate score
                dfc["score_comm"] = dfc.apply(
                    lambda row: self.eval_score_comm(row), axis=1
                )
                dfc_dict = dfc["score_comm"].to_dict()
                t_dfc.append(dfc_dict)
                # update to individuals
                dfi["score_comm"] = dfi["code-county"].map(dfc_dict)
                dfi["score"] = dfi["score_fixed"] + dfi["score_comm"]

                p = _episode

            # ------------------------------------------------------------
            # sample next survival time
            # ------------------------------------------------------------
            _new_event = self.get_new_event(dfi.iloc[idx], t_now=t)
            event_queue.put(_new_event)
            opt_verbosity >= 2 and print(self.log_event(_new_event, gen=True), file=fo)

        opt_verbosity >= 1 and print("event:happ/  finished@", file=fo)
        fo.close()

    def log_line(self):
        return (
            "event:happ/ "
            + f"{'priority':>10s} {'time':>10s} {'surv.':>10s} {'person':>11s} {'t':>1s}"
        )

    def log_event(self, event, gen=False):
        priority, (_ac, tau, idx, info) = event
        if gen:
            return (
                "event:gene/ "
                + f"{priority:10d}| {_ac:10.2f} {tau:10.2f} {idx:10d} {info}"
            )
        else:
            return (
                "event:happ/ "
                + f"{priority:10d}| {_ac:10.2f} {tau:10.2f} {idx:10d} {info}"
            )
