"""
Simulation module for criminal recidivism analysis.
------------------------------------------------------------------------------
@author: cz
@note:
    This module implements a discrete event simulation system for modeling criminal recidivism
    patterns. It tracks individuals through probation periods, handling events like:
    - Arrivals into the system
    - Recidivism events
    - Departures from probation

    The simulation maintains community-level statistics and individual trajectories to analyze
    the impact of various factors on recidivism rates.

    Key components:
    - Event class: Represents discrete events in the simulation
    - Simulator class: Main simulation engine handling event processing and state updates
    - Utility functions: Helper functions for event generation and timing
"""

import queue
from collections import defaultdict
from enum import IntEnum, Enum
import warnings
import numpy as np
import pandas as pd
import itertools
from lifelines import CoxPHFitter
from pyexcelerate import Workbook
from scipy.interpolate import interp1d

import sirakaya
from tqdm.auto import tqdm as _tqdm


class EventType(str, Enum):
    """Event types in the simulation."""

    EVENT_ARR = "arriv"
    EVENT_LEAVE = "leave"
    EVENT_RECID = "offnd"
    EVENT_END_PROD = "endpr"
    EVENT_RETURN = "retrn"


MAX_ARRESTS = 15


class ReasonToLeave(IntEnum):
    END_OF_PROCESS = 1
    RETURNING_AGENT = 2


from simulation_extra import *
from simulation_treatment import (
    treatment_null,
    treatment_rule_priority,
    treatment_rule_random,
)


def encode_priority(t):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            return np.round(t * 100).astype(int)
        except RuntimeWarning as e:
            print("Invalid value in cast for:", t)
            raise


class Event(object):

    def __init__(self, *args, idx_ref=-1, companion=None):
        """
        Args:
            time: time of the event
            tau: inter-arrival time of the event
                    wrt to the previous event
            idx: index of the event
            type: type of the event
            idx_ref: index for reference to the sample distribution
            companion: companion of the event, e.g.,
                the leaving event of an individual as for the arrival
        """
        self.time, self.tau, self.idx, self.type, *_ = args
        self.priority = encode_priority(self.time)
        self.idx_ref = idx_ref
        self.companion = companion
        self.hashid = self.__hashid__()

    def __hashid__(self):
        return f"{self.time:.2f}_{self.idx}_{self.type}"

    def unpack(self):
        return self.priority, self.time, self.tau, self.idx, self.type

    def __lt__(self, other):
        return self.priority < other.priority


# ------------------------------------------------------------------------------
# state and score functions definitions
# ------------------------------------------------------------------------------
class StateVal(object):
    def __init__(self, values):
        self.value = tuple(values) if len(values) > 1 else (values.values[0],)

    def __repr__(self):
        return str(self.value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __ne__(self, other):
        return self.value != other.value

    def __lt__(self, other):
        try:
            other_value = other.value
        except AttributeError:
            return str(self.value) < str(other)
        return self.value < other_value

    def __gt__(self, other):
        try:
            other_value = other.value
        except AttributeError:
            return str(self.value) > str(other)
        return self.value > other_value

    def __hash__(self):
        return hash(self.value)

    def __len__(self):
        return len(self.value)

    def __copy__(self):
        return StateVal(self.value)


STATE_KEY_RANGE = {
    "felony_arrest": list(range(11)),
    "age": list(range(18, 55)),
    "age_dist": [1, 2, 3, 4, 5, 6],  # ignore 0
}


class StateDefs(object):
    def __init__(self, state_keys):
        self.state_keys = state_keys
        self.scoring_weights = DEFAULT_SCORING_PARAMS
        self.state_key_range = {
            k: i
            for i, k in enumerate(
                itertools.product(*[STATE_KEY_RANGE[key] for key in state_keys])
            )
        }
        self.states_index = [
            k for k in itertools.product(*[STATE_KEY_RANGE[key] for key in state_keys])
        ]

    def get_state(self, row):
        return StateVal(row[self.state_keys].astype(float))

    def get_score(self, row):

        return sum(
            row[f"score_{key}"] * self.scoring_weights[f"score_{key}"]
            for key in self.state_keys
        )

    def update_state(self, dfi, idx, t):
        # ------------------------------------------------------------
        # update the individual score and state
        # ------------------------------------------------------------
        # update the age, age_dist, and score_age_dist according to the new age
        dfi.loc[idx, "age"] = (
            dfi.loc[idx, "age_start"] + (t - dfi.loc[idx, "arrival"]) / 365.0
        )
        dfi.loc[idx, "age_dist"] = sirakaya.age_to_age_dist(dfi.loc[idx, "age"])
        dfi.loc[idx, "score_age_dist"] = sirakaya.score_age_dist(
            dfi.loc[idx, "age_dist"]
        )
        # the score of felony_arrest is just itself
        dfi.loc[idx, "score_felony_arrest"] = dfi.loc[idx, "felony_arrest"]
        # update the state and score_state
        dfi.loc[idx, "state"] = self.get_state(dfi.loc[idx])
        dfi.loc[idx, "score_state"] = self.get_score(dfi.loc[idx])


class Simulator(object):

    # these columns are initialized for each individual when the simulation starts
    INITIALIZED_COLUMNS = [
        "acc_survival",  # accumulated survival time(s)
        "bool_left",  # whether the individual has left (end of probation term)
        "type_left",  # the reason for leaving (end of probation term / too many arrests)
        "observed",  # whether the individual has been observed (ever arrested)
        "bool_treat",  # whether the individual has been treated
        "bool_treat_made",  # whether the treatment decision (either 0 or 1) has been made
        "score_with_treat",  # score of the individual with treatment
        "ep_arrival",  # episode of arrival
        "ep_leaving",  # episode of leaving
        "return_times",  # number of times the individual has returned to the community
    ]

    SCORING_COLUMNS = ["score_fixed", "score_comm"]
    TRAJECTORY_COLUMNS = [
        "index",
        "snap",
        "arrival",
        "leaving",
        "ep_arrival",
        "ep_leaving",
        "ep_lastre",
        "felony_arrest_lst",
        "felony_arrest",
        "bool_treat",
        "bool_treat_made",
        "type_left",
    ]

    def __init__(
        self,
        eval_score_fixed,
        eval_score_comm,
        func_arrival=None,
        func_end_probation=None,
        func_leaving=None,
        state_defs=StateDefs(["felony_arrest"]),
        bool_keep_full_trajectory=True,
    ):
        self.eval_score_fixed = eval_score_fixed
        self.eval_score_comm = eval_score_comm

        self.func_arrival = func_arrival
        self.func_end_probation = (
            func_end_probation
            if func_end_probation is not None
            else self.__default_end_probation
        )
        self.func_leaving = (
            func_leaving if func_leaving is not None else self.__default_leaving
        )

        self.state_defs = state_defs
        self.state_defs.scoring_weights = DEFAULT_SCORING_PARAMS

        print(self.state_defs.scoring_weights)
        self.bool_keep_full_trajectory = bool_keep_full_trajectory

    def __default_end_probation(self, row, t_now):
        delta = row["rel_probation"]
        return Event(
            t_now + delta,
            delta,
            row.name,
            EventType.EVENT_END_PROD,
            idx_ref=-1,
        )

    def __default_leaving(self, row, t_now):
        delta = row["rel_probation"] + row["rel_off_probation"]
        return Event(
            t_now + delta,
            delta,
            row.name,
            EventType.EVENT_LEAVE,
            idx_ref=-1,
        )

    def __default_return(self, row, t_now):
        return Event(
            t_now + 0.0,
            0.0,
            row.name,
            EventType.EVENT_RETURN,
            idx_ref=-1,
        )

    # ------------------------------------------------------------------------------
    # simulation utilities
    # ------------------------------------------------------------------------------
    def survival_function(self, t, score):
        """Return estimated survival probability at time t for a given score."""
        S0_t = self.s0(t)
        return S0_t ** np.exp(score)

    def mean_rearrest_time(self, score, t_max=5000, n_steps=100):
        """
        Approximate E[T | score] ≈ ∫_0^{t_max} S(t | score) dt
        by splitting [0, t_max] into n_steps intervals.
        """
        times = np.linspace(0, t_max, n_steps + 1)  # include 0 and t_max
        surv_vals = [
            self.survival_function(t, score) for t in times
        ]  # vector of S(t_i)
        # use trapezoidal rule:
        return np.trapezoid(surv_vals, times)

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

    def run(
        self,
        dfi,  # individual dataframe, will modify inplace
        dfc,  # community dataframe, will modify inplace
        dfpop,  # population dataframe to draw new arrivals
        seed=1,
        # ----------------------------------------------------------------------
        fo=open("log.txt", "w"),
        opt_verbosity=2,
        T_max=1095,
        p_length=360,
        p_freeze=4,
        rel_off_probation=500,
        # ----------------------------------------------------------------------
        func_treatment=None,
        func_treatment_kwargs={},
    ):
        """
        Run the simulation for a given population.

        Parameters
        ----------------------------------------------------------------------
        dfi : pandas.DataFrame
            Individual dataframe containing individual-level data.
            Will be modified inplace.
        dfc : pandas.DataFrame
            Community dataframe containing community-level data.
            Will be modified inplace.
        dfpop : pandas.DataFrame
            Population dataframe containing population data.
            Will be sampled from to generate new arrivals.
        ----------------------------------------------------------------------
        fo : file object, optional
            File object to write logs to, by default open("log.txt", "w")
        opt_verbosity : int, optional
            Verbosity level for logging (0-2), by default 2
        ----------------------------------------------------------------------
        T_max : int, optional
            Maximum simulation time in days, by default 1095
        p_length : int, optional
            Length of each period in days, by default 360
        p_freeze : int, optional
            Number of periods to run before updating the score, by default 100

        Notes
        ----------------------------------------------------------------------
        The simulation runs for T_max days,
            with events being processed in chronological order.
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
        ...
        ----------------------------------------------------------------------
        """
        np.random.seed(seed)
        # create an empty priority queue
        self.event_queue = event_queue = queue.PriorityQueue()
        self.events_cancelled = set()
        t = 0.0
        p = p_freeze
        n_persons_appeared = dfi.shape[0]
        self.dfpop = dfpop
        # initialize the population
        self.dfi = dfi
        dfi[self.INITIALIZED_COLUMNS] = 0.0
        dfi["hash_leave"] = ""
        dfi["arrival"] = 0.0
        dfi["rel_off_probation"] = rel_off_probation  # initial off-probation term
        dfi["ep_leaving"] = 1e6
        dfi["ep_end_probation"] = 1e6
        dfi["age_start"] = dfi["age"].copy()
        dfi["age_dist_start"] = dfi["age_dist"].copy()
        dfi["prio_arrest"] = dfi["felony_arrest"].copy()
        dfi["state_lst"] = dfi["state"].apply(lambda x: StateVal(x.value))
        dfi["bool_in_probation"] = 1

        # statistics
        self.num_y = num_y = defaultdict(int)
        self.num_lbd = num_lbd = defaultdict(int)
        self.num_lft = num_lft = defaultdict(int)
        self.num_flow = num_flow = defaultdict(int)

        # used to update the community score
        self.num_n_y = num_n_y = defaultdict(float)
        self.num_n_time = num_n_time = defaultdict(float)
        self.num_n_mu = num_n_mu = defaultdict(float)
        self.num_n_mean_arrest = num_n_mean_arrest = defaultdict(float)

        self.t_dfc = t_dfc = []
        self.t_dfi = t_dfi = []
        self.traj_results = []

        # initialize the community score
        dfc_dict = dfc["score_comm"].to_dict()

        # county id
        self.id_cc = id_cc = dfc.index.to_list()

        opt_verbosity >= 1 and print("event:happ/  ready to go@", file=fo)
        print(self.log_header(), file=fo, flush=True)

        # assign initial treatment
        func_treatment(dfi, **func_treatment_kwargs)

        # --------------------------------------------------------------------------
        # assign end-probation & leaving events to existing individuals
        # --------------------------------------------------------------------------
        for idx, row in dfi.iterrows():
            # end-probation event
            _end_probation = self.func_end_probation(dfi.loc[idx], t)
            event_queue.put((_end_probation.priority, _end_probation))
            dfi.loc[idx, "end_probation"] = _end_probation.time
            dfi.loc[idx, "ep_end_probation"] = 1e6
            # leaving event
            _leave_event = self.func_leaving(row, t)
            dfi.loc[idx, "leaving"] = _leave_event.time
            dfi.loc[idx, "ep_leaving"] = 1e6
            event_queue.put((_leave_event.priority, _leave_event))
            dfi.loc[idx, "hash_leave"] = _leave_event.hashid

        # --------------------------------------------------------------------------
        # initialize a recidivism event if it is before the leaving event
        # --------------------------------------------------------------------------
        for idx, row in dfi.iterrows():
            _recid_event = self.get_new_recid_event(
                row, t_now=0.0, opt_verbosity=0, fo=fo
            )
            if _recid_event is not None:
                event_queue.put((_recid_event.priority, _recid_event))

        # push one arrival event
        if self.func_arrival is not None:
            _row = dfpop.sample(1).iloc[0]
            # arrival event
            _arrival = self.func_arrival(
                _row,
                t,
                n_persons_appeared + 1,  # index of the new person
                _row.name,  # the idx/name of the row in the sampling distribution (df)
            )
            if _arrival is not None:
                event_queue.put((_arrival.priority, _arrival))
                opt_verbosity >= 2 and print(
                    self.log_event(_arrival, gen=True), file=fo
                )

        # ------------------------------------------------------------
        # simulation start
        # ------------------------------------------------------------
        # keep a copy of the initial individuals dataframe
        self.dfi_init = dfi.copy(deep=True)
        self.dfc_init = dfc.copy(deep=True)

        # run until the end of the simulation by T_max days
        # initialize progress bar (updates by simulated time advance)
        _pbar = _tqdm(
            total=T_max,
            desc="agent-based simulation running...",
            leave=False,
            disable=(opt_verbosity < 1),
        )
        while t < T_max:
            if event_queue.empty():
                break

            # pop and process the newest event (highest time)
            _, event = event_queue.get()
            _, _ac, tau, idx, info = event.unpack()
            if np.isnan(tau) or event.hashid in self.events_cancelled:
                continue
            opt_verbosity >= 2 and print(self.log_event(event), file=fo)
            t = _ac
            _inc = max(0.0, min(T_max, t) - _pbar.n)
            _pbar.update(int(_inc))
            _episode = np.floor(t / p_length).astype(int)
            _bool_skip_next_offense = False

            if info in (EventType.EVENT_ARR, EventType.EVENT_RETURN):
                # new arrival and add to the population
                if info == EventType.EVENT_ARR:
                    n_persons_appeared += 1
                    dfi.loc[idx] = dfpop.loc[event.idx_ref]
                dfi.loc[idx, self.INITIALIZED_COLUMNS] = 0
                dfi.loc[idx, "arrival"] = t
                dfi.loc[idx, "ep_arrival"] = _episode
                dfi.loc[idx, "age_start"] = dfi.loc[idx, "age"]
                dfi.loc[idx, "age_dist_start"] = dfi.loc[idx, "age_dist"]
                dfi.loc[idx, "prio_arrest"] = dfi.loc[idx, "felony_arrest"]
                dfi.loc[idx, "state"] = self.state_defs.get_state(dfi.loc[idx])
                dfi.loc[idx, "state_lst"] = StateVal(dfi.loc[idx, "state"].value)
                dfi.loc[idx, "score_state"] = self.state_defs.get_score(dfi.loc[idx])
                dfi.loc[idx, "score_comm"] = dfc_dict.get(dfi.loc[idx, "code_county"])
                dfi.loc[idx, "score"] = self.produce_score(idx, dfi)
                # add a perturbation to the leaving time
                dfi.loc[idx, "bool_in_probation"] = 1
                dfi.loc[idx, "rel_probation"] *= np.random.uniform(0.7, 1.3)
                dfi.loc[idx, "rel_off_probation"] = rel_off_probation

                # not assigned to any treatment yet
                dfi.loc[idx, "score_with_treat"] = dfi.loc[idx, "score"]

                # ------------------------------------------------------------
                # produce the end-probation event
                # ------------------------------------------------------------
                _end_probation = self.func_end_probation(dfi.loc[idx], t)
                opt_verbosity >= 2 and print(
                    self.log_event(_end_probation, gen=True), file=fo
                )
                event_queue.put((_end_probation.priority, _end_probation))
                dfi.loc[idx, "end_probation"] = _end_probation.time
                dfi.loc[idx, "ep_end_probation"] = 1e6
                # ------------------------------------------------------------
                # produce the leaving event
                # ------------------------------------------------------------
                _leaving = self.func_leaving(dfi.loc[idx], t)
                opt_verbosity >= 2 and print(
                    self.log_event(_leaving, gen=True), file=fo
                )
                event_queue.put((_leaving.priority, _leaving))
                dfi.loc[idx, "leaving"] = _leaving.time
                dfi.loc[idx, "ep_leaving"] = 1e6
                dfi.loc[idx, "hash_leave"] = _leaving.hashid

                # generate next arrival individual
                _new_row = dfpop.sample(1).iloc[0]
                _arrival = self.func_arrival(
                    _new_row,
                    t,
                    n_persons_appeared + 1,
                    _new_row.name,
                )
                if _arrival is not None:
                    opt_verbosity >= 2 and print(
                        self.log_event(_arrival, gen=True), file=fo
                    )
                    event_queue.put((_arrival.priority, _arrival))

                    num_lbd[_row["code_county"], dfi.loc[idx, "state"], _episode] += 1

            elif info == EventType.EVENT_END_PROD:
                # process leaving events
                dfi.loc[idx, "ep_end_probation"] = _episode
                dfi.loc[idx, "bool_in_probation"] = 0

            elif info == EventType.EVENT_LEAVE:
                # process leaving events
                dfi.loc[idx, "bool_left"] = _bool_skip_next_offense = 1
                dfi.loc[idx, "ep_leaving"] = _episode
                dfi.loc[idx, "type_left"] = ReasonToLeave.END_OF_PROCESS

                num_lft[_row["code_county"], _row["state"], _episode] += 1

            elif info == EventType.EVENT_RECID:
                _row = dfi.loc[idx]
                # ------------------------------------------------------------
                dfi.loc[idx, "observed"] = 1
                dfi.loc[idx, "ep_lastre"] = _episode
                dfi.loc[idx, "acc_survival"] += tau

                # accumulate the number of arrests
                num_n_y[_row["code_county"]] += 1
                num_n_time[_row["code_county"]] += tau

                # per-episode number of arrests
                num_y[_row["code_county"], _row["state"], _episode] += 1

                # ------------------------------------------------------------------------------
                # an off-probation offense?
                # ------------------------------------------------------------------------------
                # if offend not in the probation term, then the individual is leaving
                #   and is returning to the community
                dfi.loc[idx, "felony_arrest"] = dfi.loc[idx, "felony_arrest"] + 1
                if dfi.loc[idx, "bool_in_probation"] == 0:
                    # this person is returning to the community (again)
                    dfi.loc[idx, "return_times"] = dfi.loc[idx, "return_times"] + 1
                    dfi.loc[idx, "bool_in_probation"] = 1
                    _return = self.__default_return(dfi.loc[idx], t)
                    opt_verbosity >= 2 and print(
                        self.log_event(_return, gen=True), file=fo
                    )
                    event_queue.put((_return.priority, _return))
                    # cancel prescheduled leaving event
                    self.events_cancelled.add(dfi.loc[idx, "hash_leave"])
                    _bool_skip_next_offense = True

            else:
                raise ValueError(f"Unknown event type: {info}")

            # ------------------------------------------------------------
            # update the community score;
            # synchronize the individual score
            # synchronize the states with the community score
            # ------------------------------------------------------------
            # if episode is upgraded;
            #  then update the score of the community
            if _episode > p:
                opt_verbosity >= 1 and print(
                    f"event:happ/\t{_episode}: update score", file=fo
                )
                for _cid in id_cc:
                    # computation of the percentage of re-arrested people
                    # this means the fraction of people who have been arrested at least once
                    # during the probation period
                    # @note: old method, not so accurate?
                    #   dfc.loc[_cid, "percent_re"]
                    #       = num_n_mu[_cid, _episode] = num_n_y[_cid] / (num_n_x[_cid] * _episode)
                    # @note: new method
                    #   compute the rate of re-arrested people
                    #   ever appeared in the county
                    _sum_observed = (
                        dfi.groupby("code_county")
                        .agg(
                            {
                                "score_fixed": "count",  # use this only to count total # of individuals
                                "observed": "sum",
                            }
                        )
                        .assign(rate=lambda x: x["observed"] / x["score_fixed"])
                    )
                    dfc.loc[_cid, "percent_re"] = num_n_mu[_cid, _episode] = (
                        _sum_observed.loc[_cid, "rate"]
                    )

                    # computation of the mean re-arrest time
                    # for each cid, just accumulate the time and the number of arrests
                    # mean_re_time = (accummulated) total_time / (number_of_arrests + 1e-10)
                    dfc.loc[_cid, "mean_re_time"] = num_n_mean_arrest[
                        _cid, _episode
                    ] = num_n_time[_cid] / (num_n_y[_cid] + 1e-10)

                # re-calculate score for the community
                dfc["score_comm"] = dfc.apply(
                    lambda row: self.eval_score_comm(row), axis=1
                )
                dfc_dict = dfc["score_comm"].to_dict()

                # ------------------------------------------------------------------------------
                # synchronize the individual score and state with the community score
                # ------------------------------------------------------------------------------
                # keep the last state if leave not earlier than the current episode
                _idx_not_before = dfi["ep_leaving"] >= p
                dfi.loc[_idx_not_before, "state_lst"] = dfi.loc[
                    _idx_not_before, "state"
                ].apply(lambda x: StateVal(x.value))
                # update state and state score.
                for idx in dfi[_idx_not_before].index:
                    self.state_defs.update_state(dfi, idx, t)

                # update the community score
                # to individuals who did not leave yet
                _idx_staying = dfi["bool_left"] != 0

                dfi.loc[_idx_staying, "score_comm"] = dfi.loc[
                    _idx_staying, "code_county"
                ].map(dfc_dict)

                # produce the score
                dfi.loc[_idx_staying, "score"] = self.produce_score(_idx_staying, dfi)

                # apply treatment rule
                func_treatment(dfi, **func_treatment_kwargs)
                dfi["bool_treat_made"] = 1

                # keep trajectory
                dfc_sorted = dfc.assign(snap=p).reset_index()
                dfi_sorted = dfi.assign(snap=p).reset_index()
                if self.bool_keep_full_trajectory:
                    t_dfc.append(dfc_sorted)
                    t_dfi.append(dfi_sorted.reindex(sorted(dfi_sorted.columns), axis=1))

                p = _episode

            # ------------------------------------------------------------
            # sample next survival time
            # ------------------------------------------------------------
            if _bool_skip_next_offense:
                # does not seem necessary to drop the individual
                # dfi.drop(index=idx, inplace=True)
                pass
            elif info in (
                EventType.EVENT_RECID,
                EventType.EVENT_ARR,
                EventType.EVENT_RETURN,
            ):
                _new_event = self.get_new_recid_event(
                    dfi.loc[idx], t_now=t, opt_verbosity=opt_verbosity, fo=fo
                )
                if _new_event is not None:
                    event_queue.put((_new_event.priority, _new_event))
        # ensure bar is complete and close cleanly
        if _pbar.n < T_max:
            _pbar.update(T_max - _pbar.n)
        _pbar.close()
        opt_verbosity >= 1 and print("event:happ/  finished@", file=fo)
        fo.close()
        self.n_persons_appeared = n_persons_appeared

    def log_header(self):
        return (
            "event:happ/   "
            + f"{'type':>4s} {'time':>10s} {'dura.':>10s} {'person':>18s} {'priority':>10s}"
        )

    def log_event(self, event, gen=False):
        priority, _ac, tau, idx, info = event.unpack()
        _agg_id = f"{idx}".zfill(10) + f"^({event.idx_ref})"
        if gen:
            return (
                "   |-:gene/   "
                + f"{info:>4s} {_ac:10.2f} {tau:10.2f} {_agg_id:>18s} {priority:10d}"
            )
        else:
            return (
                "event:happ/   "
                + f"{info:>4s} {_ac:10.2f} {tau:10.2f} {_agg_id:>18s} {priority:10d}"
            )

    def summarize(self, save=True):
        self.df_result = df_result = pd.concat(
            self.t_dfi, ignore_index=True
        ).reset_index(drop=True)
        self.df_result_comm = df_result_comm = (
            pd.concat(self.t_dfc, ignore_index=True)
            .astype({"snap": "int", "code_county": "str"})
            .reset_index(drop=True)
        )
        if save:
            self.__save_df_as_excel(df_result, "result")
            self.__save_df_as_excel(df_result_comm, "result_comm")
        # plot trend
        self.fig_trend = plot_trend(self)

    def __save_df_as_excel(self, df, name):
        # save the full result to excel
        # convert DataFrame to 2D list with header
        data = [df.columns.tolist()] + df.values.tolist()

        # write to excel via pyexcelerate
        wb = Workbook()
        wb.new_sheet("Sheet1", data=data)
        wb.save(f"{name}.xlsx")
        print(f"{name}.xlsx successfully saved...")

    def summarize_trajectory(
        self,
        p_freeze,
        df_community,
        state_lst_columns=["state_lst"],
        state_columns=["state"],
        windowsize=10,
    ):
        snaps = len(self.t_dfi)
        results = []
        results_df = []
        results_flow = []
        results_retention = []
        for p in range(0, snaps - 1):
            df_begin = (
                self.t_dfi[p]
                .copy()
                .dropna(subset=["state", "state_lst"])
                .assign(
                    state=lambda df: df["state"].apply(lambda x: x.value),
                    state_lst=lambda df: df["state_lst"].apply(lambda x: x.value),
                )
            )
            lst_cols_avail = [c for c in state_lst_columns if c in df_begin.columns]
            cur_cols_avail = [c for c in state_columns if c in df_begin.columns]

            grp_keys_lst = ["code_county"] + lst_cols_avail
            grp_keys_cur = ["code_county"] + cur_cols_avail
            grp_keys_flow = grp_keys_lst + cur_cols_avail
            _boundary = p + p_freeze
            _present_start_mask = (df_begin["ep_arrival"] < _boundary) & (
                df_begin["ep_leaving"] >= _boundary
            )
            _present_end_mask = (df_begin["ep_arrival"] <= _boundary) & (
                df_begin["ep_leaving"] > _boundary
            )
            _leave_now_mask = df_begin["ep_leaving"] == _boundary
            _arrive_now_mask = df_begin["ep_arrival"] == _boundary
            _recid_now_mask = df_begin["ep_lastre"] == _boundary

            df_presence_begin = df_begin.loc[_present_start_mask]
            x0 = df_presence_begin.groupby(grp_keys_lst)["index"].count()
            x = df_begin.loc[_present_end_mask].groupby(grp_keys_cur)["index"].count()
            y = df_begin.loc[_recid_now_mask].groupby(grp_keys_flow)["index"].count()
            yout = y.groupby(grp_keys_lst).sum()
            yin = y.groupby(grp_keys_cur).sum()
            lmd = df_begin.loc[_arrive_now_mask].groupby(grp_keys_lst)["index"].count()
            lv = (
                df_presence_begin.loc[_leave_now_mask]
                .groupby(grp_keys_lst)["index"]
                .count()
            )
            tau = df_presence_begin.groupby(grp_keys_lst)["bool_treat"].sum()
            df_traj = (
                pd.DataFrame(
                    {
                        "x0": x0,
                        "x": x,
                        "lmd": lmd,
                        "lv": lv,
                        "yin": yin,
                        "yout": yout,
                        "tau": tau,
                    }
                )
                .fillna(0)
                .rename_axis(index=grp_keys_lst)
                .assign(
                    xcal=lambda df: df.apply(
                        lambda row: row["x0"]
                        + row["lmd"]
                        - row["lv"]
                        + row["yin"]
                        - row["yout"],
                        axis=1,
                    ),
                    error=lambda df: df["xcal"] - df["x"],
                )
            )
            error = df_traj["error"].sum()
            print(f"error @ p={p}: {error}")
            rr = {}
            for cc in df_community.index:
                try:
                    if (
                        isinstance(y.index, pd.MultiIndex)
                        and "code_county" in y.index.names
                    ):
                        sub = y.xs(cc, level="code_county", drop_level=True)
                        items = sub.to_dict().items()
                        yc = [
                            [*(k if isinstance(k, tuple) else (k,)), v]
                            for k, v in items
                        ]
                    else:
                        yc = {}
                except Exception:
                    yc = {}
                try:
                    other = df_traj.loc[cc, :].to_dict()
                except Exception:
                    other = {}
                rr[cc] = {**other, "y": yc}
            results.append(rr)
            results_df.append(df_traj)

            # compute exact flows from previous state(s) to current state(s)
            try:
                # classify x/y strictly by arrest count change at the boundary:
                # y: recidivism occurs at boundary; x: present and no recidivism and not leaving
                # x: present at start and no recid at boundary (ignoring leave/end presence)
                x_flow_df = (
                    df_begin.loc[_present_start_mask & (~_recid_now_mask)]
                    .groupby(grp_keys_flow)["index"]
                    .count()
                    .rename("count")
                    .reset_index()
                )
                x_flow_df["source"] = "x"
                # ratio per origin (code_county + previous-state columns) within source x
                _xtot = x_flow_df.groupby(grp_keys_lst)["count"].transform("sum")
                x_flow_df["ratio"] = np.where(
                    _xtot > 0, x_flow_df["count"] / _xtot, 0.0
                )

                # y: present at start and recid at boundary (ignoring leave/end presence)
                y_flow_df = (
                    df_begin.loc[_present_start_mask & _recid_now_mask]
                    .groupby(grp_keys_flow)["index"]
                    .count()
                    .rename("count")
                    .reset_index()
                )
                y_flow_df["source"] = "y"
                # ratio per origin (code_county + previous-state columns) within source y
                _ytot = y_flow_df.groupby(grp_keys_lst)["count"].transform("sum")
                y_flow_df["ratio"] = np.where(
                    _ytot > 0, y_flow_df["count"] / _ytot, 0.0
                )

                # arrivals as separate source with previous-state marked 'NEW'
                arrival_flow_df = (
                    df_begin.loc[_arrive_now_mask]
                    .groupby(grp_keys_cur)["index"]
                    .count()
                    .rename("count")
                    .reset_index()
                )
                for prev_col in lst_cols_avail:
                    arrival_flow_df[prev_col] = "NEW"
                _missing_cols = [
                    c for c in grp_keys_flow if c not in arrival_flow_df.columns
                ]
                for c in _missing_cols:
                    if c in lst_cols_avail:
                        arrival_flow_df[c] = "NEW"
                    else:
                        arrival_flow_df[c] = np.nan
                arrival_flow_df = arrival_flow_df[grp_keys_flow + ["count"]]
                arrival_flow_df["source"] = "new"
                _ntot_new = arrival_flow_df.groupby(grp_keys_lst)["count"].transform(
                    "sum"
                )
                arrival_flow_df["ratio"] = np.where(
                    _ntot_new > 0, arrival_flow_df["count"] / _ntot_new, 0.0
                )

                flow_df = pd.concat(
                    [x_flow_df, y_flow_df, arrival_flow_df], ignore_index=True
                ).assign(
                    state_key=lambda df: df["state"].apply(
                        self.state_defs.state_key_range.get
                    ),
                    state_lst_key=lambda df: df["state_lst"].apply(
                        self.state_defs.state_key_range.get
                    ),
                )
            except Exception:
                # fallback to empty frame if grouping keys are missing
                flow_df = pd.DataFrame(columns=grp_keys_flow + ["count", "source", "ratio", "ratio_all"])  # type: ignore

            # compute ratios: within-source and across all sources per origin
            if len(flow_df) > 0:
                _src_tot = flow_df.groupby(grp_keys_lst + ["source"], group_keys=False)[
                    "count"
                ].transform("sum")
                flow_df["ratio"] = np.where(
                    _src_tot > 0, flow_df["count"] / _src_tot, 0.0
                )
                _orig_tot = flow_df.groupby(grp_keys_lst, group_keys=False)[
                    "count"
                ].transform("sum")
                flow_df["ratio_all"] = np.where(
                    _orig_tot > 0, flow_df["count"] / _orig_tot, 0.0
                )

            # sort by code_county, source, previous-state columns, then current-state columns
            sort_cols = ["code_county", "source"] + lst_cols_avail + cur_cols_avail
            sort_cols = [c for c in sort_cols if c in flow_df.columns]
            if len(sort_cols) > 0:
                flow_df = flow_df.sort_values(by=sort_cols).reset_index(drop=True)

            flow_df["snap"] = p
            results_flow.append(flow_df)

            # compute retention per origin based on starts (including arrivals) and leaves
            _start_mask_union = _present_start_mask | _arrive_now_mask
            _starts = (
                df_begin.loc[_start_mask_union].groupby(grp_keys_lst)["index"].count()
            )
            _leaves = (
                df_begin.loc[_leave_now_mask & _start_mask_union]
                .groupby(grp_keys_lst)["index"]
                .count()
            )
            _ret_df = (
                pd.DataFrame({"total": _starts, "left": _leaves})
                .fillna(0)
                .assign(
                    stay=lambda d: d["total"] - d["left"],
                    retention_ratio=lambda d: np.where(
                        d["total"] > 0, (d["total"] - d["left"]) / d["total"], 0.0
                    ),
                )
                .reset_index()
            )
            _ret_df["snap"] = p
            retention_df = _ret_df.assign(
                state_lst_key=lambda df: df["state_lst"].apply(
                    self.state_defs.state_key_range.get
                ),
            )

            results_retention.append(retention_df)

        # compute moving-average counts first, then ratios over last N episodes
        if len(results_flow) > 0:
            _all_flow = pd.concat(results_flow, ignore_index=True)
            needed_cols = grp_keys_flow + ["source", "snap", "count"]
            if all(c in _all_flow.columns for c in needed_cols):
                _all_flow = _all_flow.sort_values(by=grp_keys_flow + ["source", "snap"])
                # rolling average counts per origin+source+destination
                _all_flow["count_ma"] = (
                    _all_flow.groupby(grp_keys_flow + ["source"], group_keys=False)[
                        "count"
                    ]
                    .rolling(window=windowsize, min_periods=10)
                    .mean()
                    .reset_index(level=list(range(len(grp_keys_flow) + 1)), drop=True)
                )
                # per-source normalization across destinations
                _all_flow["origin_source_total_ma"] = _all_flow.groupby(
                    grp_keys_lst + ["source", "snap"], group_keys=False
                )["count_ma"].transform("sum")
                _all_flow["ratio_ma"] = np.where(
                    _all_flow["origin_source_total_ma"] > 0,
                    _all_flow["count_ma"] / _all_flow["origin_source_total_ma"],
                    0.0,
                )
                # across all sources normalization
                _all_flow["origin_total_ma"] = _all_flow.groupby(
                    grp_keys_lst + ["snap"], group_keys=False
                )["count_ma"].transform("sum")
                _all_flow["ratio_all_ma"] = np.where(
                    _all_flow["origin_total_ma"] > 0,
                    _all_flow["count_ma"] / _all_flow["origin_total_ma"],
                    0.0,
                )
                results_flow = [
                    _all_flow[_all_flow["snap"] == snap].reset_index(drop=True)
                    for snap in sorted(_all_flow["snap"].unique())
                ]

        # compute retention
        try:
            if len(results_retention) > 0:
                _all_ret = pd.concat(results_retention, ignore_index=True)
                if all(
                    c in _all_ret.columns
                    for c in grp_keys_lst + ["snap", "stay", "left"]
                ):
                    _all_ret = _all_ret.sort_values(by=grp_keys_lst + ["snap"])
                    # rolling average counts then compute ratio
                    _all_ret["stay_ma"] = (
                        _all_ret.groupby(grp_keys_lst, group_keys=False)["stay"]
                        .rolling(window=windowsize, min_periods=10)
                        .mean()
                        .reset_index(level=list(range(len(grp_keys_lst))), drop=True)
                    )
                    _all_ret["left_ma"] = (
                        _all_ret.groupby(grp_keys_lst, group_keys=False)["left"]
                        .rolling(window=windowsize, min_periods=10)
                        .mean()
                        .reset_index(level=list(range(len(grp_keys_lst))), drop=True)
                    )
                    _all_ret["total_ma"] = _all_ret["stay_ma"] + _all_ret["left_ma"]
                    _all_ret["retention_ratio_ma"] = np.where(
                        _all_ret["total_ma"] > 0,
                        _all_ret["stay_ma"] / _all_ret["total_ma"],
                        0.0,
                    )
                    results_retention = [
                        _all_ret[_all_ret["snap"] == snap].reset_index(drop=True)
                        for snap in sorted(_all_ret["snap"].unique())
                    ]
                    results_retention = [
                        _all_ret[_all_ret["snap"] == snap].reset_index(drop=True)
                        for snap in sorted(_all_ret["snap"].unique())
                    ]
        except Exception:
            pass

        # sum of last 20 results
        # without having na
        from functools import reduce

        mean_df = reduce(
            lambda x, y: x.add(y, fill_value=0), results_df[-windowsize:]
        ).astype(float)
        mean_df = mean_df / windowsize

        # build transition matrices Px and Py (state_lst_key -> state_key)

        cols_needed = {
            "state_key",
            "state_lst_key",
            "source",
            "snap",
            "ratio_ma",
        }
        if cols_needed.issubset(_all_flow.columns):
            num_states = max(self.state_defs.state_key_range.values()) + 1
            Px = np.zeros((num_states, num_states), dtype=float)
            Py = np.zeros((num_states, num_states), dtype=float)
            last_snap = int(sorted(_all_flow["snap"].unique())[-1])
            flow_last = _all_flow[_all_flow["snap"] == last_snap]

            def _fill_matrix(df_src, M):
                dfw = df_src.copy().fillna(0.0)
                dfw = dfw.dropna(
                    subset=["state_key", "state_lst_key"]
                )  # ensure valid keys
                if len(dfw) == 0:
                    return
                # use ratio_ma directly (already conditional per origin),
                # average across counties and renormalize per origin
                grouped = dfw.groupby(["state_lst_key", "state_key"], as_index=False)[
                    "ratio_ma"
                ].mean()
                row_sum = grouped.groupby("state_lst_key")["ratio_ma"].sum().to_dict()
                for _, r in grouped.iterrows():
                    i = int(r["state_lst_key"])  # origin
                    j = int(r["state_key"])  # destination
                    s = float(row_sum.get(i, 0.0))
                    p = (float(r["ratio_ma"]) / s) if s > 0 else 0.0
                    if 0 <= i < num_states and 0 <= j < num_states:
                        M[i, j] = p

            _fill_matrix(flow_last[flow_last["source"] == "x"], Px)
            _fill_matrix(flow_last[flow_last["source"] == "y"], Py)

            # save on simulator for downstream use
            self.Px = Px
            self.Py = Py
            # also save flattened vectors; fill zeros for missing entries
            self.Px_vec = Px.flatten()
            self.Py_vec = Py.flatten()

        # also, save x, x0, lmd, lv, yout, yin, tau from mean_df
        # retention_ratio_ma from results_retention[-1]

        try:
            # map mean_df to include state_lst_key for vector placement
            _md = mean_df.reset_index()
            if "state_lst" in _md.columns:
                _md["state_lst_key"] = _md["state_lst"].map(
                    self.state_defs.state_key_range
                )
            num_states = max(self.state_defs.state_key_range.values()) + 1

            def _to_vec(column_name: str) -> np.ndarray:
                vec = np.zeros(num_states, dtype=float)
                if column_name in _md.columns and "state_lst_key" in _md.columns:
                    for _, r in _md.dropna(subset=["state_lst_key"]).iterrows():
                        k = (
                            int(r["state_lst_key"])
                            if not pd.isna(r["state_lst_key"])
                            else -1
                        )
                        if 0 <= k < num_states:
                            vec[k] = (
                                float(r[column_name])
                                if not pd.isna(r[column_name])
                                else 0.0
                            )
                return vec

            self.x0_vec = _to_vec("x0")
            self.x_vec = _to_vec("x")
            self.lmd_vec = _to_vec("lmd")
            self.lv_vec = _to_vec("lv")
            self.yout_vec = _to_vec("yout")
            self.yin_vec = _to_vec("yin")
            self.tau_vec = _to_vec("tau")
        except Exception:
            pass

        try:
            # retention vector from latest retention snapshot
            if len(results_retention) > 0:
                last_ret = results_retention[-1]
                num_states = max(self.state_defs.state_key_range.values()) + 1
                self.retention_vec = np.zeros(num_states, dtype=float)
                col = (
                    "retention_ratio_ma"
                    if "retention_ratio_ma" in last_ret.columns
                    else "retention_ratio"
                )
                if "state_lst_key" in last_ret.columns and col in last_ret.columns:
                    for _, r in last_ret.dropna(subset=["state_lst_key"]).iterrows():
                        k = (
                            int(r["state_lst_key"])
                            if not pd.isna(r["state_lst_key"])
                            else -1
                        )
                        if 0 <= k < num_states:
                            self.retention_vec[k] = max(
                                float(r[col]) if not pd.isna(r[col]) else 0.0, 0.1
                            )
        except Exception:
            pass

        return (
            mean_df.reset_index().assign(
                state_lst_key=lambda df: df["state_lst"].map(
                    self.state_defs.state_key_range
                ),
            ),
            results,
            results_df,
            results_flow,
            results_retention,
        )

    # ------------------------------------------------------------------------------
    # EVENT GENERATION FUNCTIONS
    # ------------------------------------------------------------------------------
    def get_new_recid_event(self, row, t_now, opt_verbosity=2, fo=None):
        _tau = self.sample_survival_time(row["score_with_treat"])
        e = Event(
            t_now + _tau,
            _tau,
            row.name,
            EventType.EVENT_RECID,
            idx_ref=-1,
            companion=None,
        )
        opt_verbosity >= 2 and print(self.log_event(e, gen=True), file=fo)
        if e.time > row["leaving"]:
            return None
        return e


# ------------------------------------------------------------------------------
# some default utilities
# ------------------------------------------------------------------------------
def arrival_exp(beta_arrival, row, t_now, idx, idx_ref):
    _tau = np.random.exponential(beta_arrival)
    e = Event(t_now + _tau, _tau, idx, EventType.EVENT_ARR, idx_ref=idx_ref)
    return e


def arrival_no_arrival(row, t_now, idx, idx_ref):
    return None


def leaving_exp(beta_leave, row, t_now, idx):
    _tau = np.random.exponential(beta_leave)
    e = Event(t_now + _tau, _tau, idx, EventType.EVENT_LEAVE, idx_ref=-1)
    return e


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trend(simulator: Simulator):
    # Unique counties and consistent color map
    counties = simulator.df_result_comm["code_county"].unique()
    colors = px.colors.qualitative.Plotly
    color_map = {county: colors[i % len(colors)] for i, county in enumerate(counties)}

    # Create subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[
            "Mean Community Score",
            "Mean Rearrest Time",
            "Percent Rearrest",
        ],
    )

    # Add traces using consistent colors
    for county in counties:
        df_sub = simulator.df_result_comm[
            simulator.df_result_comm["code_county"] == county
        ]
        color = color_map[county]

        fig.add_trace(
            go.Scatter(
                x=df_sub["snap"],
                y=df_sub["score_comm"],
                mode="lines",
                name=county,
                line=dict(color=color),
                legendgroup=county,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_sub["snap"],
                y=df_sub["mean_re_time"],
                mode="lines",
                name=county,
                line=dict(color=color),
                legendgroup=county,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_sub["snap"],
                y=df_sub["percent_re"],
                mode="lines",
                name=county,
                line=dict(color=color),
                legendgroup=county,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # Layout
    fig.update_layout(
        height=900,
        title_text="Community Outcomes by Episode",
        xaxis3_title="Episode (x100 days)",
        yaxis_title="Mean Community Score",
        yaxis2_title="Mean Rearrest Time",
        yaxis3_title="Percent Rearrest",
        font=dict(family="Lato"),  # Set font to Times
    )
    return fig
