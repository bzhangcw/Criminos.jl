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
    EVENT_INCARCERATION = "incar"
    EVENT_ADMIT = "admit"
    EVENT_REVOKE = "revok"


class ReasonToLeave(IntEnum):
    END_OF_PROCESS = 1
    RETURNING_AGENT = 2
    INCARCERATION = 3
    TOO_MANY_RETURNS = 4


from simulation_regression import *
from simulation_treatment import (
    treatment_null,
    treatment_rule_priority,
    treatment_rule_random,
)
from simulation_stats import (
    summarize_trajectory,
    evaluation_metrics,
    recover_treatment_decision_as_df,
)
import simulation_treatment as simt
import simulation_stats as sims


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
    "offenses": list(range(11)),
    "age_dist": [1, 2, 3, 4, 5, 6],  # ignore 0
    "has_been_treated": [0, 1],
    "stage": ["p", "f"],
}
DEFAULT_SCORING_STATE_PARAMS = {
    "score_fixed": 0.7904,
    "score_comm": 0.7904,
    "score_age_dist": 0.7904,
    "score_offenses": 0.1884,
}


class StateDefs(object):

    def __init__(self, state_keys, weights=DEFAULT_SCORING_STATE_PARAMS):
        """
        Parameters
        ----------
        state_keys : list of str
            The keys of the state variables.
        weights : dict
            The weights of the state variables;
            this tells us how to score the state variables.
            some of the keys may not be used in the scoring.
        """
        self.state_keys = state_keys
        self.scoring_weights = weights
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
        # Convert numeric columns to float, but keep string columns as-is
        state_values = row[self.state_keys]
        # Try to convert to float where possible, but keep original if it fails
        converted = []
        for val in state_values:
            try:
                converted.append(float(val))
            except (ValueError, TypeError):
                converted.append(val)
        return StateVal(pd.Series(converted))

    def get_score(self, row):

        return sum(
            (
                row[f"score_{key}"] * self.scoring_weights[f"score_{key}"]
                if f"score_{key}" in self.scoring_weights
                else 0.0
            )
            for key in self.state_keys
        )

    def update_state(self, dfi, idx, t):
        # ------------------------------------------------------------
        # update the individual covariates and scores, and finally the state and score_state
        # ------------------------------------------------------------
        # update the age, age_dist, and score_age_dist according to the new age
        self.update_covariates(dfi, idx, t)
        # update the state and score_state
        dfi.loc[idx, "state"] = self.get_state(dfi.loc[idx])
        dfi.loc[idx, "score_state"] = self.get_score(dfi.loc[idx])

    def update_covariates(self, dfi, idx, t):
        """
        Update the covariates and the corresponding scores of the individual.
        This includes:
        1. age, age_dist, and score_age_dist.
        2. offenses, and score_offenses.
        """
        dfi.loc[idx, "age"] = (
            dfi.loc[idx, "age_start"] + (t - dfi.loc[idx, "arrival"]) / 365.0
        )
        dfi.loc[idx, "age_dist"] = sirakaya.age_to_age_dist(dfi.loc[idx, "age"])
        # @note: c.z (2025-12-17). previously we used score_age_dist to score the age_dist.
        # but now we use the age to score the age_dist.
        # dfi.loc[idx, "score_age_dist"] = sirakaya.score_age_dist(
        #     dfi.loc[idx, "age_dist"]
        # )
        dfi.loc[idx, "score_age_dist"] = sirakaya.score_age(dfi.loc[idx, "age"])
        # the score of offenses is just itself
        dfi.loc[idx, "score_offenses"] = dfi.loc[idx, "offenses"]


class Simulator(object):

    # these columns are initialized for each individual when the simulation starts
    INITIALIZED_COLUMNS = [
        "acc_survival",  # accumulated survival time(s)
        "bool_left",  # whether the individual has left (end of probation term)
        "type_left",  # the reason for leaving (end of probation term / too many arrests)
        "observed",  # whether the individual has been observed (ever arrested)
        "treat_start",  # the time when the treatment started
        "treat_end",  # the time when the treatment ended
        "bool_treat",  # whether the individual is in treatment
        "bool_treat_made",  # whether the treatment decision (either 0 or 1) has been made
        "ep_arrival",  # episode of arrival
        "ep_leaving",  # episode of leaving
        "return_times",  # number of times the individual has returned to the community
    ]

    def __init__(
        self,
        eval_score_fixed,
        eval_score_comm,
        func_arrival=None,
        func_end_probation=None,
        func_leaving=None,
        state_defs=StateDefs(["offenses"]),
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

    def __default_incarceration(self, row, t_now):
        return Event(
            t_now + 0.0,
            0.0,
            row.name,
            EventType.EVENT_INCARCERATION,
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
        treatment_effect=0.5,
        func_treatment=None,
        func_treatment_kwargs={},
        # ----------------------------------------------------------------------
        str_qualifying_for_treatment="",
        str_current_enroll="",
        max_returns=3,
        max_offenses=15,
        bool_return_can_be_treated=1,
        bool_has_incarceration=True,
        prison_rate_scaler=0.01,
        length_scaler=1.0,
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
        - num_n_mean_offenses: Mean arrest statistics
        ...
        ----------------------------------------------------------------------
        """
        np.random.seed(seed)
        # double check the scoring weights
        print("=" * 60)
        print(
            f"activated state weights: (produce state score)\n{json.dumps(self.state_defs.scoring_weights, indent=2)}"
        )
        print(
            f"activated final weights:\n{json.dumps(self._scoring_weights, indent=2)}"
        )
        print("=" * 60)
        # create an empty priority queue
        self.event_queue = event_queue = queue.PriorityQueue()
        self.events_cancelled = dict()
        t = 0.0
        p = p_freeze
        n_persons_appeared = dfi.shape[0]
        self.n_persons_initial = n_persons_appeared
        self.person_id_max = person_id_max = dfi.index.max()
        self.dfpop = dfpop
        # initialize the population
        self.dfi = dfi
        # some 0 columns to be initialized
        dfi[self.INITIALIZED_COLUMNS] = 0.0
        # other columns
        dfi["hash_leave"] = ""
        dfi["arrival"] = 0.0
        dfi["rel_off_probation"] = rel_off_probation  # initial off-probation term
        dfi["ep_leaving"] = 1e6
        dfi["ep_end_probation"] = 1e6
        dfi["age_start"] = dfi["age"].copy()
        dfi["age_dist_start"] = dfi["age_dist"].copy()
        dfi["prio_offenses"] = dfi["offenses"].copy()
        dfi["stage"] = "p"
        dfi["score_stage"] = 1.0  # 1.0 for probation stage
        dfi["state_lst"] = dfi["state"].apply(lambda x: StateVal(x.value))
        dfi["bool_can_be_treated"] = 1

        # statistics
        self.num_y = num_y = defaultdict(int)
        self.num_lbd = num_lbd = defaultdict(int)
        self.num_lft = num_lft = defaultdict(int)
        self.num_flow = num_flow = defaultdict(int)

        # used to update the community score
        self.num_n_y = num_n_y = defaultdict(float)
        self.num_n_time = num_n_time = defaultdict(float)
        self.num_n_mu = num_n_mu = defaultdict(float)
        self.num_n_mean_offenses = num_n_mean_offenses = defaultdict(float)
        self.num_n_enrollment = num_n_enrollment = dict()

        self.t_dfc = t_dfc = []
        self.t_dfi = t_dfi = []
        self.t_dftau = t_dftau = []
        self.traj_results = []

        # initialize the community score
        dfc_dict = dfc["score_comm"].to_dict()

        # county id
        self.id_cc = id_cc = dfc.index.to_list()

        opt_verbosity >= 1 and print("event:happ/  ready to go@", file=fo)
        print(self.log_header(), file=fo, flush=True)

        # --------------------------------------------------------------------------
        # assign initial treatment
        # @note: c.z (2025-12-11)
        # do not assign treatment initially
        # idx_selected = func_treatment(
        #     dfi,
        #     **func_treatment_kwargs,
        #     str_qualifying=str_qualifying_for_treatment,
        #     str_current_enroll=str_current_enroll,
        # )
        # --------------------------------------------------------------------------
        # assign end-probation & leaving events to existing individuals
        # --------------------------------------------------------------------------
        for idx, row in dfi.iterrows():
            # end-probation event
            _end_probation = self.func_end_probation(dfi.loc[idx], t)
            event_queue.put((_end_probation.priority, _end_probation))
            dfi.loc[idx, "end_probation"] = _end_probation.time
            dfi.loc[idx, "ep_end_probation"] = 1e6
            dfi.loc[idx, "hash_end_probation"] = _end_probation.hashid
            opt_verbosity >= 2 and print(self.log_event(_end_probation), file=fo)

        for idx, row in dfi.iterrows():
            # leaving event
            _leave_event = self.func_leaving(row, t)
            dfi.loc[idx, "leaving"] = _leave_event.time
            dfi.loc[idx, "ep_leaving"] = 1e6
            dfi.loc[idx, "hash_leave"] = _leave_event.hashid
            event_queue.put((_leave_event.priority, _leave_event))
            opt_verbosity >= 2 and print(self.log_event(_leave_event), file=fo)

        # --------------------------------------------------------------------------
        # initialize a recidivism event if it is before the leaving event
        # --------------------------------------------------------------------------
        for idx, row in dfi.iterrows():
            _recid_event = self.get_new_recid_event(
                row, t_now=0.0, opt_verbosity=0, fo=fo
            )
            if _recid_event is not None:
                event_queue.put((_recid_event.priority, _recid_event))
                opt_verbosity >= 2 and print(self.log_event(_recid_event), file=fo)

        # push one arrival event
        if self.func_arrival is not None:
            _row = dfpop.sample(1, weights="weight").iloc[0]
            # arrival event
            _arrival = self.func_arrival(
                _row,
                t,
                person_id_max + 1,  # index of the new person
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
            _cancel_by_who = self.events_cancelled.get(event.hashid)
            if np.isnan(tau) or _cancel_by_who is not None:
                opt_verbosity >= 2 and print(
                    self.log_event(event, cancel=_cancel_by_who), file=fo
                )
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
                    person_id_max += 1
                    # Batch copy row data
                    dfi.loc[idx] = dfpop.loc[event.idx_ref]
                    # Batch assign multiple columns at once
                    age_val = dfi.at[idx, "age"]
                    age_dist_val = dfi.at[idx, "age_dist"]
                    offenses_val = dfi.at[idx, "offenses"]

                    # Use .at for single value assignments (faster than .loc)
                    dfi.at[idx, "bool_can_be_treated"] = 1
                    dfi.at[idx, "age_start"] = age_val
                    dfi.at[idx, "age_dist_start"] = age_dist_val
                    dfi.at[idx, "prio_offenses"] = offenses_val
                    dfi.at[idx, "has_been_treated"] = 0
                    dfi.at[idx, "score_has_been_treated"] = 0.0
                    # Batch assign initialized columns
                    dfi.loc[idx, self.INITIALIZED_COLUMNS] = 0
                else:
                    # return can be treated?
                    dfi.at[idx, "bool_can_be_treated"] = bool_return_can_be_treated

                # Batch assign common columns
                dfi.at[idx, "arrival"] = t
                dfi.at[idx, "ep_arrival"] = _episode
                dfi.at[idx, "stage"] = "p"
                dfi.at[idx, "score_stage"] = 1.0  # 1.0 for probation stage

                # arrival / return, then start probation immediately
                # add a perturbation to the leaving time
                perturbation = np.random.uniform(0.7, 1.3) * length_scaler
                dfi.at[idx, "rel_probation"] = (
                    dfi.at[idx, "rel_probation"] * perturbation
                )
                dfi.at[idx, "rel_off_probation"] = rel_off_probation * length_scaler

                # Cache row for multiple accesses
                row_data = dfi.loc[idx]

                # update the state and scores
                state_val = self.state_defs.get_state(row_data)
                dfi.at[idx, "state"] = state_val
                dfi.at[idx, "state_lst"] = StateVal(state_val.value)
                dfi.at[idx, "score_state"] = self.state_defs.get_score(row_data)
                dfi.at[idx, "score_comm"] = dfc_dict.get(row_data["code_county"])
                dfi.at[idx, "score"] = self.produce_score(idx, dfi)

                # ------------------------------------------------------------
                # produce the end-probation event
                # ------------------------------------------------------------
                _end_probation = self.func_end_probation(row_data, t)
                opt_verbosity >= 2 and print(
                    self.log_event(_end_probation, gen=True), file=fo
                )
                event_queue.put((_end_probation.priority, _end_probation))
                dfi.at[idx, "end_probation"] = _end_probation.time
                dfi.at[idx, "ep_end_probation"] = 1e6
                # ------------------------------------------------------------
                # produce the leaving event
                # ------------------------------------------------------------
                _leaving = self.func_leaving(row_data, t)
                opt_verbosity >= 2 and print(
                    self.log_event(_leaving, gen=True), file=fo
                )
                event_queue.put((_leaving.priority, _leaving))
                dfi.at[idx, "leaving"] = _leaving.time
                dfi.at[idx, "ep_leaving"] = 1e6
                dfi.at[idx, "hash_leave"] = _leaving.hashid

                # generate next arrival individual
                if info == EventType.EVENT_ARR:
                    _new_row = dfpop.sample(1).iloc[0]
                    _arrival = self.func_arrival(
                        _new_row,
                        t,
                        person_id_max + 1,
                        _new_row.name,
                    )
                    if _arrival is not None:
                        opt_verbosity >= 2 and print(
                            self.log_event(_arrival, gen=True), file=fo
                        )
                        event_queue.put((_arrival.priority, _arrival))

                        num_lbd[
                            _row["code_county"], dfi.loc[idx, "state"], _episode
                        ] += 1

            elif info == EventType.EVENT_END_PROD:
                # process leaving events
                dfi.at[idx, "ep_end_probation"] = _episode
                dfi.at[idx, "stage"] = "f"
                dfi.at[idx, "score_stage"] = 0.0  # 0.0 for off-probation stage
                _revoke_event = revert_treatment(dfi, idx, t)
                if _revoke_event is not None:
                    opt_verbosity >= 2 and print(
                        self.log_event(_revoke_event, gen=False), file=fo
                    )

            elif info == EventType.EVENT_LEAVE:
                # process leaving events
                dfi.at[idx, "bool_left"] = _bool_skip_next_offense = 1
                dfi.at[idx, "ep_leaving"] = _episode
                dfi.at[idx, "type_left"] = ReasonToLeave.END_OF_PROCESS
                num_lft[_row["code_county"], _row["state"], _episode] += 1

            elif info == EventType.EVENT_INCARCERATION:
                # process incarceration events
                dfi.at[idx, "bool_left"] = _bool_skip_next_offense = 1
                dfi.at[idx, "ep_leaving"] = _episode
                dfi.at[idx, "type_left"] = ReasonToLeave.INCARCERATION
                num_lft[_row["code_county"], _row["state"], _episode] += 1

            elif info == EventType.EVENT_RECID:
                _row = dfi.loc[idx]
                # ------------------------------------------------------------
                dfi.at[idx, "observed"] = 1
                dfi.at[idx, "ep_lastre"] = _episode
                dfi.at[idx, "acc_survival"] = dfi.at[idx, "acc_survival"] + tau

                # Cache frequently accessed values
                code_county = _row["code_county"]
                state = _row["state"]

                # accumulate the number of arrests
                num_n_y[code_county] += 1
                num_n_time[code_county] += tau

                # per-episode number of arrests
                num_y[code_county, state, _episode] += 1
                current_offenses = dfi.at[idx, "offenses"] + 1
                if current_offenses > max_offenses:
                    current_offenses = max_offenses
                dfi.at[idx, "offenses"] = current_offenses

                # ------------------------------------------------------------
                # an incarceration event?
                # ------------------------------------------------------------
                stage = dfi.at[idx, "stage"]
                _bool_off_probation = stage == "f"
                prison_rate = dfi.at[idx, "prison_rate"]
                _bool_is_incarceration_event = (prison_rate > 0) and (
                    np.random.uniform(0, 1) < prison_rate * prison_rate_scaler
                )
                _bool_off_probation_and_returned_too_many_times = (
                    _bool_off_probation and dfi.at[idx, "return_times"] >= max_returns
                )
                _bool_should_be_incarcerated = (
                    _bool_is_incarceration_event
                    or _bool_off_probation_and_returned_too_many_times
                )
                if _bool_should_be_incarcerated and bool_has_incarceration:
                    _incarceration = self.__default_incarceration(dfi.loc[idx], t)
                    opt_verbosity >= 2 and print(
                        self.log_event(_incarceration, gen=True), file=fo
                    )
                    event_queue.put((_incarceration.priority, _incarceration))
                    # cancel prescheduled leaving event
                    self.events_cancelled[dfi.loc[idx, "hash_leave"]] = (
                        _incarceration.hashid
                    )
                    self.events_cancelled[dfi.loc[idx, "hash_end_probation"]] = (
                        _incarceration.hashid
                    )
                    _bool_skip_next_offense = True
                    _revoke_event = revert_treatment(dfi, idx, t)
                    if _revoke_event is not None:
                        opt_verbosity >= 2 and print(
                            self.log_event(_revoke_event, gen=False), file=fo
                        )

                elif _bool_off_probation:
                    # this person is returning to the community (again)
                    dfi.at[idx, "return_times"] = dfi.at[idx, "return_times"] + 1
                    _return = self.__default_return(_row, t)
                    opt_verbosity >= 2 and print(
                        self.log_event(_return, gen=False), file=fo
                    )
                    event_queue.put((_return.priority, _return))

                    # cancel prescheduled leaving event
                    self.events_cancelled[dfi.at[idx, "hash_leave"]] = _return.hashid
                    _bool_skip_next_offense = True
                    _revoke_event = revert_treatment(dfi, idx, t)
                    if _revoke_event is not None:
                        opt_verbosity >= 2 and print(
                            self.log_event(_revoke_event, gen=False), file=fo
                        )

            else:
                raise ValueError(f"Unknown event type: {info}")

            # ------------------------------------------------------------
            # sample next survival time
            # ------------------------------------------------------------
            if _bool_skip_next_offense:
                # does not seem necessary to drop the individual
                # dfi.drop(index=idx, inplace=True)
                pass

            elif info in (
                EventType.EVENT_ARR,
                EventType.EVENT_RECID,
                EventType.EVENT_RETURN,
            ):
                _new_event = self.get_new_recid_event(
                    dfi.loc[idx], t_now=t, opt_verbosity=opt_verbosity, fo=fo
                )
                if _new_event is not None:
                    event_queue.put((_new_event.priority, _new_event))

            # ------------------------------------------------------------
            # update the community score;
            # synchronize the individual score
            # synchronize the states with the community score
            # ------------------------------------------------------------
            # if episode is upgraded;
            #  then update the score of the community
            if _episode > p:
                opt_verbosity >= 1 and print(
                    f"event:happ/\t{_episode}: update community score", file=fo
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
                    # mean_re_time = (accummulated) total_time / (number_of_offensess + 1e-10)
                    dfc.loc[_cid, "mean_re_time"] = num_n_mean_offenses[
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

                # ------------------------------------------------------------------------------
                # apply treatment rule
                # @note: this make the score_treatment column to be non-zero
                # ------------------------------------------------------------------------------
                idx_selected = func_treatment(
                    dfi,
                    # @note: by c.z (2025-09-10)
                    # actually not needed because we will mark the decision as made;
                    # will not be able to enter the pool again...
                    str_qualifying=str_qualifying_for_treatment,
                    str_current_enroll=str_current_enroll,
                    **func_treatment_kwargs,
                    t=t,
                )
                # Apply treatment effect to scores
                # Since S(t|score) = S0(t)^exp(score) and higher score means more dangerous:
                # - Higher score → exp(score) larger → S(t) smaller (since 0<S0<1) → higher offense
                # To reduce offense, we need to lower the score:
                # If treatment_effect = 0.5: score + log(0.5) = score - 0.693
                # → Lower score → Higher survival → Lower offense ✓
                dfi["bool_treat_made"] = 1
                for idx in idx_selected:
                    _admit_event = admit_treatment(dfi, idx, t, treatment_effect)
                    # and we have to update the state again and score for these people
                    self.state_defs.update_state(dfi, idx, t)
                    dfi.loc[idx, "score"] = self.produce_score(idx, dfi)
                    opt_verbosity >= 2 and print(
                        self.log_event(_admit_event, gen=False),
                        file=fo,
                    )

                current_enrollment = dfi.query(str_current_enroll).shape[0]
                self.num_n_enrollment = {
                    p: current_enrollment,
                    **self.num_n_enrollment,
                }

                if self.bool_keep_full_trajectory:
                    # keep trajectory
                    dfc_sorted = dfc.assign(snap=p).reset_index()
                    dfi_sorted = dfi.assign(snap=p).reset_index()
                    t_dfc.append(dfc_sorted)
                    t_dfi.append(dfi_sorted.reindex(sorted(dfi_sorted.columns), axis=1))
                    t_dftau.append(
                        dfi.loc[idx_selected, sorted(dfi.columns)]
                        .assign(snap=p)
                        .reset_index()
                    )
                # if self.bool_keep_full_trajectory:
                #     # keep trajectory - store individuals still in system OR leaving this period
                #     # Need to include:
                #     # 1. Active individuals: (ep_arrival <= _boundary) & (ep_leaving > _boundary)
                #     # 2. People leaving NOW: (ep_leaving == _boundary) - needed for departure counts
                #     # Combined: (ep_arrival <= _boundary) & (ep_leaving >= _boundary)
                #     _boundary = p + p_freeze
                #     _active_mask = (dfi["ep_arrival"] <= _boundary) & (
                #         dfi["ep_leaving"] >= _boundary
                #     )
                #     dfc_sorted = dfc.assign(snap=p).reset_index()
                #     # Only store active individuals (not yet left)
                #     dfi_active = dfi[_active_mask]
                #     dfi_sorted = dfi_active.assign(snap=p).reset_index()
                #     t_dfc.append(dfc_sorted)
                #     t_dfi.append(dfi_sorted.reindex(sorted(dfi_sorted.columns), axis=1))
                #     # For treatment trajectory, only include selected individuals who are active
                #     if len(idx_selected) > 0:
                #         _selected_active = [
                #             i for i in idx_selected if i in dfi_active.index
                #         ]
                #         if len(_selected_active) > 0:
                #             t_dftau.append(
                #                 dfi_active.loc[_selected_active, sorted(dfi.columns)]
                #                 .assign(snap=p)
                #                 .reset_index()
                #             )
                #         else:
                #             # No selected individuals are active
                #             t_dftau.append(
                #                 pd.DataFrame(columns=list(dfi.columns) + ["snap"])
                #             )
                #     else:
                #         # No individuals selected for treatment
                #         t_dftau.append(
                #             pd.DataFrame(columns=list(dfi.columns) + ["snap"])
                #         )

                p = _episode

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

    def log_event(self, event, gen=False, cancel=None):
        priority, _ac, tau, idx, info = event.unpack()
        _agg_id = f"{idx}".zfill(10) + f"^({event.idx_ref})"
        if gen:
            return (
                "   |-:gene/   "
                + f"{info:>4s} {_ac:10.2f} {tau:10.2f} {_agg_id:>18s} {priority:10d}"
            )
        elif cancel is not None:
            return (
                "event:canc/   "
                + f"{info:>4s} {_ac:10.2f} {tau:10.2f} {_agg_id:>18s} {priority:10d} cancelled by {cancel}"
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
        self.fig_trend = plot_community_trend(self)

    def summarize_trajectory(self, *args, **kwargs):
        return summarize_trajectory(self, *args, **kwargs)

    def __save_df_as_excel(self, df, name):
        # save the full result to excel
        # convert DataFrame to 2D list with header
        data = [df.columns.tolist()] + df.values.tolist()

        # write to excel via pyexcelerate
        wb = Workbook()
        wb.new_sheet("Sheet1", data=data)
        wb.save(f"{name}.xlsx")
        print(f"{name}.xlsx successfully saved...")

    # ------------------------------------------------------------------------------
    # EVENT GENERATION FUNCTIONS
    # ------------------------------------------------------------------------------
    def get_new_recid_event(self, row, t_now, opt_verbosity=2, fo=None):
        _tau = self.sample_survival_time(row["score"])
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
def admit_treatment(dfi, idx, t, treatment_effect, **kwargs):
    dfi.loc[idx, "bool_treat"] = 1
    dfi.loc[idx, "has_been_treated"] = 1
    dfi.loc[idx, "score_treatment"] = np.log(treatment_effect)
    # Apply treatment to selected
    dfi.loc[idx, "treat_start"] = t
    return Event(t, 0.0, idx, EventType.EVENT_ADMIT, idx_ref=-1)


def revert_treatment(dfi, idx, t, bool_keep_treatment_effect=True, **kwargs):
    is_treated = dfi.loc[idx, "bool_treat"]
    dfi.loc[idx, "treat_end"] = t if is_treated == 1.0 else 0.0
    dfi.loc[idx, "bool_treat"] = 0.0
    if not bool_keep_treatment_effect:
        dfi.loc[idx, "score_treatment"] = 0.0
    if is_treated == 1.0:
        return Event(t, 0.0, idx, EventType.EVENT_REVOKE, idx_ref=-1)
    return None


def arrival_exp(beta_arrival, row, t_now, idx, idx_ref):
    _tau = np.random.exponential(beta_arrival)
    e = Event(t_now + _tau, _tau, idx, EventType.EVENT_ARR, idx_ref=idx_ref)
    return e


def arrival_constant(beta_arrival, row, t_now, idx, idx_ref):
    _tau = beta_arrival
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


def plot_community_trend(simulator: Simulator):
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
