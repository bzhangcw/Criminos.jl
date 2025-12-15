"""
Simulation setup module - provides common configuration and functions for running policies.

Usage:
    from simulation_setup import default_settings, run_name, run_sim, tests

    # Run a single policy with repeats
    simulators = run_name("null", repeat=3)

    # Or run directly
    sim = run_sim("custom", func_treatment, func_treatment_kwargs, seed=1)

    # Access settings
    default_settings.p_freeze
    default_settings.treatment_capacity

Environment variables:
    DBG: 0 = formal run (default), 1 = debug mode with more arrivals
"""

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import simulation
import simulation_treatment as smt
import sirakaya
import input


class SimulationSetup:
    """Configuration class for simulation parameters."""

    # Default values
    p_length = 60
    p_freeze = 0
    T_max = 30000
    treatment_capacity = 50
    treatment_effect = 1.0
    """@note: try not change to 1
    1: vital for memory need about 5G RAM
    2: 1 will be very slow to converge (need many epochs))
    """
    beta_arrival = 2
    communities = {1}
    fit_kwargs = {
        "new_col": "felony_arrest",
        "bool_use_cph": False,
        "baseline": "exponential",
    }
    func_to_call = simulation.arrival_constant
    state_defs = simulation.StateDefs(
        ["felony_arrest", "age_dist", "bool_in_probation"]
    )

    # Treatment eligibility and enrollment settings
    max_returns = 3
    str_qualifying_for_treatment = (
        "bool_left == 0 "
        + "& bool_in_probation == 1"
        + "& bool_can_be_treated == 1"
        + "& has_been_treated == 0"
    )
    str_current_enroll = "bool_left == 0 & bool_treat == 1 & bool_in_probation == 1"
    bool_return_can_be_treated = 1
    bool_keep_treatment_effect = 1

    # Data (loaded once)
    _data_loaded = False
    dfr = None
    df = None
    df_individual = None
    df_community = None
    dfc = None
    dfi = None
    dfpop0 = None
    simulator = None

    # Debug mode from environment
    dbg = int(os.environ.get("DBG", "0"))

    def __init__(self):
        # Adjust settings based on debug mode
        if self.dbg == 1:
            self.beta_arrival = 10
            self.p_length = 250
            self.T_max = 10000

        self._load_data()
        self._print_summary()

    def _print_summary(self):
        """Print summary of main statistics and settings."""
        n_episodes = self.T_max // self.p_length
        state_dims = self.state_defs.state_keys

        print("=" * 60)
        print("SimulationSetup Summary")
        print("=" * 60)
        print(
            f"  Mode:              {'DEBUG' if self.dbg else 'FORMAL'} (DBG={self.dbg})"
        )
        print("-" * 60)
        print("Simulation Parameters:")
        print(f"  T_max:             {self.T_max:,} time units")
        print(f"  p_length:          {self.p_length} (episode length)")
        print(f"  p_freeze:          {self.p_freeze} (freeze period)")
        print(f"  n_episodes:        ~{n_episodes}")
        print("-" * 60)
        print("Treatment Settings:")
        print(f"  capacity:          {self.treatment_capacity} per episode")
        print(f"  effect:            {self.treatment_effect}")
        print(f"  max_returns:       {self.max_returns}")
        print(f"  return_can_treat:  {self.bool_return_can_be_treated}")
        print(f"  qualifying:        {self.str_qualifying_for_treatment}")
        print(f"  current_enroll:    {self.str_current_enroll}")
        print("-" * 60)
        print("Population & Arrivals:")
        print(f"  beta_arrival:      {self.beta_arrival} (arrival rate)")
        print(f"  communities:       {self.communities}")
        print(f"  initial pop:       {len(self.dfpop0)} individuals")
        print(f"  available pool:    {len(self.dfi)} individuals")
        print("-" * 60)
        print("State Space:")
        print(f"  dimensions:        {state_dims}")
        print(f"  n_states:          {len(self.state_defs.state_key_range)}")
        print("=" * 60)

    def _load_data(self, ignore_communities=True):
        """Load data and initialize simulator."""
        if SimulationSetup._data_loaded:
            return

        # Load raw data
        self.dfr, self.df, self.df_individual, self.df_community = input.load_data()

        # Create initial simulator for scoring
        self.simulator = simulation.Simulator(
            eval_score_fixed=sirakaya.eval_score_fixed,
            eval_score_comm=sirakaya.eval_score_comm,
            func_arrival=self.arrival_func,
            func_leaving=None,
            state_defs=self.state_defs,
        )
        simulation.refit_baseline(self.simulator, self.df_individual, **self.fit_kwargs)

        # Filter by communities
        self.dfc = self.df_community[
            self.df_community.index.isin(self.communities)
        ].copy()

        # if ignore_community, we change the community to {communities}
        # and then we can sample from the entire population
        if ignore_communities:
            _valid_population = self.df_individual.assign(
                code_county=list(self.communities)[0]
            )
        else:
            _valid_population = self.df_individual[
                self.df_individual["code_county"].isin(self.communities)
            ]

        self.dfi = (
            _valid_population.reset_index(drop=True)
            .copy()
            .assign(
                bool_in_probation=1,
                state=lambda x: x.apply(
                    lambda row: self.state_defs.get_state(row), axis=1
                ),
                score_state=lambda x: x.apply(
                    lambda row: self.state_defs.get_score(row), axis=1
                ),
                score=lambda x: x.apply(
                    lambda row: self.simulator.produce_score(row.name, x), axis=1
                ),
            )
        )

        # Sample initial population
        np.random.seed(123)
        self.dfpop0 = self.dfi.sample(30)

        SimulationSetup._data_loaded = True

    def arrival_func(self, *args, **kwargs):
        # Access via class to avoid descriptor binding
        func = type(self).func_to_call
        return func(self.beta_arrival, *args, **kwargs)


# Create default settings instance
default_settings = SimulationSetup()


def run_sim(
    name,
    func_treatment=None,
    func_treatment_kwargs=None,
    seed=1,
    verbosity=2,
    output_dir=None,
    settings=None,
):
    """Run a single simulation with given treatment function."""
    if settings is None:
        settings = default_settings

    simulator = simulation.Simulator(
        eval_score_fixed=sirakaya.eval_score_fixed,
        eval_score_comm=sirakaya.eval_score_comm,
        func_arrival=settings.arrival_func,
        func_leaving=None,
        state_defs=settings.state_defs,
    )
    simulation.refit_baseline(simulator, settings.df_individual, **settings.fit_kwargs)

    # Determine log path
    log_path = f"{output_dir}/log.txt" if output_dir else f"results/log.dyn.{name}.log"

    simulator.run(
        seed=seed,
        dfc=settings.dfc.copy(),
        dfi=settings.dfpop0.copy(),
        dfpop=settings.dfi.copy(),
        fo=open(log_path, "w"),
        opt_verbosity=verbosity,
        T_max=settings.T_max,
        p_length=settings.p_length,
        p_freeze=settings.p_freeze,
        treatment_effect=settings.treatment_effect,
        func_treatment=func_treatment,
        func_treatment_kwargs=func_treatment_kwargs,
        str_qualifying_for_treatment=settings.str_qualifying_for_treatment,
        str_current_enroll=settings.str_current_enroll,
        bool_return_can_be_treated=settings.bool_return_can_be_treated,
        max_returns=settings.max_returns,
    )
    if verbosity > 0:
        print(
            "people appeared/new",
            simulator.n_persons_appeared,
            simulator.n_persons_appeared - simulator.n_persons_initial,
        )
    return simulator


def run_name(name, repeat=3, start=0, output_dir="results", settings=None):
    """
    Run a policy with multiple repeats, saving each to its own directory.

    Args:
        name: policy name (must be a key in tests dict)
        repeat: number of repetitions
        start: starting index for repetition numbering (default 0)
        output_dir: base directory for outputs
        settings: SimulationSetup instance (default: default_settings)

    Returns:
        repeat: the number of repetitions (simulators are not kept in memory)
    """
    if settings is None:
        settings = default_settings

    from simulation_summary import dump_rep_metrics

    policy_tests = get_tests(settings)
    policy_dir = f"{output_dir}/{name}"
    os.makedirs(policy_dir, exist_ok=True)

    for k in tqdm(range(start, start + repeat), desc=f"Running {name}"):
        rep_dir = f"{policy_dir}/{k}"
        os.makedirs(rep_dir, exist_ok=True)

        sim = run_sim(
            name,
            policy_tests[name][0],
            policy_tests[name][1],
            seed=k,
            verbosity=2,
            output_dir=rep_dir,
            settings=settings,
        )
        dump_rep_metrics(sim, rep_dir, settings.p_freeze)
        # sim goes out of scope and can be garbage collected

    return repeat


# ------------------------------------------------------------
# define policy configurations to test
# ------------------------------------------------------------
def create_lowj_lowa_prob_vector(j_max=4, a_max=2):
    pr = {}
    for j in range(j_max):
        for a in range(a_max):
            pr[(j, a, 1.0)] = 1.0
    return pr


def create_threshj_lowa_prob_vector(j_min=2, a_max=2, j_max=20):
    pr = {}
    for j in range(j_min, j_max):
        for a in range(a_max):
            pr[(j, a, 1.0)] = 1.0
    return pr


pr_lowj_lowa = create_lowj_lowa_prob_vector()
pr_threshj_lowa = create_threshj_lowa_prob_vector()
thres_cutoff = {
    1: 0,  # if in age-group 1, cut-off below 1
    2: 1,  # if in age-group 2, cut-off below 2
    3: 2,  # if in age-group 3, cut-off below 3
    4: 3,
    5: 4,
    6: 10,
}


# ------------------------------------------------------------
# get test configurations
# ------------------------------------------------------------
def get_tests(settings=None):
    """Return test configurations using given settings."""
    if settings is None:
        settings = default_settings

    return {
        "null": (smt.treatment_null, dict()),
        "random": (
            smt.treatment_rule_random,
            dict(capacity=settings.treatment_capacity),
        ),
        "high-risk": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=False,
            ),
        ),
        "low-risk": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=True,
            ),
        ),
        # "low-age-high-prev": (
        #     smt.treatment_rule_priority,
        #     dict(
        #         key="_new_prior",
        #         capacity=settings.treatment_capacity,
        #         ascending=True,
        #         effect=settings.treatment_effect,
        #         to_compute=lambda df: df.apply(
        #             lambda row: (row["age_dist"], -row["felony_arrest"]), axis=1
        #         ),
        #     ),
        # ),
        # "low-age-low-prev": (
        #     smt.treatment_rule_priority,
        #     dict(
        #         key="_new_prior",
        #         capacity=settings.treatment_capacity,
        #         ascending=True,
        #         effect=settings.treatment_effect,
        #         to_compute=lambda df: df.apply(
        #             lambda row: (row["age_dist"], row["felony_arrest"]), axis=1
        #         ),
        #     ),
        # ),
        "high-risk-only-young": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=False,
                exclude=lambda row: row["age_dist"] >= 3,
            ),
        ),
        "age-tolerance": (
            smt.treatment_rule_priority,
            dict(
                key="score",
                capacity=settings.treatment_capacity,
                ascending=False,
                exclude=lambda row: row["felony_arrest"]
                < thres_cutoff[row["age_dist"]],
            ),
        ),
        # ------------------------------------------------------------
        # fluid/probabilistic policies
        # ------------------------------------------------------------
        "fluid-low-age-low-prev": (
            smt.treatment_rule_priority_fluid,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                prob_vector=pr_lowj_lowa,
            ),
        ),
        "fluid-low-age-threshold-offenses": (
            smt.treatment_rule_priority_fluid,
            dict(
                key="_new_prior",
                capacity=settings.treatment_capacity,
                prob_vector=pr_threshj_lowa,
            ),
        ),
    }


# Default tests using default_settings
tests = get_tests()
