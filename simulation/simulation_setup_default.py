"""
Simulation setup module - provides common configuration and functions for running policies.

Usage:
    from simulation_setup import run_name, run_sim, tests, p_freeze

    # Run a single policy with repeats
    simulators = run_name("null", repeat=3)

    # Or run directly
    sim = run_sim("custom", func_treatment, func_treatment_kwargs, seed=1)

Environment variables:
    DBG: 0 = formal run (default), 1 = debug mode with more arrivals
"""

import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import simulation
import sirakaya
import input

# Debug mode from environment
dbg = int(os.environ.get("DBG", "0"))
print(f"[simulation_setup] DBG Mode: {dbg}")

# values to be used in the simulation
beta_arrival = 50
p_length = 200
p_freeze = 5
T_max = 20000
treatment_capacity = 50
treatment_effect = 0.5
state_defs = simulation.StateDefs(["felony_arrest", "age_dist", "bool_in_probation"])
communities = {1}
# define kwargs for refit the survival
fit_kwargs = {
    "new_col": "felony_arrest",
    "bool_use_cph": False,
    "baseline": "exponential",
}
func_to_call = simulation.arrival_constant


def arrival_func(*args, **kwargs):
    return func_to_call(beta_arrival, *args, **kwargs)


# ----------------------------------------------------------
# this code snippet will produce
# - dfr: original dataset with a few auxillary computations
# - df: dfr after filling the NA values
# - df_community: community-level dataset
# - df_individual: individual-level dataset
#       (merge community-level information)
# ----------------------------------------------------------
import input

dfr, df, df_individual, df_community = input.load_data()

simulator = simulation.Simulator(
    eval_score_fixed=sirakaya.eval_score_fixed,
    eval_score_comm=sirakaya.eval_score_comm,
    func_arrival=arrival_func,
    # func_arrival=simulation.arrival_no_arrival,
    func_leaving=None,  # leave by end of probation
    state_defs=state_defs,
)

simulation.refit_baseline(simulator, df_individual, **fit_kwargs)

dfc = df_community[df_community.index.isin(communities)].copy()
dfi = (
    df_individual[df_individual["code_county"].isin(communities)]
    .reset_index(drop=True)
    .copy()
    .assign(
        bool_in_probation=1,
        state=lambda x: x.apply(lambda row: state_defs.get_state(row), axis=1),
        score_state=lambda x: x.apply(lambda row: state_defs.get_score(row), axis=1),
        score=lambda x: x.apply(
            lambda row: simulator.produce_score(row.name, x), axis=1
        ),
    )
)

np.random.seed(123)
if dbg == 0:
    # formal run
    beta_arrival = 1
    dfpop0 = dfi.sample(20)
elif dbg == 1:
    beta_arrival = 200
    dfpop0 = dfi.sample(50)

print(f"pop at start: {dfpop0.shape}, beta_arrival: {beta_arrival}")


def run_sim(name, func_treatment=None, func_treatment_kwargs=None, seed=1, verbosity=0):

    simulator = simulation.Simulator(
        eval_score_fixed=sirakaya.eval_score_fixed,
        eval_score_comm=sirakaya.eval_score_comm,
        func_arrival=arrival_func,
        # func_arrival=simulation.arrival_no_arrival,
        func_leaving=None,  # leave by end of probation
        state_defs=state_defs,
    )
    simulation.refit_baseline(simulator, df_individual, **fit_kwargs)

    simulator.run(
        seed=seed,
        dfc=dfc.copy(),
        dfi=dfpop0.copy(),
        dfpop=dfi.copy(),
        fo=open(f"results/log.dyn.{name}.log", "w"),
        opt_verbosity=verbosity,
        T_max=T_max,
        p_length=p_length,
        p_freeze=p_freeze,
        treatment_effect=treatment_effect,
        func_treatment=func_treatment,
        func_treatment_kwargs=func_treatment_kwargs,
    )
    print(
        "people appeared/new",
        simulator.n_persons_appeared,
        simulator.n_persons_appeared - simulator.n_persons_initial,
    )
    simulator.summarize(save=False)
    return simulator


tests = {
    "null": (simulation.treatment_null, dict()),
    "random": (
        simulation.treatment_rule_random,
        dict(capacity=treatment_capacity),
    ),
    "high_risk": (
        simulation.treatment_rule_priority,
        dict(
            key="score",
            capacity=treatment_capacity,
            ascending=False,
        ),
    ),
    "low_risk": (
        simulation.treatment_rule_priority,
        dict(
            key="score",
            capacity=treatment_capacity,
            ascending=True,
        ),
    ),
    "low_age_high_prev": (
        simulation.treatment_rule_priority,
        dict(
            key="_new_prior",
            capacity=treatment_capacity,
            ascending=True,
            to_compute=lambda df: df.apply(
                lambda row: (row["age_dist"], -row["felony_arrest"]), axis=1
            ),
        ),
    ),
    "low_age_low_prev": (
        simulation.treatment_rule_priority,
        dict(
            key="_new_prior",
            capacity=treatment_capacity,
            ascending=True,
            to_compute=lambda df: df.apply(
                lambda row: (row["age_dist"], row["felony_arrest"]), axis=1
            ),
        ),
    ),
    "high_prev_low_age": (
        simulation.treatment_rule_priority,
        dict(
            key="_new_prior",
            capacity=treatment_capacity,
            ascending=True,
            to_compute=lambda df: df.apply(
                lambda row: (-row["felony_arrest"], row["age_dist"]), axis=1
            ),
        ),
    ),
}


def run_name(name, repeat=3):
    """Run a policy with multiple repeats, showing progress with tqdm."""
    simulators = []
    for k in tqdm(range(repeat), desc=f"Running {name}"):
        sim = run_sim(name, tests[name][0], tests[name][1], seed=k, verbosity=2)
        simulators.append(sim)
    return simulators
