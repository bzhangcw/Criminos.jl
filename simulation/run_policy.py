import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns

import json
import plotly
from tqdm import tqdm

import simulation, sirakaya
from simulation_setup import *
from simulation_summary import *

# python run_policy.py low_age_low_prev 3
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_policy.py <policy_name> <repeat>")
        # show all policy names
        print("Available policies:")
        for policy in tests.keys():
            print(f"- {policy}")
        sys.exit(1)
    policy_name = sys.argv[1]
    repeat = int(sys.argv[2])
    _sim = run_name(policy_name, repeat=repeat)
    dump_metrics_to_h5(policy_name, _sim, p_freeze)
