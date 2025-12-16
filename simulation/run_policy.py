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
    if len(sys.argv) != 5:
        print(
            "Usage: python run_policy.py <policy_name> <repeat> <repeat_start> <output_dir>"
        )
        # show all policy names
        print("Available policies:")
        for policy in tests.keys():
            print(f"- {policy}")
        sys.exit(1)
    policy_name = sys.argv[1]
    repeat = int(sys.argv[2])
    repeat_start = int(sys.argv[3])
    output_dir = sys.argv[4]
    run_name(policy_name, repeat=repeat, start=repeat_start, output_dir=output_dir)
