import os
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import ListedColormap


rbgs = [
    (0.0, 0.6056031611752245, 0.9786801175696073),
    (0.8888735002725197, 0.4356491903481899, 0.2781229361419437),
    (0.2422242978521988, 0.6432750931576305, 0.3044486515341153),
    (0.7644401754934357, 0.44411177946877667, 0.8242975359232757),
    (0.6755439572114058, 0.5556623322045815, 0.09423433626639476),
    (4.821181565883848e-7, 0.6657589812923558, 0.6809969518707946),
    (0.930767491919665, 0.3674771896571418, 0.5757699667547833),
    (0.7769816661712935, 0.5097431319944512, 0.1464252569555494),
    (3.8077343912812365e-7, 0.6642678029460113, 0.5529508754522481),
    (0.558464964115081, 0.5934846564332881, 0.11748125233232112),
    (5.947623876556563e-7, 0.6608785231434255, 0.7981787608414301),
    (0.6096707676128643, 0.49918492100827794, 0.9117812665042643),
    (0.3800016049820355, 0.5510532724353505, 0.9665056985227145),
    (0.9421816479542178, 0.37516423354097606, 0.4518168202944591),
    (0.8684020893043973, 0.3959893639954848, 0.7135147524811882),
    (0.423146743646308, 0.6224954944199984, 0.1987706025213047),
]
juliacmaps = ListedColormap(rbgs)
# Configure rcParams to use LaTeX and serif font
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
        "image.cmap": "juliacmaps",  # Set custom colormap
    }
)
colors = juliacmaps(np.linspace(0, 1, len(rbgs)))  # Get all colors from the colormap
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)


def sum_y_and_count_groupby_N(y, cumsum):
    M, N = y.shape
    results = []

    for t in range(N):
        # Create a DataFrame for easier grouping
        df = pd.DataFrame({"y": y[:, t], "cumsum": cumsum[:, t]})

        # Group by the cumulative sum
        grouped_sum = df.groupby("cumsum")["y"].sum()
        grouped_count = df.groupby("cumsum")["y"].count()

        # Combine the sum and count into one DataFrame
        result = pd.DataFrame({"$y$": grouped_sum, "$x$": grouped_count})
        results.append(result)

    return results


def find_first(arr):
    indices = np.where(arr == 1)[0]
    index = indices[0] if indices.size > 0 else -1
    return index


# import all functions from records.py
from records import *

# import all functions from plots.py
from plots import *
