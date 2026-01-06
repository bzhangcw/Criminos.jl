import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

import simulation
import sirakaya

import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# Default font for all plots (HTML and PDF)
# ------------------------------------------------------------

# Configure matplotlib font globally
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
    }
)

# ------------------------------------------------------------
# Policy color mapping for consistent colors across figures
# (matches policy names from get_tests() in simulation_setup.py)
# ------------------------------------------------------------
POLICY_COLORS = {
    "null": "#7f7f7f",  # gray
    "random": "#9467bd",  # purple
    # ------------------------------------------------------------
    # priority policies; will exhaust the capacity
    # ------------------------------------------------------------
    "high-risk": "#d62728",  # red
    "low-risk": "#546de5",  # green
    "high-risk-cutoff": "#e377c2",  # pink
    "high-risk-only-young": "#e377c2",  # pink
    "high-risk-lean-young": "#2ca02c",  # green
    "age-tolerance": "#17becf",  # cyan
    "age-first": "#51a3e6",  # soft blue
    "age-first-high-risk": "#e67051",  # soft red
    # ------------------------------------------------------------
    # fluid policies; may not exhaust the capacity
    # ------------------------------------------------------------
    "fluid-low-age-low-prev": "#1f77b4",  # blue
    "fluid-low-age-threshold-offenses": "#2ca02c",  # green
}
POLICY_ALIAS = {
    "age-first": "age-first-low-risk",
}

# Hatch patterns for legend: policies starting with "age-" get dashed pattern
# See matplotlib hatch patterns: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
POLICY_HATCHES = {
    "age-tolerance": "//",
    "age-first": "//",
    "age-first-high-risk": "//",
    "fluid-low-age-low-prev": "//",
    "fluid-low-age-threshold-offenses": "//",
}

# Line styles for plotting series: policies starting with "age-" get dashed lines
# See matplotlib linestyles: '-' (solid), '--' (dashed), '-.' (dashdot), ':' (dotted)
POLICY_LINESTYLES = {
    "age-tolerance": "--",
    "age-first": "--",
    "age-first-high-risk": "--",
    "fluid-low-age-low-prev": "--",
    "fluid-low-age-threshold-offenses": "--",
}


def get_policy_color(policy_name, fallback_idx=0):
    """Get color for a policy name, with fallback to tab10 colors."""
    import matplotlib.pyplot as plt

    if policy_name in POLICY_COLORS:
        return POLICY_COLORS[policy_name]
    # Fallback to tab10 colors
    colors = plt.cm.tab10.colors
    return colors[fallback_idx % len(colors)]


def get_policy_hatch(policy_name):
    """Get hatch pattern for a policy name.

    Returns hatch pattern if defined in POLICY_HATCHES,
    or '//' for policies starting with 'age-', otherwise None (solid fill).
    """
    if policy_name in POLICY_HATCHES:
        return POLICY_HATCHES[policy_name]
    # Default: age- policies get dashed hatch, others get solid fill
    if policy_name.startswith("age-"):
        return "//"
    return None


def get_policy_linestyle(policy_name):
    """Get line style for a policy name.

    Returns linestyle if defined in POLICY_LINESTYLES,
    or '--' for policies starting with 'age-', otherwise '-' (solid line).
    """
    if policy_name in POLICY_LINESTYLES:
        return POLICY_LINESTYLES[policy_name]
    # Default: age- policies get dashed lines, others get solid lines
    if policy_name.startswith("age-"):
        return "--"
    return "-"


# ------------------------------------------------------------
# dump / read metrics to / from h5 file (per-rep structure)
# ------------------------------------------------------------
def dump_rep_metrics(simulator, output_dir, p_freeze):
    """
    Save a single repetition's metrics to {output_dir}/metrics.h5.

    Args:
        simulator: a single simulator object
        output_dir: directory to save metrics.h5 (e.g., results/policy_name/0/)
        p_freeze: p_freeze parameter for summarize_trajectory
        window_for_ratios: window_for_ratios parameter for summarize_trajectory
    """
    import h5py

    simulator.summarize(save=False)
    mean_df, results, results_df, results_flow, results_retention = (
        simulator.summarize_trajectory(
            p_freeze=p_freeze, state_lst_columns=["state_lst"], state_columns=["state"]
        )
    )

    # also save the dfi as excel file
    simulator.dfi.to_excel(f"{output_dir}/dfi.xlsx")
    results_df[-1].to_excel(f"{output_dir}/df_result_last.xlsx")

    metrics = simulation.evaluation_metrics(results_df)

    # Save to HDF5 file with flat structure (one 1D array per metric)
    filepath = f"{output_dir}/metrics.h5"
    with h5py.File(filepath, "w") as f:
        for metric_key, values in metrics.items():
            f.create_dataset(metric_key, data=values)

    print(f"Saved metrics to {filepath}")


def read_metrics_from_h5(name, output_dir="results"):
    """
    Load and aggregate metrics from all reps in {output_dir}/{name}/*/metrics.h5.

    Args:
        name: policy name
        output_dir: base directory containing policy subdirectories

    Returns:
        policy_metrics: dict[metric_key] -> 2D array (repeats x episodes)
        policy_agg: dict[metric_key] -> dict with 'mean', 'max', 'min', 'std'
    """
    import h5py

    policy_dir = f"{output_dir}/{name}"

    # Find all rep directories (numeric subdirectories)
    rep_dirs = sorted(
        [
            d
            for d in os.listdir(policy_dir)
            if os.path.isdir(f"{policy_dir}/{d}") and d.isdigit()
        ],
        key=int,
    )

    if not rep_dirs:
        raise ValueError(f"No repetition directories found in {policy_dir}")

    # Load metrics from each rep
    all_rep_metrics = []
    for rep in rep_dirs:
        filepath = f"{policy_dir}/{rep}/metrics.h5"
        if not os.path.exists(filepath):
            print(f"Warning: metrics.h5 not found for {rep}, skipping")
            continue
        with h5py.File(filepath, "r") as f:
            rep_metrics = {key: f[key][:] for key in f.keys()}
        all_rep_metrics.append(rep_metrics)

    # Stack into matrices and compute aggregates
    if not all_rep_metrics:
        raise ValueError(
            f"No metrics.h5 files found in any repetition directory under {policy_dir}"
        )
    metric_keys = all_rep_metrics[0].keys()
    policy_metrics = {}
    policy_agg = {}

    # Per-state fields are flattened and vary in length - store as list, no aggregates
    # These share the same index structure (state_index_flat / state_lengths)
    per_state_keys = {
        # Shared indices
        "state_index_flat",
        "state_lengths",
        # Treatment data
        "treatment_tau_flat",
        "treatment_tau_rel_flat",
        # Population data
        "population_x0_flat",
        "population_yin_flat",
        # Legacy names (for backward compatibility)
        "treatment_index_flat",
        "treatment_lengths",
    }

    for metric_key in metric_keys:
        arrays = [m[metric_key] for m in all_rep_metrics]

        if metric_key in per_state_keys:
            # Store as list of arrays (no stacking/aggregation)
            policy_metrics[metric_key] = arrays
            # No aggregation for treatment fields
            continue

        # Find minimum length across all reps (in case of different episode counts)
        min_len = min(len(arr) for arr in arrays)

        # Truncate to minimum length and stack into matrix (n_reps x n_episodes)
        matrix = np.array([arr[:min_len] for arr in arrays])
        policy_metrics[metric_key] = matrix

        # Compute aggregates
        policy_agg[metric_key] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0),
            "min": np.min(matrix, axis=0),
            "max": np.max(matrix, axis=0),
        }

    return policy_metrics, policy_agg


def load_to_metrics(
    all_metrics, agg_metrics, name, output_dir="results", print_tree=False
):
    """
    Load metrics from policy directory and add to existing dicts.

    Args:
        all_metrics: existing all_metrics dict to update
        agg_metrics: existing agg_metrics dict to update
        name: policy name (subdirectory under output_dir)
        output_dir: base directory containing policy subdirectories
        print_tree: if True, print directory tree after loading all policies

    Returns:
        all_metrics, agg_metrics (updated in place, also returned for convenience)

    Example:
        all_metrics, agg_metrics = {}, {}
        load_to_metrics(all_metrics, agg_metrics, "null", "results")
        load_to_metrics(all_metrics, agg_metrics, "random", "results", print_tree=True)
        plot_metric(all_metrics, "total_offenses", start=20)
    """
    import os

    policy_metrics, policy_agg = read_metrics_from_h5(name, output_dir)
    all_metrics[name] = policy_metrics
    agg_metrics[name] = policy_agg

    # Print directory tree if requested (only for this policy)
    if print_tree:
        _print_directory_tree(output_dir, loaded_policies=[name])

    return all_metrics, agg_metrics


def _print_directory_tree(
    directory, prefix="", max_depth=2, current_depth=0, loaded_policies=None
):
    """
    Print directory tree structure, showing only loaded policies.

    Args:
        directory: root directory to print
        prefix: prefix for indentation (used internally for recursion)
        max_depth: maximum depth to traverse
        current_depth: current depth (used internally for recursion)
        loaded_policies: list of policy names that were loaded (only these will be shown)
    """
    import os

    if current_depth == 0:
        print(f"\nDirectory tree for '{directory}':")
        print(f"{directory}/")

    if current_depth >= max_depth:
        return

    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return

    # Separate directories and files
    all_dirs = [
        e
        for e in entries
        if os.path.isdir(os.path.join(directory, e)) and not e.startswith(".")
    ]
    all_files = [
        e
        for e in entries
        if os.path.isfile(os.path.join(directory, e)) and not e.startswith(".")
    ]

    # Only show loaded policies at root level
    if current_depth == 0 and loaded_policies is not None:
        # Show only loaded policy names and their subdirectories
        dirs = [d for d in all_dirs if d in loaded_policies]

        for i, d in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}{d}/")

            # Recurse into this policy directory to show its contents
            extension = "    " if is_last_dir else "│   "
            _print_directory_tree(
                os.path.join(directory, d),
                prefix + extension,
                max_depth,
                current_depth + 1,
                loaded_policies,
            )
    elif current_depth > 0:
        # Inside policy directories, show all subdirectories
        dirs = all_dirs

        for i, d in enumerate(dirs):
            is_last_dir = i == len(dirs) - 1
            connector = "└── " if is_last_dir else "├── "
            print(f"{prefix}{connector}{d}/")

            # Continue recursing
            extension = "    " if is_last_dir else "│   "
            _print_directory_tree(
                os.path.join(directory, d),
                prefix + extension,
                max_depth,
                current_depth + 1,
                loaded_policies,
            )


# ------------------------------------------------------------
# add metrics to existing dicts, not from h5 file
# ------------------------------------------------------------
def add_to_metrics(all_metrics, agg_metrics, k, simulators, p_freeze, windowsize=20):
    """
    Add a new policy's metrics to existing all_metrics and agg_metrics dicts.

    Args:
        all_metrics: existing all_metrics dict to update
        agg_metrics: existing agg_metrics dict to update
        k: name/key for the new policy
        simulators: list of simulators (or single simulator) for the new policy
        p_freeze: p_freeze parameter for summarize_trajectory

    Returns:
        all_metrics, agg_metrics (updated in place, also returned for convenience)
    """
    # Handle both single simulator and list of simulators
    if not isinstance(simulators, list):
        simulators = [simulators]

    # Collect metrics from each repeat
    repeat_metrics = []
    for simulator in simulators:
        simulator.summarize(save=False)
        mean_df, results, results_df, results_flow, results_retention = (
            simulator.summarize_trajectory(
                p_freeze=p_freeze,
                state_lst_columns=["state_lst"],
                state_columns=["state"],
            )
        )
        repeat_metrics.append(simulation.evaluation_metrics(results_df))

    # Convert to matrix form: each metric becomes (n_repeats x n_episodes)
    metric_keys = repeat_metrics[0].keys()
    all_metrics[k] = {}
    agg_metrics[k] = {}

    for metric_key in metric_keys:
        # Stack all repeats into a matrix
        matrix = np.array([m[metric_key] for m in repeat_metrics])
        all_metrics[k][metric_key] = matrix

        # Compute aggregates across repeats (axis=0)
        agg_metrics[k][metric_key] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0),
            "min": np.min(matrix, axis=0),
            "max": np.max(matrix, axis=0),
        }

    return all_metrics, agg_metrics


# ------------------------------------------------------------
# plot metrics from all_metrics, which is a dict of results
# ------------------------------------------------------------
def _compute_rolling_sum_per_rep(data_2d, num_period_window):
    """
    Compute rolling sum over window for each rep.

    Args:
        data_2d: 2D array (n_reps, n_episodes)
        num_period_window: window size for rolling sum

    Returns:
        2D array (n_reps, n_episodes) with rolling sums
    """
    n_reps, n_episodes = data_2d.shape
    cumsum = np.cumsum(data_2d, axis=1)
    result = np.zeros_like(data_2d)
    for i in range(n_episodes):
        if i < num_period_window:
            result[:, i] = cumsum[:, i]
        else:
            result[:, i] = cumsum[:, i] - cumsum[:, i - num_period_window]
    return result


def _compute_cumulative_mean_per_rep(data_2d):
    """
    Compute cumulative mean for each rep.

    Args:
        data_2d: 2D array (n_reps, n_episodes)

    Returns:
        2D array (n_reps, n_episodes) with cumulative means
    """
    cumsum = np.cumsum(data_2d, axis=1)
    divisor = np.arange(1, data_2d.shape[1] + 1)
    return cumsum / divisor


def plot_metric(
    metrics_dict,
    metric_key,
    ylabel=None,
    title=None,
    show_ci=True,
    start=20,
    output_path=None,
    num_period_window=None,
    relative=False,
    relative_policy="null",
):
    """
    Plot a single metric for all simulations with confidence intervals.
    Uses raw per-rep data for consistent statistics with equilibrium computation.
    Computes rolling sum over last num_period_window periods per rep, then aggregates.
    If num_period_window is None, computes cumulative mean per rep, then aggregates.

    Args:
        metrics_dict: all_metrics from collect_metrics() (dict with 2D arrays per metric)
        metric_key: e.g. 'total_population', 'total_offenses', 'total_enrollment',
                    'total_incarcerated', 'total_returns', 'offense_rate', etc.
        ylabel: optional y-axis label (defaults to metric_key)
        title: optional plot title
        show_ci: whether to show confidence interval (mean ± 2*std band)
        start: starting episode index (skip first N episodes)
        output_path: path to save PDF (default: /tmp/{metric_key}-trend.pdf)
        num_period_window: window size for rolling sum (if None, uses cumulative mean)
        relative: if True, plot values relative to relative_policy (default: False)
        relative_policy: policy name to use as baseline for relative plots (default: "null")
    """
    fig = go.Figure()
    import matplotlib.colors as mcolors

    # Compute baseline values if relative mode is enabled (per-rep)
    baseline_per_rep = None
    if relative:
        if relative_policy not in metrics_dict:
            print(
                f"Warning: relative_policy '{relative_policy}' not found in metrics_dict. "
                "Plotting absolute values instead."
            )
            relative = False
        elif metric_key not in metrics_dict[relative_policy]:
            print(
                f"Warning: metric_key '{metric_key}' not found for relative_policy '{relative_policy}'. "
                "Plotting absolute values instead."
            )
            relative = False
        else:
            baseline_data = metrics_dict[relative_policy][metric_key]
            # Check if it's 2D array (raw per-rep data) or dict (agg_metrics)
            if isinstance(baseline_data, np.ndarray) and baseline_data.ndim == 2:
                if num_period_window is not None:
                    baseline_per_rep = _compute_rolling_sum_per_rep(
                        baseline_data, num_period_window
                    )
                else:
                    baseline_per_rep = _compute_cumulative_mean_per_rep(baseline_data)
            else:
                print(
                    f"Warning: Expected 2D array for relative_policy '{relative_policy}'. "
                    "Plotting absolute values instead."
                )
                relative = False

    for i, (k, metrics) in enumerate(metrics_dict.items()):
        if metric_key not in metrics:
            continue

        data = metrics[metric_key]

        # Check if it's 2D array (raw per-rep data) or dict (agg_metrics - legacy)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            # Raw per-rep data: compute rolling sum/cumulative mean per rep, then aggregate
            if num_period_window is not None:
                processed = _compute_rolling_sum_per_rep(data, num_period_window)
            else:
                processed = _compute_cumulative_mean_per_rep(data)

            # Subtract baseline if relative mode is enabled (per-rep subtraction)
            if relative and baseline_per_rep is not None:
                # Ensure shapes match
                min_len = min(processed.shape[1], baseline_per_rep.shape[1])
                processed = processed[:, :min_len] - baseline_per_rep[:, :min_len]

            # Aggregate across reps
            raw_mean = np.mean(processed, axis=0)
            raw_std = np.std(processed, axis=0)
        elif isinstance(data, dict) and "mean" in data:
            # Legacy agg_metrics format - keep for backward compatibility
            print(
                f"Warning: {k} uses legacy agg_metrics format. Consider using all_metrics."
            )
            if num_period_window is not None:
                cumsum = np.cumsum(data["mean"])
                raw_mean = np.array(
                    [
                        (
                            cumsum[i]
                            if i < num_period_window
                            else cumsum[i] - cumsum[i - num_period_window]
                        )
                        for i in range(len(data["mean"]))
                    ]
                )
                raw_std = data["std"] * np.sqrt(
                    np.minimum(np.arange(1, len(data["std"]) + 1), num_period_window)
                )
            else:
                raw_mean = np.cumsum(data["mean"]) / np.arange(1, len(data["mean"]) + 1)
                raw_std = data["std"] / np.sqrt(np.arange(1, len(data["std"]) + 1))

            if relative and baseline_per_rep is not None:
                baseline_mean = np.mean(baseline_per_rep, axis=0)
                min_len = min(len(raw_mean), len(baseline_mean))
                raw_mean = raw_mean[:min_len] - baseline_mean[:min_len]
        else:
            print(f"Skipping {k}: expected 2D array or dict with 'mean' and 'std'")
            continue

        # Use same policy colors as box plots
        color_raw = get_policy_color(k, fallback_idx=i)
        # Convert to hex for Plotly
        if isinstance(color_raw, str) and color_raw.startswith("#"):
            color = color_raw
        else:
            color = mcolors.to_hex(color_raw[:3])

        mean_vals = raw_mean[start:]
        std_vals = raw_std[start:]
        min_vals = mean_vals - 2 * std_vals
        max_vals = mean_vals + 2 * std_vals

        # Episodes start from (start)
        episodes = np.arange(start, start + len(mean_vals))

        # Convert color to rgba with alpha
        rgb = pc.hex_to_rgb(color) if color.startswith("#") else pc.unlabel_rgb(color)
        fill_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)"

        # Add confidence interval (min-max band)
        if show_ci and min_vals is not None and max_vals is not None:
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([episodes, episodes[::-1]]),
                    y=np.concatenate([max_vals, min_vals[::-1]]),
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    legendgroup=k,
                    name=f"{k} (range)",
                    hoverinfo="skip",
                )
            )

        # Add mean line
        fig.add_trace(
            go.Scatter(
                x=episodes,
                y=mean_vals,
                mode="lines",
                name=k,
                legendgroup=k,
                line=dict(color=color),
            )
        )

    fig.update_layout(
        title=title or metric_key,
        xaxis_title="episode",
        yaxis_title=None,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="grey",
            mirror=False,
        ),
        yaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="grey",
            mirror=False,
        ),
    )

    # Save to PDF with no legend and no title
    if output_path is None:
        output_path = f"/tmp/{metric_key}-trend.pdf"
    # Create a copy for saving with modifications
    fig_save = go.Figure(fig)
    fig_save.update_layout(
        showlegend=False,
        title=None,
    )
    fig_save.write_image(output_path)
    print(f"Saved to {output_path}")

    fig.show()
    return fig


# ------------------------------------------------------------
# main function to produce figures for the papers.
# ------------------------------------------------------------


def plot_to_tex_equilibrium_box(
    metrics_dict,
    metric_key,
    ylabel="value",
    title=None,
    output_path="/tmp/fig.pdf",
    show_labels=False,
    num_period_window=None,
):
    """
    Plot the equilibrium box plot for the given metric.
    Computes sum over last num_period_window periods: sum_{t=T-W+1}^{T} x_t for each repetition
    If num_period_window is None, computes mean over ALL episodes: (1/T) * sum_{t=1}^{T} x_t

    Args:
        metrics_dict: policy_metrics from read_metrics_from_h5() (dict with 2D arrays: repeats x episodes)
        metric_key: e.g. 'total_population', 'total_offenses', 'total_enrollment',
                    'total_incarcerated', 'total_returns', 'offense_rate', etc.
        ylabel: optional y-axis label (defaults to metric_key)
        title: optional plot title
        output_path: path to save the file (default "/tmp/fig.pdf")
                     Supports .pdf, .png, .pgf, or .tex extensions
        show_labels: if True, show policy names on x-axis (default False)
        num_period_window: window size for sum over last periods (if None, uses mean over all)
    """

    # Collect data for box plot: list of (policy_name, flattened_values)
    box_data = []
    labels = []

    for k, metrics in metrics_dict.items():
        if metric_key not in metrics:
            continue

        data = metrics[metric_key]
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            print(f"Skipping {k}: expected 2D array format (repeats x episodes)")
            continue

        # Compute sum over last num_period_window or mean over ALL episodes for each repetition
        if num_period_window is not None:
            # Sum over last num_period_window periods for each repetition
            vals = np.sum(data[:, -num_period_window:], axis=1)
        else:
            # Mean over ALL episodes for each repetition
            vals = np.mean(data, axis=1)
        box_data.append(vals)
        labels.append(k)

    if not box_data:
        print(f"No data found for metric '{metric_key}'")
        return None

    # Create box plot
    fig, ax = plt.subplots(
        figsize=(max(6, len(labels) * 1.2), 3.5),
    )

    # Use empty labels if show_labels is False
    display_labels = labels if show_labels else [""] * len(labels)
    bp = ax.boxplot(box_data, labels=display_labels, patch_artist=True)

    # Style the box plot using policy colors and add mean values
    for i, (box, median, label) in enumerate(zip(bp["boxes"], bp["medians"], labels)):
        color = get_policy_color(label, fallback_idx=i)
        # Convert hex to rgba with alpha
        if isinstance(color, str) and color.startswith("#"):
            rgb = mcolors.hex2color(color)
            box.set_facecolor((*rgb, 0.6))
            box.set_edgecolor(color)
        else:
            box.set_facecolor((*color[:3], 0.6))
            box.set_edgecolor(color)
        median.set_color("black")
        median.set_linewidth(1.5)

    # Set font for axis labels and ticks
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Rotate labels if many policies and labels are shown
    if show_labels and len(labels) > 4:
        plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    # Determine output format from extension
    if output_path.endswith((".pgf", ".tex")):
        # Save as PGF/TikZ for LaTeX
        plt.savefig(output_path, bbox_inches="tight", backend="pgf")
    else:
        # Save as raster/vector format (PDF, PNG, etc.)
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.show()

    print(f"Saved to {output_path}")
    return fig


import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def fsd_test(xa, xb, alpha=0.05):
    """
    Test if xa first-order stochastically dominates xb (xa is better, i.e., lower values).

    For lower-is-better metrics (like number of offenses):
    - xa dominates xb if F_a(x) >= F_b(x) for all x
    - Meaning: xa has more probability mass on lower (better) values
    - Equivalently: the CDF of xa is always above or equal to the CDF of xb

    Args:
        xa: array of values for policy A (lower is better)
        xb: array of values for policy B (lower is better)
        alpha: significance level for hypothesis test

    Returns:
        dict with keys:
            - ks_stat: KS test statistic
            - p_value: p-value for one-sided KS test (H1: xa dominates xb)
            - fsd_empirical: True if F_a(x) >= F_b(x) everywhere (empirical FSD)
            - reject_at_alpha: True if we reject null at given alpha
    """
    xa, xb = np.asarray(xa), np.asarray(xb)

    # For lower-is-better: xa dominates xb if F_a >= F_b
    # KS test with alternative='greater' tests if CDF of xa is above CDF of xb
    # This means xa has more mass on lower values (is better)
    ks = ks_2samp(xa, xb, alternative="greater")

    # empirical FSD check: F_a(x) >= F_b(x) for all x
    # (xa's CDF should be above or equal to xb's CDF everywhere)
    grid = np.sort(np.unique(np.r_[xa, xb]))
    Fa = np.searchsorted(np.sort(xa), grid, side="right") / len(xa)
    Fb = np.searchsorted(np.sort(xb), grid, side="right") / len(xb)
    # xa dominates xb if Fa >= Fb everywhere (more mass on lower values)
    no_cross = np.all(Fa >= Fb - 1e-12)

    return {
        "ks_stat": ks.statistic,
        "p_value": ks.pvalue,
        "fsd_empirical": bool(no_cross),
        "reject_at_alpha": ks.pvalue < alpha,
    }


def pairwise_fsd_test(
    metrics_dict,
    metric_key,
    alpha=0.05,
    return_format="dataframe",
):
    """
    Perform pairwise FSD tests between all policies using data from box plot statistics.
    Computes mean over ALL episodes: (1/T) * sum_{t=1}^{T} x_t

    Args:
        metrics_dict: either all_metrics or agg_metrics from collect_metrics()
        metric_key: which metric to test (e.g., 'num_offense', 'num_treated')
        alpha: significance level for hypothesis test (default: 0.05)
        return_format: "dataframe" or "dict" (default: "dataframe")

    Returns:
        If return_format="dataframe":
            A pandas DataFrame with MultiIndex columns containing:
            - p_value: p-value matrix (row dominates column if p < alpha)
            - ks_stat: KS statistic matrix
            - fsd_empirical: empirical FSD matrix (True if row dominates column)
            - mean_diff: mean difference matrix (row - column, negative means row is better)

        If return_format="dict":
            Dictionary with keys 'p_value', 'ks_stat', 'fsd_empirical', 'mean_diff',
            each containing a DataFrame with policies as both rows and columns.

    Example:
        >>> results = pairwise_fsd_test(all_metrics, 'num_offense', last_n=5)
        >>> # Check if 'high-risk' dominates 'null' at alpha=0.05
        >>> results['p_value'].loc['high-risk', 'null'] < 0.05
    """
    # Collect data for each policy (same as box plot)
    policy_data = {}

    for k, metrics in metrics_dict.items():
        if metric_key not in metrics:
            continue

        data = metrics[metric_key]

        # Auto-detect format and compute mean over ALL episodes
        if isinstance(data, dict) and "mean" in data:
            # agg_metrics format: mean over ALL episodes
            vals = np.array([np.mean(data["mean"])])
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                # all_metrics format (matrix: repeats x episodes)
                # Mean over ALL episodes for each repetition
                vals = np.mean(data, axis=1)
            else:
                # 1D array (single run): mean over ALL episodes
                vals = np.array([np.mean(data)])
        else:
            continue

        policy_data[k] = vals

    if not policy_data:
        print(f"No data found for metric '{metric_key}'")
        return None

    policies = list(policy_data.keys())
    n_policies = len(policies)

    # Initialize result matrices
    p_values = np.full((n_policies, n_policies), np.nan)
    ks_stats = np.full((n_policies, n_policies), np.nan)
    fsd_empirical = np.full((n_policies, n_policies), False)
    mean_diffs = np.full((n_policies, n_policies), np.nan)

    # Perform pairwise tests
    for i, policy_a in enumerate(policies):
        for j, policy_b in enumerate(policies):
            if i == j:
                # Diagonal: policy compared to itself
                p_values[i, j] = 1.0
                ks_stats[i, j] = 0.0
                fsd_empirical[i, j] = True
                mean_diffs[i, j] = 0.0
            else:
                xa = policy_data[policy_a]
                xb = policy_data[policy_b]

                # Test if policy_a dominates policy_b
                result = fsd_test(xa, xb, alpha=alpha)
                p_values[i, j] = result["p_value"]
                ks_stats[i, j] = result["ks_stat"]
                fsd_empirical[i, j] = result["fsd_empirical"]
                mean_diffs[i, j] = np.mean(xa) - np.mean(xb)

    # Create DataFrames with policy names as index and columns
    p_value_df = pd.DataFrame(p_values, index=policies, columns=policies)
    ks_stat_df = pd.DataFrame(ks_stats, index=policies, columns=policies)
    fsd_empirical_df = pd.DataFrame(fsd_empirical, index=policies, columns=policies)
    mean_diff_df = pd.DataFrame(mean_diffs, index=policies, columns=policies)

    if return_format == "dict":
        return {
            "p_value": p_value_df,
            "ks_stat": ks_stat_df,
            "fsd_empirical": fsd_empirical_df,
            "mean_diff": mean_diff_df,
        }
    else:
        # Return as MultiIndex DataFrame
        result_df = pd.concat(
            [p_value_df, ks_stat_df, fsd_empirical_df, mean_diff_df],
            keys=["p_value", "ks_stat", "fsd_empirical", "mean_diff"],
            axis=1,
        )
        return result_df


def plot_policy_legend(
    policy_names=None,
    output_path="/tmp/legend.pdf",
    ncol=None,
    nrows=1,
    orientation="horizontal",
    figsize=None,
    alpha=0.6,
    columnspacing=1.0,
    style="line",
    linewidth=6,
):
    """
    Create a standalone legend figure showing policy colors.

    Args:
        policy_names: list of policy names to include (default: all in POLICY_COLORS)
        output_path: path to save the file (default "/tmp/legend.pdf")
                     Supports .pdf, .png, .pgf, or .tex extensions
        ncol: number of columns in legend (default: auto based on orientation and nrows)
        nrows: number of rows to display legend items (default: 1)
        orientation: "horizontal" or "vertical"
        figsize: tuple (width, height) in inches (default: auto)
        alpha: transparency for patches (default 0.6 to match box plots)
        columnspacing: space between columns in legend (default: 1.0, smaller = closer)
        style: "line" for line series legends, "patch" for box plot legends (default: "line")
        linewidth: line width for line style legends (default: 2)

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib.colors as mcolors

    if policy_names is None:
        policy_names = list(POLICY_COLORS.keys())

    # Create legend handles
    handles = []
    for i, name in enumerate(policy_names):
        if name in POLICY_ALIAS:
            real_name = POLICY_ALIAS[name]
        else:
            real_name = name
        color = get_policy_color(name, fallback_idx=i)

        if style == "line":
            # Line style for series plots
            linestyle = get_policy_linestyle(name)
            line = mlines.Line2D(
                [],
                [],
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=real_name,
            )
            handles.append(line)
        else:
            # Patch style for box plots
            hatch = get_policy_hatch(name)
            # Convert to RGBA with alpha
            if isinstance(color, str) and color.startswith("#"):
                rgb = mcolors.hex2color(color)
                facecolor = (*rgb, alpha)
            else:
                facecolor = (*color[:3], alpha)
            patch = mpatches.Patch(
                facecolor=facecolor, edgecolor=color, hatch=hatch, label=name
            )
            handles.append(patch)

    # Determine layout
    if ncol is None:
        if orientation == "horizontal":
            # Distribute items across nrows
            ncol = (len(policy_names) + nrows - 1) // nrows  # Ceiling division
        else:
            ncol = 1

    if figsize is None:
        if orientation == "horizontal":
            figsize = (min(len(policy_names) * 1.5, 10), 0.5 * nrows)
        else:
            figsize = (2, len(policy_names) * 0.3)

    # Create figure with just the legend
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Use global font with size 10 for legend
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        frameon=False,
        columnspacing=columnspacing,
    )

    plt.tight_layout()

    # Determine output format from extension
    if output_path.endswith((".pgf", ".tex")):
        # Save as PGF/TikZ for LaTeX
        plt.savefig(output_path, bbox_inches="tight", backend="pgf")
    else:
        # Save as PDF or PNG
        plt.savefig(output_path, bbox_inches="tight", dpi=150)

    plt.show()

    print(f"Saved legend to {output_path}")
    return fig
