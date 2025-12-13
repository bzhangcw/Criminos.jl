import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

import simulation
import sirakaya

import plotly.express as px
import plotly.graph_objects as go

RESULT_PREFIX = os.environ.get("RESULT_PREFIX", "results/metrics")


# ------------------------------------------------------------
# dump / read metrics to / from h5 file
# ------------------------------------------------------------
def dump_metrics_to_h5(name, simulators, p_freeze, windowsize=20):
    """
    Save metrics for a policy to a standalone HDF5 file.

    Args:
        name: policy name (used for filename)
        simulators: list of simulators (or single simulator)
        p_freeze: p_freeze parameter for summarize_trajectory
        windowsize: windowsize parameter for summarize_trajectory

    Returns:
        all_metrics: dict[metric_key] -> 2D array (repeats x episodes)
        agg_metrics: dict[metric_key] -> dict with 'mean', 'max', 'min', 'std'
    """
    import h5py

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
                windowsize=windowsize,
            )
        )
        repeat_metrics.append(simulation.evaluation_metrics(results_df))

    # Convert to matrix form: each metric becomes (n_repeats x n_episodes)
    metric_keys = repeat_metrics[0].keys()
    all_metrics = {}
    agg_metrics = {}

    for metric_key in metric_keys:
        # Stack all repeats into a matrix
        matrix = np.array([m[metric_key] for m in repeat_metrics])
        all_metrics[metric_key] = matrix

        # Compute aggregates across repeats (axis=0)
        agg_metrics[metric_key] = {
            "mean": np.mean(matrix, axis=0),
            "std": np.std(matrix, axis=0),
            "min": np.min(matrix, axis=0),
            "max": np.max(matrix, axis=0),
        }

    # Save to HDF5 file
    with h5py.File(f"{RESULT_PREFIX}-{name}.h5", "w") as f:
        for metric_key, matrix in all_metrics.items():
            # Save raw matrix (repeats x episodes)
            f.create_dataset(f"{metric_key}/data", data=matrix)
            # Save aggregates
            f.create_dataset(f"{metric_key}/mean", data=agg_metrics[metric_key]["mean"])
            f.create_dataset(f"{metric_key}/std", data=agg_metrics[metric_key]["std"])
            f.create_dataset(f"{metric_key}/min", data=agg_metrics[metric_key]["min"])
            f.create_dataset(f"{metric_key}/max", data=agg_metrics[metric_key]["max"])

    print(f"Saved metrics to results/metrics-{name}.h5")
    return all_metrics, agg_metrics


def read_metrics_from_h5(filepath):
    """
    Load metrics for a policy from HDF5 file.

    Args:
        filepath: path to HDF5 file

    Returns:
        policy_metrics: dict[metric_key] -> 2D array (repeats x episodes)
        policy_agg: dict[metric_key] -> dict with 'mean', 'max', 'min', 'std'
    """
    import h5py

    policy_metrics = {}
    policy_agg = {}

    with h5py.File(filepath, "r") as f:
        for metric_key in f.keys():
            policy_metrics[metric_key] = f[f"{metric_key}/data"][:]
            policy_agg[metric_key] = {
                "mean": f[f"{metric_key}/mean"][:],
                "std": f[f"{metric_key}/std"][:],
                "min": f[f"{metric_key}/min"][:],
                "max": f[f"{metric_key}/max"][:],
            }

    return policy_metrics, policy_agg


def load_to_metrics(all_metrics, agg_metrics, name, filepath):
    """
    Load metrics from HDF5 file and add to existing dicts.

    Args:
        all_metrics: existing all_metrics dict to update
        agg_metrics: existing agg_metrics dict to update
        name: key name for this policy in the dicts
        filepath: path to HDF5 file

    Returns:
        all_metrics, agg_metrics (updated in place, also returned for convenience)

    Example:
        all_metrics, agg_metrics = {}, {}
        load_to_metrics(all_metrics, agg_metrics, "null", "results/metrics-null.h5")
        load_to_metrics(all_metrics, agg_metrics, "random", "results/metrics-random.h5")
        plot_metric(all_metrics, "total_offenses", start=20)
    """
    policy_metrics, policy_agg = read_metrics_from_h5(filepath)
    all_metrics[name] = policy_metrics
    agg_metrics[name] = policy_agg
    return all_metrics, agg_metrics


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
        windowsize: windowsize parameter for summarize_trajectory

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
                windowsize=windowsize,
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
def plot_metric(
    metrics_dict, metric_key, ylabel=None, title=None, show_ci=True, start=20
):
    """
    Plot a single metric for all simulations with confidence intervals.

    Args:
        metrics_dict: either all_metrics or agg_metrics from collect_metrics()
        metric_key: e.g. 'total_population', 'total_offenses', 'total_enrollment',
                    'total_incarcerated', 'total_returns', 'offense_rate', etc.
        ylabel: optional y-axis label (defaults to metric_key)
        title: optional plot title
        show_ci: whether to show confidence interval (min/max band)
        start: starting episode index (skip first N episodes)
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, (k, metrics) in enumerate(metrics_dict.items()):
        if metric_key not in metrics:
            continue

        data = metrics[metric_key]
        color = colors[i % len(colors)]

        # Auto-detect format: agg_metrics has dict with 'mean', all_metrics has ndarray
        if isinstance(data, dict) and "mean" in data:
            # agg_metrics format: use mean ± 1.5*std
            mean_vals = data["mean"][start:]
            std_vals = data["std"][start:]
            min_vals = np.maximum(mean_vals - 1 * std_vals, 0)
            max_vals = mean_vals + 1 * std_vals
        elif isinstance(data, np.ndarray):
            # all_metrics format (matrix: repeats x episodes)
            if data.ndim == 2:
                mean_vals = np.mean(data[:, start:], axis=0)
                std_vals = np.std(data[:, start:], axis=0)
                min_vals = np.maximum(mean_vals - 1.5 * std_vals, 0)
                max_vals = mean_vals + 1.5 * std_vals
            else:
                # 1D array (single run)
                mean_vals = data[start:]
                min_vals = max_vals = None
        else:
            continue

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
        xaxis_title="Episode",
        yaxis_title=ylabel or metric_key,
        hovermode="x unified",
    )
    fig.show()
    return fig
