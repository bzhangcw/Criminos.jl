"""
Sensitivity Analysis Script for Off-Probation Term Length and Prison Scale Factor.

This script analyzes simulation results across varying off-probation term length (tl) and
prison scale factor (sf) parameters, computing performance metrics for different policies.

Returns DataFrames indexed by (term_length, scale_factor, policy) with mean ± std of
per-repetition values.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings

# Import existing functionality
from simulation_summary import read_metrics_from_h5, get_policy_color


def parse_directory_structure(
    results_dir: str,
) -> Tuple[List[int], List[float], List[str]]:
    """
    Scan results directory to find all (term_length, scale_factor) combinations and available policies.

    Args:
        results_dir: Path to result-ofpl-scl directory

    Returns:
        Tuple of (term_lengths, scale_factors, policies)
        - term_lengths: List of term length values (e.g., [500, 1000, 1500, ...])
        - scale_factors: List of scale factor values (e.g., [0.05, 0.1, 0.2, ...])
        - policies: List of policy names (e.g., ['high-risk', 'low-risk', ...])
    """
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")

    term_lengths = set()
    scale_factors = set()
    policies = set()

    # Pattern to match directory names like "tl-4000-sf-1.0"
    pattern = re.compile(r"tl-(\d+)-sf-([\d.]+)")

    for dirname in os.listdir(results_dir):
        dirpath = os.path.join(results_dir, dirname)
        if not os.path.isdir(dirpath):
            continue

        match = pattern.match(dirname)
        if match:
            term_length = int(match.group(1))
            scale_factor = float(match.group(2))
            term_lengths.add(term_length)
            scale_factors.add(scale_factor)

            # Find policies in this directory
            for policy_name in os.listdir(dirpath):
                policy_path = os.path.join(dirpath, policy_name)
                if os.path.isdir(policy_path) and policy_name != "logs":
                    policies.add(policy_name)

    return (sorted(list(term_lengths)), sorted(list(scale_factors)), sorted(list(policies)))


def load_policy_metrics(
    results_dir: str, term_length: int, scale_factor: float, policy_name: str
) -> Tuple[Dict, Dict]:
    """
    Load metrics for a specific policy at given term length and scale factor.

    Args:
        results_dir: Path to result-ofpl-scl directory
        term_length: Off-probation term length value
        scale_factor: Prison scale factor value
        policy_name: Name of policy (e.g., 'high-risk', 'null')

    Returns:
        Tuple of (policy_metrics, policy_agg) from read_metrics_from_h5()
        - policy_metrics: dict[metric_key] -> 2D array (repeats × episodes)
        - policy_agg: dict[metric_key] -> dict with 'mean', 'std', 'min', 'max'
    """
    policy_dir = f"tl-{term_length}-sf-{scale_factor}"
    output_dir = os.path.join(results_dir, policy_dir)

    if not os.path.exists(os.path.join(output_dir, policy_name)):
        raise ValueError(
            f"Policy directory not found: {os.path.join(output_dir, policy_name)}"
        )

    return read_metrics_from_h5(policy_name, output_dir)


def compute_equilibrium_value(
    policy_metrics: Dict, metric_key: str, num_period_window: int
) -> np.ndarray:
    """
    Compute equilibrium value by summing over last num_period_window periods.

    Args:
        policy_metrics: Dict from read_metrics_from_h5 with 2D arrays
        metric_key: Name of metric to compute (e.g., 'total_offenses')
        num_period_window: Number of periods to sum over

    Returns:
        Array of values, one per repetition (shape: n_reps,)
    """
    if metric_key not in policy_metrics:
        raise ValueError(f"Metric '{metric_key}' not found in policy_metrics")

    data = policy_metrics[metric_key]

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError(
            f"Expected 2D array for metric '{metric_key}', got {type(data)} with shape {getattr(data, 'shape', 'N/A')}"
        )

    # Sum over last num_period_window periods for each repetition
    # data shape: (n_reps, n_episodes)
    vals = np.sum(data[:, -num_period_window:], axis=1)

    return vals


def analyze_sensitivity(
    results_dir: str, metrics: List[str], policies: List[str], summary_wd: int
) -> Dict[str, pd.DataFrame]:
    """
    Analyze sensitivity across term length and scale factor parameters.

    Args:
        results_dir: Path to result-ofpl-scl directory
        metrics: List of metric names (e.g., ['total_offenses'])
        policies: List of policy names (e.g., ['high-risk', 'low-risk', 'null'])
        summary_wd: Window size for equilibrium computation (num_period_window)

    Returns:
        dict: {metric_name: DataFrame} where each DataFrame has:
            - Index: MultiIndex (term_length, scale_factor, policy)
            - Columns: ['mean', 'std', 'n_reps', 'enrollment_mean', 'enrollment_std']
            - Values: Absolute values averaged over summary_wd periods
            - All statistics are per-period averages (divided by summary_wd)
    """
    print(f"Parsing directory structure from {results_dir}...")
    term_lengths, scale_factors, available_policies = parse_directory_structure(results_dir)

    print(f"Found {len(term_lengths)} term lengths: {term_lengths}")
    print(f"Found {len(scale_factors)} scale factors: {scale_factors}")
    print(f"Found {len(available_policies)} policies: {available_policies}")

    # Validate requested policies
    for policy in policies:
        if policy not in available_policies:
            warnings.warn(f"Policy '{policy}' not found in results directory")

    # Filter to available policies
    policies_to_analyze = [p for p in policies if p in available_policies]
    print(f"Analyzing {len(policies_to_analyze)} policies: {policies_to_analyze}")

    # Collect results for each metric
    results = {metric: [] for metric in metrics}

    # Loop through all combinations
    total_combinations = len(term_lengths) * len(scale_factors) * len(policies_to_analyze)
    processed = 0

    for term_length in term_lengths:
        for scale_factor in scale_factors:
            for policy in policies_to_analyze:
                processed += 1
                print(
                    f"Processing [{processed}/{total_combinations}]: "
                    f"term_length={term_length}, scale_factor={scale_factor}, policy={policy}"
                )

                try:
                    # Load policy metrics
                    policy_metrics, _ = load_policy_metrics(
                        results_dir, term_length, scale_factor, policy
                    )

                    # Compute enrollment statistics (absolute values)
                    # Divide by summary_wd to get average per period
                    enrollment_vals = compute_equilibrium_value(
                        policy_metrics, "total_enrollment", summary_wd
                    )
                    enrollment_mean = float(np.mean(enrollment_vals)) / summary_wd
                    enrollment_std = float(np.std(enrollment_vals)) / summary_wd

                    # Compute statistics for each metric
                    for metric in metrics:
                        try:
                            # Compute equilibrium values (sum over summary_wd periods)
                            policy_vals = compute_equilibrium_value(
                                policy_metrics, metric, summary_wd
                            )

                            # Compute absolute statistics (averaged per period)
                            # Divide by summary_wd to get per-period average
                            policy_vals_per_period = policy_vals / summary_wd

                            # Store result
                            results[metric].append(
                                {
                                    "term_length": term_length,
                                    "scale_factor": scale_factor,
                                    "policy": policy,
                                    "mean": float(np.mean(policy_vals_per_period)),
                                    "std": float(np.std(policy_vals_per_period)),
                                    "n_reps": len(policy_vals_per_period),
                                    "enrollment_mean": enrollment_mean,
                                    "enrollment_std": enrollment_std,
                                }
                            )

                        except Exception as e:
                            warnings.warn(
                                f"Failed to compute metric '{metric}' for "
                                f"tl={term_length}, sf={scale_factor}, policy={policy}: {e}"
                            )
                            continue

                except Exception as e:
                    warnings.warn(
                        f"Failed to load policy '{policy}' for "
                        f"tl={term_length}, sf={scale_factor}: {e}"
                    )
                    continue

    # Convert to DataFrames with MultiIndex
    dfs = {}
    for metric in metrics:
        if not results[metric]:
            warnings.warn(f"No results collected for metric '{metric}'")
            continue

        df = pd.DataFrame(results[metric])
        df = df.set_index(["term_length", "scale_factor", "policy"])
        df = df.sort_index()
        dfs[metric] = df

        print(f"\nMetric '{metric}': {len(df)} combinations")

    return dfs


def plot_scale_factor_trends(
    df: pd.DataFrame,
    metric_name: str,
    term_length: int,
    output_path: str,
    policies: Optional[List[str]] = None,
    ylabel: Optional[str] = None,
    show_std: bool = True,
):
    """
    Plot trends vs. scale factor for each policy at a given term length.

    Args:
        df: DataFrame from analyze_sensitivity (for one metric)
        metric_name: Name of the metric (for title)
        term_length: Term length value to plot (e.g., 2000)
        output_path: Base path to save files (will save .png and .pgf)
        policies: List of policies to plot (default: all in df)
        ylabel: Y-axis label (default: metric_name)
        show_std: Whether to show confidence interval region (default: True)
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter data for the given term length
    if term_length not in df.index.get_level_values("term_length"):
        raise ValueError(f"Term length {term_length} not found in data")

    subset = df.loc[term_length]

    # Get policies to plot
    if policies is None:
        policies = sorted(subset.index.get_level_values("policy").unique())

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot each policy
    for i, policy in enumerate(policies):
        if policy not in subset.index.get_level_values("policy"):
            warnings.warn(f"Policy '{policy}' not found for term_length={term_length}")
            continue

        policy_data = subset.xs(policy, level="policy")

        # Get color
        color = get_policy_color(policy, fallback_idx=i)

        # Get data as numpy arrays
        scale_factors = policy_data.index.to_numpy()
        means = policy_data["mean"].to_numpy()

        # Add confidence interval region if requested
        if show_std and "std" in policy_data.columns:
            stds = policy_data["std"].to_numpy()
            ax.fill_between(
                scale_factors,
                means - stds,
                means + stds,
                color=color,
                alpha=0.2,
            )

        # Plot line on top of the shaded region
        ax.plot(
            scale_factors,
            means,
            marker="o",
            label=policy,
            color=color,
            linewidth=2,
            markersize=6,
        )

    # Formatting
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Save figure in both PNG and PGF formats
    plt.tight_layout()

    # Use output_path as base (it should not have an extension)
    # Save PNG
    png_path = output_path + ".png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")

    # Save PGF
    pgf_path = output_path + ".pgf"
    plt.savefig(pgf_path, bbox_inches="tight")

    plt.close()

    print(f"Saved plot to {png_path} and {pgf_path}")


def plot_all_term_lengths(
    dfs: Dict[str, pd.DataFrame],
    output_dir: str,
    policies: Optional[List[str]] = None,
    show_std: bool = True,
):
    """
    Plot scale factor trends for all metrics and all term length values.

    Args:
        dfs: Dict of DataFrames from analyze_sensitivity
        output_dir: Directory to save PNG and PGF files
        policies: List of policies to plot (default: all)
        show_std: Whether to show confidence interval region (default: True)
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, df in dfs.items():
        # Get all term length values
        term_lengths = sorted(df.index.get_level_values("term_length").unique())

        print(f"\nPlotting {metric_name} for {len(term_lengths)} term lengths...")

        for term_length in term_lengths:
            # Create base filename (without extension)
            filename = f"{metric_name}_term_length_{term_length}"
            output_path = os.path.join(output_dir, filename)

            try:
                plot_scale_factor_trends(
                    df=df,
                    metric_name=metric_name,
                    term_length=term_length,
                    output_path=output_path,
                    policies=policies,
                    show_std=show_std,
                )
            except Exception as e:
                warnings.warn(f"Failed to plot {metric_name} for term_length={term_length}: {e}")
                continue

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Off-Probation Term Length and Scale Factor Sensitivity Analysis")
    print("=" * 60)

    # Configuration
    results_dir = "result-ofpl-scl-4"
    metrics = ["total_offenses", "offense_rate"]
    policies = ["high-risk", "low-risk", "age-first"]
    summary_wd = 20

    # Run analysis
    print(f"\nConfiguration:")
    print(f"  Results directory: {results_dir}")
    print(f"  Metrics: {metrics}")
    print(f"  Policies: {policies}")
    print(f"  Summary window: {summary_wd}")
    print()

    try:
        dfs = analyze_sensitivity(results_dir, metrics, policies, summary_wd)

        # Display results
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)

        for metric, df in dfs.items():
            print(f"\n{metric}:")
            print(f"  Shape: {df.shape}")
            print(f"  Index levels: {df.index.names}")
            print(f"  Columns: {df.columns.tolist()}")
            print("\nFirst few rows:")
            print(df.head(10))

            # Example queries
            print(f"\nExample: All results for term_length=2000:")
            if 2000 in df.index.get_level_values("term_length"):
                print(df.loc[2000].head())

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
