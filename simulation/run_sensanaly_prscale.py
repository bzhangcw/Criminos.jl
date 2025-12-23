"""
Example usage of the prison scale factor sensitivity analysis script.

This script demonstrates how to:
1. Run the sensitivity analysis
2. Access and filter results
3. Save results to CSV files
4. Generate plots for all metrics
"""

from sensanaly_prscale import analyze_sensitivity, plot_all_term_lengths
import pandas as pd
import os

# Configuration
results_dir = "result-ofpl-scl-4"
metrics = [
    "offense_rate",
    "incarceration_rate",
]
policies = ["high-risk", "low-risk", "age-first"]
summary_wd = 20  # Window size for computing equilibrium (last N periods)

print("=" * 60)
print("Running Prison Scale Factor Sensitivity Analysis")
print("=" * 60)
print(f"Results directory: {results_dir}")
print(f"Metrics: {metrics}")
print(f"Policies: {policies}")
print(f"Summary window: {summary_wd} periods")
print()

# Run analysis
dfs = analyze_sensitivity(results_dir, metrics, policies, summary_wd)

print("\n" + "=" * 60)
print("Analysis Complete - Saving Results")
print("=" * 60)

# Create output directory inside results_dir
output_dir = os.path.join(results_dir, "sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)

# Save each metric to a separate CSV file
for metric, df in dfs.items():
    filename = os.path.join(output_dir, f"sensitivity_{metric}.csv")
    df.to_csv(filename)
    print(f"Saved {filename} (shape: {df.shape})")

print("\n" + "=" * 60)
print("Generating Plots")
print("=" * 60)

# Generate plots for all metrics
plot_all_term_lengths(
    dfs=dfs,
    output_dir=output_dir,
    policies=policies,
    show_std=True,
)

print("\n" + "=" * 60)
print(f"All results saved to {output_dir}/")
print("=" * 60)
