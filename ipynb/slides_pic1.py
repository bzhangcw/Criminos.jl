# Calculate the unique values and counts for column 'N'
unique_N, counts_N = np.unique(df["N"], return_counts=True)
value_counts_N = dict(zip(unique_N, counts_N))
values_N = list(value_counts_N.keys())
counts_N = list(value_counts_N.values())

# Calculate the unique values and counts for column 'Np'
unique_Np, counts_Np = np.unique(df["Np"], return_counts=True)
value_counts_Np = dict(zip(unique_Np, counts_Np))
values_Np = list(value_counts_Np.keys())
counts_Np = list(value_counts_Np.values())

# Calculate the counts of Recidivism_Arrest_Year1 grouped by Np
grouped_recidivism = df.groupby("Np")["Recidivism_Arrest_Year1"].sum()

# Ensure that all columns cover the same x-ticks
all_values = sorted(
    set(values_N).union(set(values_Np)).union(set(grouped_recidivism.index))
)
counts_N_full = [value_counts_N.get(x, 0) for x in all_values]
counts_Np_full = [value_counts_Np.get(x, 0) for x in all_values]
counts_R_full = [grouped_recidivism.get(x, 0) for x in all_values]

# Set the width of the bars
bar_width = 0.25
x = np.arange(len(all_values))

# Start creating the grouped bar plot
fig, ax1 = plt.subplots(figsize=(11, 4))

# Plot grouped bars for N, Np, and Recidivism_Arrest_Year1
ax1.bar(
    x - bar_width / 2,
    counts_R_full,
    width=bar_width,
    label=r"\# re-offending ind.",
    edgecolor="black",
)
ax1.bar(
    x + bar_width / 2,
    counts_Np_full,
    width=bar_width,
    label=r"total \# ind.",
    edgecolor="black",
)

# Add labels and legend for the bar plot
ax1.set_xlabel(r"\# prior convictions")
ax1.set_ylabel(r"\# individuals")
ax1.set_xticks(x)
ax1.set_xticklabels(all_values)
ax1.legend(loc="upper right")

# Create a secondary y-axis for the rate of offending
ax2 = ax1.twinx()
ax2.set_ylim(0.1, 0.7)
ax2.set_ylabel(". ")
# Save and show the plot
plt.tight_layout()
plt.savefig("nij-x.pdf", dpi=400)
plt.show()


# Start creating the grouped bar plot
fig, ax1 = plt.subplots(figsize=(11, 4))

# Plot grouped bars for N, Np, and Recidivism_Arrest_Year1
ax1.bar(
    x - bar_width / 2,
    counts_R_full,
    width=bar_width,
    label=r"\# recidivists",
    edgecolor="black",
    alpha=0.5,
)
ax1.bar(
    x + bar_width / 2,
    counts_Np_full,
    width=bar_width,
    label=r"\# individuals",
    edgecolor="black",
    alpha=0.5,
)

# Add labels and legend for the bar plot
ax1.set_xlabel(r"\# prior convictions")
ax1.set_ylabel(r"\# individuals")
ax1.set_xticks(x)
ax1.set_xticklabels(all_values)
# ax1.legend(loc='upper left')

# Create a secondary y-axis for the rate of offending
ax2 = ax1.twinx()
rate_of_offending = [
    counts_R_full[idx] / counts_Np_full[idx] for idx in range(len(counts_R_full) - 1)
]
ax2.plot(
    x[0:-1], rate_of_offending, marker="o", label=r"Rate of re-offending", linewidth=2.0
)
ax2.set_ylabel("Rate of offending")
ax2.set_ylim(0.1, 0.7)

# Add a legend for the secondary axis
ax2.legend(loc="upper right")

# Save and show the plot
plt.tight_layout()
plt.savefig("nij-xr.pdf", dpi=400)
plt.show()
