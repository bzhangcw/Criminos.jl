import os
import matplotlib.pyplot as plt
import imageio


def plot_histograms(grouped_results, *counters):
    all_values, counts_N_full, counts_Np_full, counts_R_full, *_ = counters
    num_plots = len(grouped_results)
    if num_plots > 1:
        for i in range(0, num_plots, plots_per_fig):
            fig, axes = plt.subplots(2, 2, figsize=(12, 6))
            axes = axes.flatten()

            for j in range(plots_per_fig):
                if i + j < num_plots:
                    ax = axes[j]
                    result = grouped_results[i + j]
                    result.plot(kind="bar", ax=ax, width=0.8)
                    ax.set_title(f"$t={i + j}$")
                    ax.set_xlabel("$j$")
                    ax.set_ylabel("individuals")
    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        axes = [axes]
        ax = axes[0]
        result = grouped_results[-1].rename(
            columns={"$y$": r"$y^\texttt{a}_{t+1}$", "$x$": r"$x^\texttt{a}_{t}$"}
        )
        result.plot(kind="bar", ax=ax, width=0.8)
        ax.set_xlabel("$j$")
        ax.set_ylabel("individuals")

    ax = axes[-1]
    result = grouped_results[-1]
    ax.plot(
        counts_Np_full,
        label=r"${x^\texttt{s}_t}$",
        color="red",
        alpha=0.5,
        linestyle="--",
        linewidth=4,
    )
    ax.plot(
        counts_R_full,
        label=r"${y^\texttt{s}_{t+1}}$",
        color="blue",
        alpha=0.5,
        linestyle="--",
        linewidth=4,
    )
    ax.legend()
    plt.tight_layout()
    return fig


import plotly.graph_objs as go


def plot_grouped_histograms_with_slider(grouped_results, *counters):
    num_plots = len(grouped_results)

    # Create a figure
    fig = go.Figure()

    # Add all grouped bar charts to the figure but make them all invisible except the first one
    for i, result in enumerate(grouped_results):
        categories = list(result.index)
        values1 = result["$y$"]
        values2 = result["$x$"]

        fig.add_trace(go.Bar(x=categories, y=values1, name=f"y", visible=(i == 0)))
        fig.add_trace(go.Bar(x=categories, y=values2, name=f"x", visible=(i == 0)))

    # Create slider steps
    steps = []
    for i in range(num_plots):
        step = dict(
            method="update",
            args=[
                {
                    "visible": [False] * 2 * num_plots,  # Hide all traces first
                    "title": f"Time t={i}",
                }
            ],
            label=f"t={i}",
        )
        step["args"][0]["visible"][2 * i] = True  # Show the first bar of the i-th plot
        step["args"][0]["visible"][
            2 * i + 1
        ] = True  # Show the second bar of the i-th plot
        steps.append(step)

    # Create the slider
    sliders = [
        dict(active=0, currentvalue={"prefix": "Time: "}, pad={"t": 50}, steps=steps)
    ]

    # Update layout with slider and bar mode
    fig.update_layout(
        sliders=sliders,
        title_text="Evolution",
        xaxis_title="j: Prior convictions",
        yaxis_title="Value",
        barmode="group",  # Group bars side by side
        height=600,
        width=800,
    )

    # Show the figure​
    fig.show()
    return fig


def plot_grouped_histograms_with_gif(
    grouped_results, *counters, gif_name="animation.gif"
):

    num_plots = len(grouped_results)
    fig = go.Figure()

    # Add all grouped bar charts to the figure but set them invisible initially
    for i, result in enumerate(grouped_results):
        categories = list(result.index)
        values1 = result["$y$"]
        values2 = result["$x$"]

        fig.add_trace(go.Bar(x=categories, y=values1, name=f"y", visible=False))
        fig.add_trace(go.Bar(x=categories, y=values2, name=f"x", visible=False))

    # Create a directory to save frames
    os.makedirs("temp_frames", exist_ok=True)
    images = []

    for i in range(num_plots):
        # Set all traces to invisible
        for trace in fig.data:
            trace.visible = False

        # Make the current pair of bars visible
        fig.data[2 * i].visible = True  # y
        fig.data[2 * i + 1].visible = True  # x

        # Update layout title
        fig.update_layout(
            title_text=f"Time t={i}",
            xaxis_title="j: Prior convictions",
            yaxis_title="Value",
            barmode="group",
            height=600,
            width=800,
        )

        # Save the current frame as an image
        filename = f"temp_frames/frame_{i:03d}.png"
        fig.write_image(filename)
        images.append(imageio.imread(filename))

    # Create a GIF from the saved images
    imageio.mimsave(gif_name, images, fps=5)

    return fig


def plot_grouped_histograms_with_gif_plt(
    grouped_results, *counters, gif_name="animation.gif"
):
    num_plots = len(grouped_results)

    # Create a directory to save frames
    os.makedirs("temp_frames", exist_ok=True)
    images = []

    for i, result in enumerate(grouped_results):
        categories = list(result.index)
        values1 = result["$y$"]
        values2 = result["$x$"]

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot bars
        ax.bar(categories, values1, label="$y$", alpha=0.6)
        ax.bar(categories, values2, label="$x$", alpha=0.6)

        # Set title and labels
        ax.set_title(f"$t$={i}")
        # ax.set_xlabel("$j$: Prior convictions")
        # ax.set_ylabel("Value")
        ax.legend()

        # Save the frame
        filename = f"temp_frames/frame_{i:03d}.png"
        plt.savefig(filename, dpi=100)
        images.append(imageio.imread(filename))

        # Close the plot to avoid memory issues
        plt.close()

    # Create a GIF from the saved images
    imageio.mimsave(gif_name, images, fps=2)

    return gif_name
