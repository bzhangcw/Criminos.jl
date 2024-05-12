using Plots
pgfplotsx()  # Set PGFPlotsX as the backend

# Define the range of a values
aValues = 0.1:0.2:0.8  # Julia range from 0.1 to 0.8 with step 0.2


# Define names for the legend, using LaTeX syntax directly
names = [L"\mathbf{a(\tau) = %$a}" for a in aValues]

# Define the function
f(y, a) = -3y + a * y^2

final_plot = plot(
    legend=:outertopright,
    xlabel=L"y_j",
    ylabel=L"\omega(y_j, a)",
    xtickfont=font(15),
    ytickfont=font(15),
    legendfontsize=14,
    titlefontsize=22,
    dpi=1000,
)

# Generate the plots
for (i, a) in enumerate(aValues)
    yv = 0:0.1:10
    fv = f.(yv, a)
    plot!(final_plot,
        yv,
        fv,
        label=names[i],
        linewidth=2
    )
end

# Combine the plots into one

# Show the final plot
savefig(final_plot, "result/1d.pdf")
