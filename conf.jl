
################################################################################
# !!!todo, change to argparse
################################################################################
correlation_styles = [
    :none,
    :diagonal,
    :uppertriangular,
    :symmetric
]
style_retention = :rand
style_correlation = :uppertriangular
style_correlation_seed = rand
style_correlation_psd = true # whether to ensure psd
style_correlation_subp = true # whether to use subpopulation correlation
style_mixin = Criminos.mixed_in_gnep_best!
style_mixin_name = style_mixin |> nameof
# style_decision = Criminos.decision_matching_lh
style_decision = Criminos.decision_identity
style_decision_name = style_decision |> nameof

style_arr = (
    style_retention,
    style_correlation,
    style_correlation_subp,
    style_mixin_name,
    style_decision_name
)
style_name = Printf.format(join(["%s" for _ in style_arr], "-") |> Printf.Format, style_arr...)
style_disp = Printf.format(join(["- %s" for _ in style_arr], "\n") |> Printf.Format, style_arr...)

println(repeat("-", 80))
println("style_name:\n$style_disp")
println(repeat("-", 80))

bool_use_html = true
bool_init = true
bool_compute = true
bool_plot_trajectory = true
bool_plot_surface = false
if bool_use_html
    plotlyjs()
    format = "html"
else
    pgfplotsx()
    format = "pdf"
end
ratio_group = 1 # ratio of trajectories to be plotted

# -----------------------------------------------------------------------------
# problem size
# -----------------------------------------------------------------------------
Random.seed!(5)
K = 10000           # number of maximum iterations
n = 8               # state size: 0, 1, ..., n-1
# number of subpopulations
ℜ = 2
group_size = [1:5:5*ℜ...]