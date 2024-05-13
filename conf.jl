
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
style_mixin = Criminos.mixed_in_gnep_best
style_mixin_name = style_mixin |> nameof
style_decision = Criminos.decision_matching_lh
style_decision_name = style_decision |> nameof

style_name = @sprintf("%s-%s-%s-%s",
    style_retention,
    style_correlation,
    style_mixin_name,
    style_decision_name
)

println(repeat("-", 80))
println("style_name: $style_name")
println(repeat("-", 80))

bool_use_html = true
bool_init = true
bool_compute = true
bool_plot_trajectory = true
bool_plot_surface = true
if bool_use_html
    plotlyjs()
    format = "html"
else
    pgfplotsx()
    format = "pdf"
end
ratio_group = 1 # ratio of groups to be plotted
# -----------------------------------------------------------------------------
# problem size
# -----------------------------------------------------------------------------
Random.seed!(5)
n = 8
K = 2e3