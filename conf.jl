using Base.Filesystem
using Dates, YAML

# get the current date and format it as yyyymmdd
current_date = Dates.format(now(), "yyyymmdd/HHMM")

# Create the folder
folder_name = current_date

################################################################################
# !!!todo, change to argparse
################################################################################
result_dir = "result-$folder_name"
mkpath(result_dir)

println(repeat("-", 80))
println("result_dir: $result_dir")
println(repeat("-", 80))

# style_retention = :rand
# style_correlation = :uppertriangular
# style_correlation_seed = rand
# style_correlation_psd = true # whether to ensure psd
# style_correlation_subp = false # whether to use subpopulation correlation
# style_mixin = Criminos.mixed_in_gnep_best!
# style_mixin_name = style_mixin |> nameof
# style_decision = Criminos.decision_identity
# # style_decision = Criminos.decision_matching
# # style_decision = Criminos.decision_matching_lh
# style_decision_name = style_decision |> nameof
# bool_use_html = true
# bool_init = true
# bool_conv = true
# bool_compute = false
# bool_plot_trajectory = false
# bool_plot_surface = false

module CriminosConfigs
using YAML, Criminos
variables_from_yaml = YAML.load_file("conf.yaml")
style_retention = Symbol(variables_from_yaml["style_retention"])
style_correlation = Symbol(variables_from_yaml["style_correlation"])
style_correlation_seed = eval(variables_from_yaml["style_correlation_seed"] |> Symbol)
style_correlation_psd = variables_from_yaml["style_correlation_psd"]
style_correlation_subp = variables_from_yaml["style_correlation_subp"]
style_mixin = getfield(Criminos, Symbol(variables_from_yaml["style_mixin"]))
style_mixin_name = style_mixin |> nameof
style_decision = getfield(Criminos, Symbol(variables_from_yaml["style_decision"]))
style_decision_name = style_decision |> nameof
bool_use_html = variables_from_yaml["bool_use_html"]
bool_init = variables_from_yaml["bool_init"]
bool_conv = variables_from_yaml["bool_conv"]
bool_compute = variables_from_yaml["bool_compute"]
bool_plot_trajectory = variables_from_yaml["bool_plot_trajectory"]
bool_plot_surface = variables_from_yaml["bool_plot_surface"]
α₁ = variables_from_yaml["α₁"]
α₂ = variables_from_yaml["α₂"]
seed_number = variables_from_yaml["seed_number"]
end

cc = CriminosConfigs
correlation_styles = [
    :none,
    :diagonal,
    :uppertriangular,
    :symmetric
]

style_arr = (
    cc.style_retention,
    cc.style_correlation,
    cc.style_correlation_subp,
    cc.style_mixin_name,
    cc.style_decision_name
)

style_name = Printf.format(join(["%s" for _ in style_arr], "-") |> Printf.Format, style_arr...)
style_disp = Printf.format(join(["- %s" for _ in style_arr], "\n") |> Printf.Format, style_arr...)

println(repeat("-", 80))
println("Simulation Style Configs:\n")
display(sort(cc.variables_from_yaml))
println(repeat("-", 80))

if cc.bool_use_html
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
Random.seed!(cc.seed_number)
K = 2000           # number of maximum iterations
n = 8               # state size: 0, 1, ..., n-1
# number of subpopulations
ℜ = 2
group_size = [1:5:5*ℜ...]
group_new_ratio = [1:3:3*ℜ...]
# group_size = reverse!([1:5:5*ℜ...])