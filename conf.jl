
module CriminosConfigs
using Base.Filesystem, Dates, YAML, Criminos
f_config = nothing
if length(ARGS) == 0
    println("using configuration file: conf.yaml")
    f_config = "conf.yaml"
else
    println("using configuration file: $(ARGS[1])")
    f_config = ARGS[1]
end

# get the current date and format it as yyyymmdd
current_date = Dates.format(now(), "yyyymmdd/HHMMSS")

# Create the folder
folder_name = current_date
result_dir = "result-$folder_name"
mkpath(result_dir)

println(repeat("-", 80))
println("result_dir: $result_dir")
println(repeat("-", 80))
variables_from_yaml = YAML.load_file(f_config)
variables_from_yaml["output_dir"] = result_dir
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
R = variables_from_yaml["R"]
group_size = variables_from_yaml["group_size"]
group_new_ratio = variables_from_yaml["group_new_ratio"]
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

style_name = @sprintf "%s" cc.style_decision_name

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
ℜ = cc.R
group_size = cc.group_size
group_new_ratio = cc.group_new_ratio