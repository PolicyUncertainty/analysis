# %%
from set_paths import create_path_dict
from set_styles import set_plot_defaults
from specs.derive_specs import generate_derived_and_data_derived_specs

# %%
# Set up paths and configurations
path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)
show_plots = False
save_plots = True

# Set plot defaults
set_plot_defaults()

# %%
# Set up data path
model_name = specs["model_name"]

# %%
# Create detailed lifecycle plots for baseline
from simulation.figures.detailed_lc_results import plot_detailed_lifecycle_results
baseline_lc_path = path_dict["simulation_data"] + "/baseline/" + f"baseline_lc_{model_name}.csv"
plot_detailed_lifecycle_results(
    df_results_path=baseline_lc_path,
    path_dict=path_dict,
    specs=specs,
    subfolder="baseline",
    show=show_plots,
    save=save_plots
)

print("Detailed lifecycle plots completed.")