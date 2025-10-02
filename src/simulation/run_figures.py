# %%
import pandas as pd

from set_paths import create_path_dict
from set_styles import set_plot_defaults
from simulation.sim_tools.calc_life_cycle_detailed import calc_life_cycle_detailed
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
# Create detailed lifecycle plots for baseline scenario (SRA 67, with uncertainty and misinformation vs. no uncertainty)
from simulation.figures.detailed_lc_results import plot_detailed_lifecycle_results

baseline_data_path = (
    path_dict["simulation_data"] + "/baseline/" + f"baseline_lc_{model_name}.csv"
)
comparison_data_path = (
    path_dict["simulation_data"]
    + "/baseline/"
    + f"baseline_lc_{model_name}_no_uncertainty.csv"
)

df_baseline = pd.read_csv(baseline_data_path, index_col=[0, 1, 2])
df_comparison = pd.read_csv(comparison_data_path, index_col=[0, 1, 2])

plot_detailed_lifecycle_results(
    df_baseline=df_baseline,
    df_comparison=df_comparison,
    path_dict=path_dict,
    model_name=model_name,
    specs=specs,
    comparison_name="no_reform_expected",
    show=show_plots,
    save=save_plots,
)

print("Detailed lifecycle plots completed.")
