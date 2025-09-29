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

plot_detailed_lifecycle_results(
    df_results_path=baseline_data_path,
    df_results_comparison_path=comparison_data_path,
    path_dict=path_dict,
    model_name=model_name,
    specs=specs,
    subfolder="baseline",
    comparison_name="true_SRA_known",
    show=show_plots,
    save=save_plots
)


# Create detailed lifecycle plots for counterfactual scenario (SRA 69, with uncertainty and misinformation vs. no uncertainty)
from simulation.figures.detailed_lc_results import plot_detailed_lifecycle_results

baseline_data_path = path_dict["simulation_data"] + "/sra_69/" + f"sra_69_lc_{model_name}.csv"
comparison_data_path = path_dict["simulation_data"] + "/sra_69/" + f"sra_69_lc_{model_name}_no_uncertainty.csv"

plot_detailed_lifecycle_results(
    df_results_path=baseline_data_path,
    df_results_comparison_path=comparison_data_path,
    path_dict=path_dict,
    specs=specs,
    subfolder="sra_69",
    comparison_name="true_SRA_known",
    show=show_plots,
    save=save_plots
)


# Create detailed lifecycle plots for counterfactual scenario (SRA 69, with uncertainty and misinformation vs. no uncertainty)
from simulation.figures.detailed_lc_results import plot_detailed_lifecycle_results

baseline_data_path = path_dict["simulation_data"] + "/sra_69/" + f"sra_69_lc_{model_name}.csv"
comparison_data_path = path_dict["simulation_data"] + "/sra_69/" + f"sra_69_lc_{model_name}_no_uncertainty.csv"

plot_detailed_lifecycle_results(
    df_results_path=baseline_data_path,
    df_results_comparison_path=comparison_data_path,
    path_dict=path_dict,
    specs=specs,
    subfolder="sra_69",
    comparison_name="true_SRA_known",
    show=show_plots,
    save=save_plots,
)

print("Detailed lifecycle plots completed.")
