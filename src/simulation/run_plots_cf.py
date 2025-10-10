# %% Set paths of project
import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from simulation.figures.announcment_timing import announcement_timing_lc_plot
from simulation.figures.commitment import commitment_lc_plot
from simulation.figures.debias import debias_lc_plot
from simulation.figures.sra_increase import (
    sra_increase_aggregate_plot,
    sra_increase_aggregate_plot_by_het,
)
from simulation.tables.baseline_increase import generate_latex_table
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

model_name = specs["model_name"]
het_names = ["men", "women"]

latex_table = generate_latex_table(path_dict, model_name)
# # Exclude the first rpw
sra_increase_aggregate_plot(path_dict, model_name)
sra_increase_aggregate_plot_by_het(
    path_dict=path_dict,
    het_names=het_names,
    fig_name="by_gender",
    model_name=model_name,
)
# Generate and print table
# announcement_timing_lc_plot(path_dict, model_name)
# debias_lc_plot(path_dict, model_name)
# commitment_lc_plot(path_dict, model_name)

plt.show()
