# %% Set paths of project
import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from simulation.figures.sra_increase import (
    sra_increase_aggregate_plot,
    sra_increase_aggregate_plot_by_het,
)
from simulation.tables.baseline_expectation import (
    generate_baseline_expectation_table_for_all_types,
)
from simulation.tables.ex_post import generate_ex_post_table_for_all_types
from simulation.tables.sra_increase_table import sra_increase_table
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

model_name = specs["model_name"]

# Generate tables
sra_increase_table(path_dict, model_name)
generate_baseline_expectation_table_for_all_types(path_dict, specs, model_name)
generate_ex_post_table_for_all_types(path_dict, specs, model_name)

# Generate plots
sra_increase_aggregate_plot(path_dict, model_name)
sra_increase_aggregate_plot_by_het(
    path_dict=path_dict,
    fig_name="by_gender",
    model_name=model_name,
)
# Generate and print table
# announcement_timing_lc_plot(path_dict, model_name)
# debias_lc_plot(path_dict, model_name)
# commitment_lc_plot(path_dict, model_name)

# plt.show()
