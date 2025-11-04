# %% Set paths of project
import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from simulation.figures.retirement_bunching import (
    plot_retirement_bunching,
)
from simulation.figures.sra_increase import (
    sra_increase_aggregate_plot,
    sra_increase_aggregate_plot_by_het,
)
from simulation.tables.baseline_expectation import (
    generate_baseline_expectation_table_for_all_types,
)
from simulation.tables.debias_table import aggregate_comparison_baseline_cf
from simulation.tables.ex_post import create_ex_post_ex_ante_table
from simulation.tables.sra_increase_table import (
    sra_increase_table,
    welfare_table,
    welfare_table_2,
)
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

model_name = specs["model_name"]
het_names = ["men", "women"]
het_label = "gender"

# Generate tables
sra_increase_table(path_dict, model_name)
create_ex_post_ex_ante_table(path_dict, specs)

# Bunching
plot_retirement_bunching(
    path_dict=path_dict,
    specs=specs,
    model_name=model_name,
)


# # Generate plots
sra_increase_aggregate_plot(path_dict, model_name)
welfare_table(path_dict, model_name)
welfare_table_2(path_dict, model_name)
aggregate_comparison_baseline_cf(
    path_dict=path_dict,
    model_name=model_name,
    file_append="all",
)
# sra_increase_aggregate_plot_by_het(
#     path_dict=path_dict,
#     fig_name="by_" + het_label,
#     model_name=model_name,
#     het_names=het_names,
# )
# Generate and print table
# announcement_timing_lc_plot(path_dict, model_name)
# debias_lc_plot(path_dict, model_name)
# commitment_lc_plot(path_dict, model_name)

# plt.show()
