import os

import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
beliefs_plots_path = path_dict["plots"] + "beliefs/"
os.makedirs(beliefs_plots_path, exist_ok=True)
specs = read_and_derive_specs(path_dict["specs"])
show_plots = input("Show plots? (y/n): ").strip().lower() == "y"

# data plots
from beliefs.soep_is.belief_data_plots import (
    plot_erp_beliefs_by_cohort,
    plot_erp_violin_plots_by_cohort,
    plot_informed_share_by_cohort,
    plot_sra_beliefs_by_cohort,
)

plot_sra_beliefs_by_cohort(path_dict, show=show_plots)
plt.savefig(
    beliefs_plots_path + "sra_beliefs_by_cohort.png", bbox_inches="tight", transparent=True, dpi=300,
)
plot_erp_beliefs_by_cohort(path_dict, show=show_plots)
plt.savefig(
    beliefs_plots_path + "erp_beliefs_by_cohort.png", bbox_inches="tight", transparent=True, dpi=300,
)
plot_erp_violin_plots_by_cohort(path_dict, censor_above=25, show=show_plots)
plt.savefig(
    beliefs_plots_path + "erp_violin_plots_by_cohort.png", bbox_inches="tight", transparent=True, dpi=300,
)
plot_informed_share_by_cohort(path_dict, show=show_plots)
plt.savefig(
    beliefs_plots_path + "informed_share_by_cohort.png", bbox_inches="tight", transparent=True, dpi=300,
)

# sra plots
from beliefs.belief_formalization.sra_plots import (
    plot_example_sra_evolution,
    plot_expected_sra_vs_birth_year,
    plot_truncated_normal_for_response,
)

responses = [[10, 60, 30],[30, 40, 30], [20, 50, 30]]
for response in responses:
    response_str = "_".join(map(str, response))
    plot_truncated_normal_for_response(response, options=specs, show=show_plots)
    plt.savefig(
        beliefs_plots_path + f"truncated_normal_for_response_{response_str}.png", transparent=True, dpi=300,
        bbox_inches="tight",
    )
plot_expected_sra_vs_birth_year(df=None, path_dict=path_dict, show=show_plots)
plt.savefig(
    beliefs_plots_path + "expected_sra_vs_birth_year.png", bbox_inches="tight", transparent=True, dpi=300,
)

plot_example_sra_evolution(
    alpha_star=0.0,
    SRA_30=67,
    resolution_age=63,
    use_estimated_params=True,
    path_dict=path_dict,
    show=show_plots,
)
plt.savefig(
    beliefs_plots_path + "example_sra_evolution_no_increase.png", transparent=True, dpi=300,
    bbox_inches="tight",
)


# erp plots
from beliefs.belief_formalization.erp_plots import plot_predicted_vs_actual_informed_share

plot_predicted_vs_actual_informed_share(path_dict, specs, show=show_plots)
plt.savefig(
    beliefs_plots_path + "predicted_vs_actual_informed_share.png", transparent=True, dpi=300,
    bbox_inches="tight",
)
