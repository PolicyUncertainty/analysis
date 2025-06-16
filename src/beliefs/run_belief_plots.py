import os

import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
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
    path_dict["belief_plots"] + "sra_beliefs_by_cohort.png", bbox_inches="tight"
)
plot_erp_beliefs_by_cohort(path_dict, show=show_plots)
plt.savefig(
    path_dict["belief_plots"] + "erp_beliefs_by_cohort.png", bbox_inches="tight"
)
plot_erp_violin_plots_by_cohort(path_dict, show=show_plots)
plt.savefig(
    path_dict["belief_plots"] + "erp_violin_plots_by_cohort.png", bbox_inches="tight"
)
plot_informed_share_by_cohort(path_dict, show=show_plots)
plt.savefig(
    path_dict["belief_plots"] + "informed_share_by_cohort.png", bbox_inches="tight"
)

# sra plots
from beliefs.sra_beliefs.sra_plots import (
    plot_expected_sra_vs_birth_year,
    plot_truncated_normal_for_response,
)

responses = [[30, 40, 30], [20, 50, 30], [10, 60, 30]]
for response in responses:
    response_str = "_".join(map(str, response))
    plot_truncated_normal_for_response(response, options=specs, show=show_plots)
    plt.savefig(
        path_dict["belief_plots"] + f"truncated_normal_for_response_{response_str}.png",
        bbox_inches="tight",
    )
plot_expected_sra_vs_birth_year(df=None, path_dict=path_dict, show=show_plots)
plt.savefig(
    path_dict["belief_plots"] + "expected_sra_vs_birth_year.png", bbox_inches="tight"
)
