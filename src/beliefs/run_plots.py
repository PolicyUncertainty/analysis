import os

import matplotlib.pyplot as plt
import pandas as pd

# erp plots
from beliefs.erp_beliefs.erp_plots import (
    plot_predicted_vs_actual_informed_share,
    plot_predicted_vs_actual_means,
)
from beliefs.erp_beliefs.plots_violins import (
    plot_erp_violin_plots_by_age,
    plot_erp_violin_plots_by_cohort,
)
from beliefs.erp_beliefs.share_fit import (
    plot_predicted_informed_shares_by_education,
)

# data plots
from beliefs.soep_is.belief_data_plots import (
    plot_erp_beliefs_by_cohort,
    plot_informed_share_by_cohort,
)
from beliefs.soep_is.belief_data_plots_by_age import (
    plot_erp_beliefs_by_age,
    plot_informed_share_by_age,
)
from beliefs.sra_beliefs.plots_attr_percentage import (
    plot_sra_beliefs_by_age,
    plot_sra_beliefs_by_cohort,
)

# sra plots
from beliefs.sra_beliefs.sra_plots import (
    plot_alpha_heterogeneity_coefficients_combined,
    plot_expected_sra_vs_birth_year,
    plot_truncated_normal_for_response,
)
from beliefs.sra_beliefs.truncated_fit_ar1 import plot_ar1_fit
from beliefs.sra_beliefs.zepplin import plot_example_sra_evolution
from set_paths import create_path_dict
from set_styles import set_plot_defaults
from specs.derive_specs import read_and_derive_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = read_and_derive_specs(path_dict["specs"])
show_plots = False
save_plots = True

paper_graphs_only = True
# Set plot defaults
set_plot_defaults()

plot_sra_beliefs_by_age(path_dict, show=show_plots, save=save_plots)
plot_erp_violin_plots_by_age(
    path_dict, censor_above=20, show=show_plots, save=save_plots
)
plot_predicted_informed_shares_by_education(
    path_dict, specs, show=show_plots, save=save_plots, by_education=False
)
plot_ar1_fit(
    path_dict,
    show=show_plots,
    save=save_plots,
)

if paper_graphs_only:
    exit()

plot_erp_beliefs_by_age(path_dict, show=show_plots, save=save_plots)
plot_predicted_vs_actual_informed_share(
    path_dict, specs, show=show_plots, save=save_plots
)
plot_informed_share_by_cohort(path_dict, show=show_plots, save=save_plots)

plot_erp_beliefs_by_cohort(path_dict, show=show_plots, save=save_plots)


plot_sra_beliefs_by_cohort(path_dict, show=show_plots, save=save_plots)

plot_erp_violin_plots_by_cohort(
    path_dict, censor_above=20, show=show_plots, save=save_plots
)


plot_informed_share_by_age(path_dict, show=show_plots, save=save_plots)


responses = [[30, 40, 30], [20, 50, 30], [100, 0, 0]]
for response in responses:
    plot_truncated_normal_for_response(
        response, options=specs, show=show_plots, save=save_plots, path_dict=path_dict
    )
plot_expected_sra_vs_birth_year(
    df=None, path_dict=path_dict, show=show_plots, save=save_plots
)

plot_example_sra_evolution(
    alpha_star=0.0,
    SRA_30=67,
    resolution_age=63,
    use_estimated_params=True,
    path_dict=path_dict,
    show=show_plots,
    save=save_plots,
)

# sra heterogeneity plots
plot_alpha_heterogeneity_coefficients_combined(
    path_dict=path_dict, show=show_plots, save=save_plots
)
plot_predicted_vs_actual_means(
    path_dict, specs, show=show_plots, save=save_plots, censor_above=20
)
