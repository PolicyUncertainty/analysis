import os

import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from set_styles import set_plot_defaults
from specs.derive_specs import generate_derived_and_data_derived_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = generate_derived_and_data_derived_specs(path_dict)
show_plots = False
save_plots = True
paper_plots = True

# Set plot defaults
set_plot_defaults()

# Note: Plots require estimation to be run first to generate necessary data files

# wage plots (requires wage_estimation_sample_with_predictions.csv)
from first_step_estimation.plots.wage_plots import (
    plot_wage_regression_by_experience,
    plot_wage_regression_results,
)

plot_wage_regression_results(
    path_dict, specs, show=show_plots, save=save_plots, paper_plots=paper_plots
)
if not paper_plots:
    plot_wage_regression_by_experience(
        path_dict, specs, show=show_plots, save=save_plots
    )

# partner wage plots (requires partner_wage_estimation_sample_with_predictions.csv)
from first_step_estimation.plots.partner_wage_plots import plot_partner_wage_results

plot_partner_wage_results(
    path_dict, specs, show=show_plots, save=save_plots, paper_plots=paper_plots
)

# credited periods plots (CREATES ITS OWN DATA; TODO: FIX)
from first_step_estimation.plots.credited_periods_plots import (
    plot_credited_periods_results,
)

plot_credited_periods_results(path_dict, specs, show=show_plots, save=save_plots)

from first_step_estimation.plots.children import plot_children

plot_children(path_dict, specs, show=show_plots, paper_plot=paper_plots)

from first_step_estimation.plots.family_plots import plot_partner_shares

plot_partner_shares(path_dict, specs, load_data=True, paper_plot=paper_plots)

# health state plots (requires health_transition_estimation_sample.pkl)
from first_step_estimation.plots.health_states_plots import (
    plot_health_transition_prob,
    plot_healthy_unhealthy,
)

plot_healthy_unhealthy(
    path_dict, specs, show=show_plots, save=save_plots, paper_plot=paper_plots
)
if not paper_plots:
    plot_health_transition_prob(path_dict, specs, show=show_plots, save=save_plots)

# job separation plots (requires job separation sample)
from first_step_estimation.plots.job_separation_plots import plot_job_separations

plot_job_separations(
    path_dict, specs, show=show_plots, save=save_plots, paper_plot=paper_plots
)

# mortality plots (requires mortality estimation results)
from first_step_estimation.plots.mortality_plots import (
    plot_mortality,
    plot_mortality_hazard_ratios,
)

plot_mortality(
    path_dict, specs, show=show_plots, save=save_plots, paper_plot=paper_plots
)
if not paper_plots:
    plot_mortality_hazard_ratios(path_dict, specs, show=show_plots, save=save_plots)
