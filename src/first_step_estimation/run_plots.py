import os

import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from set_styles import set_plot_defaults
from specs.derive_specs import read_and_derive_specs

# paths, specs, create directories
path_dict = create_path_dict(define_user=True)
specs = read_and_derive_specs(path_dict["specs"])
show_plots = False
save_plots = True

# Set plot defaults
set_plot_defaults()

# Note: Plots require estimation to be run first to generate necessary data files

# wage plots (requires wage_estimation_sample_with_predictions.csv)
from first_step_estimation.plots.wage_plots import plot_wage_regression_results
plot_wage_regression_results(path_dict, specs, show=show_plots, save=save_plots)

# partner wage plots (requires partner_wage_estimation_sample_with_predictions.csv)
from first_step_estimation.plots.partner_wage_plots import plot_partner_wage_results
plot_partner_wage_results(path_dict, specs, show=show_plots, save=save_plots)

# family transition plots (requires partner_transition_matrix.csv)
from first_step_estimation.plots.family_plots import plot_family_transition_results
plot_family_transition_results(path_dict, specs, show=show_plots, save=save_plots)

# credited periods plots (requires SOEP data access)
from first_step_estimation.plots.credited_periods_plots import plot_credited_periods_results
plot_credited_periods_results(path_dict, specs, show=show_plots, save=save_plots)