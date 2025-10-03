# First Step Estimation Runner
from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs

input_str = input(
    "\n\n Which of the following steps do you want to estimate? Please type the corresponding letter. \n"
    "\n   - [a]ll"
    "\n   - [w]age"
    "\n   - [p]artner wage"
    "\n   - [j]ob separation"
    "\n   - [f]amily transition"
    "\n   - [h]ealth transition"
    "\n   - [m]ortality estimation"
    "\n   - [c]redited periods estimation"
    "\n"
)

LOAD_DATA = True
# Set define user only to true if we need raw soep data
define_user = True if not LOAD_DATA else False

paths_dict = create_path_dict(define_user=define_user)
specs = read_and_derive_specs(paths_dict["specs"])

if input_str in ["a", "w"]:
    from first_step_estimation.estimation.wage_estimation import estimate_wage_parameters
    estimate_wage_parameters(paths_dict, specs)

if input_str in ["a", "p"]:
    from first_step_estimation.estimation.partner_wage_estimation import estimate_partner_wage_parameters
    estimate_partner_wage_parameters(paths_dict, specs)

if input_str in ["a", "j"]:
    from first_step_estimation.estimation.job_sep_estimation import est_job_sep
    est_job_sep(paths_dict, specs, load_data=LOAD_DATA)

if input_str in ["a", "f"]:
    from first_step_estimation.estimation.family_estimation import (
        estimate_nb_children,
        estimate_partner_transitions,
    )
    estimate_partner_transitions(paths_dict, specs, load_data=LOAD_DATA)
    estimate_nb_children(paths_dict, specs)

if input_str in ["a", "h"]:
    from first_step_estimation.estimation.health_estimation import estimate_health_transitions_parametric
    estimate_health_transitions_parametric(paths_dict, specs)

if input_str in ["a", "m"]:
    from first_step_estimation.estimation.mortality_estimation import estimate_mortality
    estimate_mortality(paths_dict, specs)

if input_str in ["a", "c"]:
    from first_step_estimation.estimation.credited_periods_estimation import calibrate_credited_periods
    calibrate_credited_periods(paths_dict, load_data=LOAD_DATA)
