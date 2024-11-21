# Set paths of project
from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs


input_str = input(
    "\n\n Which of the following steps do you want to estimate? Please type the corresponding letter. "
    "\n   - [s]ra process"
    "\n   - [w]age"
    "\n   - [p]artner wage"
    "\n   - [j]ob separation"
    "\n   - family [t]ransition"
    "\n   - [h]ealth transition"
    "\n   - [i]nformed state transition"
    "\n"
)
# Set define user only to true if estimate SRA process as we need raw soep data there
define_user = True if input_str in ["s", "i"] else False

paths_dict = create_path_dict(define_user=define_user)
specs = read_and_derive_specs(paths_dict["specs"])

if input_str == "s":
    # Estimate parameters of SRA truncated normal distributions
    from estimation.first_step_estimation.est_SRA_expectations import (
        estimate_truncated_normal,
    )

    df_exp_policy_dist = estimate_truncated_normal(paths_dict, specs, load_data=False)

    # Estimate SRA random walk
    from estimation.first_step_estimation.est_SRA_random_walk import (
        est_SRA_params,
    )

    est_SRA_params(paths_dict)

if input_str == "w":
    # Estimate wage parameters
    # Average wage parameters are estimated to compute education-specific pensions
    from estimation.first_step_estimation.est_wage_equation import (
        estimate_wage_parameters,
    )

    estimate_wage_parameters(paths_dict, specs)

if input_str == "p":
    # Estimate partner wage parameters for men and women
    from estimation.first_step_estimation.est_partner_wage_equation import (
        estimate_partner_wage_parameters,
    )

    estimate_partner_wage_parameters(paths_dict, specs, est_men=True)
    estimate_partner_wage_parameters(paths_dict, specs, est_men=False)
    # calculate_partner_hours(paths_dict)

if input_str == "j":
    # Estimate job separation
    from estimation.first_step_estimation.est_job_sep import est_job_sep

    est_job_sep(paths_dict, specs, load_data=True)

if input_str == "t":
    # Estimate partner transitions
    from estimation.first_step_estimation.est_family_transitions import (
        estimate_partner_transitions,
        estimate_nb_children,
    )

    estimate_partner_transitions(paths_dict, specs)
    estimate_nb_children(paths_dict, specs)

if input_str == "h":
    # Estimate health transitions
    from estimation.first_step_estimation.est_health_transition import (
        estimate_health_transitions, estimate_health_transitions_parametric
    )

    #estimate_health_transitions(paths_dict, specs)
    estimate_health_transitions_parametric(paths_dict, specs)

if input_str == "i":
    # Estimate informed state transition
    from estimation.first_step_estimation.est_informed_state_transition import (
        calibrate_uninformed_hazard_rate,
    )

    calibrate_uninformed_hazard_rate(paths_dict, specs)

# %%
