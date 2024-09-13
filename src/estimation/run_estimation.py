# Set paths of project
from model_code.derive_specs import read_and_derive_specs
from set_paths import create_path_dict


input_str = input(
    "\n\n Which of the following steps do you want to estimate? Please type the corresponding letter. "
    "\n\n 1 First step ('f' for all in this category):"
    "\n   - [s]ra process"
    "\n   - [w]age"
    "\n   - [p]artner wage"
    "\n   - [j]ob separation"
    "\n   - family [t]ransition"
    "\n\n 2 Estimate [m]odel."
    "\n\n Input: "
)
# Set define user only to true if estimate SRA process as we need raw soep data there
define_user = True if input_str in ["f", "s"] else False

paths_dict = create_path_dict(define_user=define_user)
specs = read_and_derive_specs(paths_dict["specs"])

if input_str == "f" or input_str == "s":
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

if input_str == "f" or input_str == "w":
    # Estimate wage parameters
    # Average wage parameters are estimated to compute education-specific pensions
    from estimation.first_step_estimation.est_wage_equation import (
        estimate_wage_parameters,
        estimate_average_wage_parameters,
    )

    estimate_wage_parameters(paths_dict)
    estimate_average_wage_parameters(paths_dict)

if input_str == "f" or input_str == "p":
    # Estimate partner wage parameters for men and women
    from estimation.first_step_estimation.est_partner_wage_equation import (
        estimate_partner_wage_parameters,
        calculate_partner_hours,
    )

    estimate_partner_wage_parameters(paths_dict, specs, est_men=True)
    estimate_partner_wage_parameters(paths_dict, specs, est_men=False)
    calculate_partner_hours(paths_dict)

if input_str == "f" or input_str == "j":
    # Estimate job separation
    from estimation.first_step_estimation.est_job_sep import est_job_sep

    est_job_sep(paths_dict, specs, load_data=True)

if input_str == "f" or input_str == "t":
    # Estimate partner transitions
    from estimation.first_step_estimation.est_family_transitions import (
        estimate_partner_transitions,
        estimate_nb_children,
    )

    estimate_partner_transitions(paths_dict, specs)
    estimate_nb_children(paths_dict, specs)

if input_str == "m":
    from estimation.estimate_setup import estimate_model

    estimation_results = estimate_model(paths_dict, load_model=True)
    print(estimation_results)


# %%
