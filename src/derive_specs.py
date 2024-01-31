import numpy as np
from process_data.steps.est_ret_age_expectations import (
    estimate_policy_expectation_parameters,
)
from process_data.steps.est_wage_equation import estimate_wage_parameters
from process_data.steps.gather_decision_data import gather_decision_data


def generate_derived_specs(options):
    # Generate age at which overall you can actually retire
    options["min_ret_age"] = options["min_SRA"] - options["ret_years_before_SRA"]
    # Number of periods in model
    options["n_periods"] = options["end_age"] - options["start_age"] + 1
    # you can retire from min retirement age until max retirement age
    options["n_possible_ret_ages"] = options["max_ret_age"] - options["min_ret_age"] + 1
    return options


def generate_derived_and_data_derived_options(options, project_paths, load_data=True):
    options = generate_derived_specs(options)
    # Generate number of policy states between 67 and min_SRA
    policy_expectation_params = estimate_policy_expectation_parameters(
        project_paths, options, load_data=load_data
    )
    policy_step_size = policy_expectation_params.iloc[1, 0]
    add_policy_states = np.ceil((67 - options["min_SRA"]) / policy_step_size)
    # when you are (start_age) years old, there can be as many policy states as there
    # are years until (resolution_age). These are added to the number of policy states
    #

    options["n_possible_policy_states"] = (
        add_policy_states + options["resolution_age"] - options["start_age"] + 1
    )
    options["belief_update_increment"] = policy_step_size
    # Wage parameters
    wage_params = estimate_wage_parameters(project_paths, options, load_data=load_data)
    options["gamma_0"] = wage_params.loc["constant", "parameter"]
    options["gamma_1"] = wage_params.loc["full_time_exp", "parameter"]
    options["gamma_2"] = wage_params.loc["full_time_exp_sq", "parameter"]

    # Max experience
    data_decision = gather_decision_data(
        project_paths, options, policy_step_size, load_data=load_data
    )
    options["max_init_experience"] = (
        data_decision["experience"] - data_decision["period"]
    ).max()
    return options
