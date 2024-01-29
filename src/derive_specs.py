import numpy as np
from process_data.steps.est_ret_age_expectations import (
    estimate_policy_expectation_parameters,
)


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
    policy_expectation_params = estimate_policy_expectation_parameters(
        project_paths, options, load_data=load_data
    )
    policy_step_size = policy_expectation_params.iloc[1, 0]
    add_policy_states = np.ceil((67 - options["min_SRA"]) / policy_step_size)
    options["n_possible_policy_states"] = (
        add_policy_states + options["resolution_age"] - options["start_age"] + 1
    )
    return options
