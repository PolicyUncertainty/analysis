import numpy as np
from model_code.belief_process import exp_ret_age_transition_matrix
from process_data.steps.est_SRA_expectations import estimate_truncated_normal
from process_data.steps.est_SRA_random_walk import gen_exp_val_params_and_plot
from process_data.steps.est_SRA_random_walk import gen_var_params_and_plot
from process_data.steps.est_wage_equation import (
    estimate_wage_parameters,
)
from process_data.steps.gather_decision_data import gather_decision_data


def generate_derived_specs(options):
    # Number of periods in model
    options["n_periods"] = options["end_age"] - options["start_age"] + 1
    # you can retire from min retirement age until max retirement age
    options["n_possible_ret_ages"] = options["max_ret_age"] - options["min_ret_age"] + 1
    options["n_policy_states"] = int(
        ((options["max_SRA"] - options["min_SRA"]) / options["SRA_grid_size"]) + 1
    )
    options["SRA_values_policy_states"] = np.arange(
        options["min_SRA"],
        options["max_SRA"] + options["SRA_grid_size"],
        options["SRA_grid_size"],
    )

    return options


def generate_derived_and_data_derived_options(options, project_paths, load_data=True):
    options = generate_derived_specs(options)

    # Generate dummy transition matrix

    alpha_hat = gen_exp_val_params_and_plot(project_paths, None, load_data=load_data)
    sigma_sq_hat = gen_var_params_and_plot(project_paths, None, load_data=load_data)

    options["beliefs_trans_mat"] = exp_ret_age_transition_matrix(
        options=options,
        alpha_hat=alpha_hat,
        sigma_sq_hat=sigma_sq_hat,
    )

    # Generate number of policy states between 67 and min_SRA
    wage_params = estimate_wage_parameters(project_paths, options, load_data=load_data)
    options["gamma_0"] = wage_params.loc["constant", "parameter"]
    options["gamma_1"] = wage_params.loc["full_time_exp", "parameter"]
    options["gamma_2"] = wage_params.loc["full_time_exp_sq", "parameter"]

    # # Max experience
    data_decision = gather_decision_data(project_paths, options, load_data=load_data)
    options["max_init_experience"] = (
        data_decision["experience"] - data_decision["period"]
    ).max()
    return options
