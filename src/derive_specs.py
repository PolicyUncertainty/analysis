import numpy as np
from model_code.belief_process import exp_ret_age_transition_matrix
from process_data.steps.est_SRA_expectations import estimate_truncated_normal
from process_data.steps.est_SRA_random_walk import gen_exp_val_params_and_plot
from process_data.steps.est_SRA_random_walk import gen_var_params_and_plot
from process_data.steps.est_wage_equation import (
    estimate_wage_parameters,
)
from process_data.steps.gather_decision_data import gather_decision_data


def generate_derived_specs(specs):
    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # you can retire from min retirement age until max retirement age
    specs["n_possible_ret_ages"] = specs["max_ret_age"] - specs["min_ret_age"] + 1
    specs["n_policy_states"] = int(
        ((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )

    return specs


def generate_derived_and_data_derived_specs(specs, project_paths, load_data=True):
    specs = generate_derived_specs(specs)

    # Generate belief transition matrix
    alpha_hat = gen_exp_val_params_and_plot(project_paths, None, load_data=load_data)
    sigma_sq_hat = gen_var_params_and_plot(project_paths, None, load_data=load_data)
    # Generate policy state steps for individuals who start in 0. First calculate the
    # per year expected increase in policy state
    life_span = specs["end_age"] - specs["start_age"] + 1
    per_period_expec_increase = np.arange(life_span) * alpha_hat
    # Then round to the nearest value, which you can do by multiplying with the
    # inverse of the grid size. In the baseline case 1 / 0.25 = 4
    multiplier = 1 / specs["SRA_grid_size"]
    specs["policy_state_increases"] = np.round(per_period_expec_increase * multiplier)

    specs["beliefs_trans_mat"] = exp_ret_age_transition_matrix(
        options=specs,
        alpha_hat=alpha_hat,
        sigma_sq_hat=sigma_sq_hat,
    )

    # Generate number of policy states between 67 and min_SRA
    wage_params = estimate_wage_parameters(project_paths, specs, load_data=load_data)
    specs["gamma_0"] = wage_params.loc["constant", "parameter"]
    specs["gamma_1"] = wage_params.loc["full_time_exp", "parameter"]
    specs["gamma_2"] = wage_params.loc["full_time_exp_sq", "parameter"]
    specs["income_shock_scale"] = wage_params.loc["income_shock_std", "parameter"]

    # calculate value of pension point based on unweighted average wage over 40 years of work
    exp_grid = np.arange(1, 41)
    wage_grid = (
        specs["gamma_0"]
        + specs["gamma_1"] * exp_grid
        + specs["gamma_2"] * exp_grid**2
    )
    specs["pension_point_value"] = wage_grid.mean() / 40 * 0.48

    # max initial experience
    data_decision = gather_decision_data(project_paths, specs, load_data=load_data)
    specs["max_init_experience"] = (
        data_decision["experience"] - data_decision["period"]
    ).max()
    return specs
