import pickle

import estimagic as em
import numpy as np
from dcegm.solve import get_solve_func_for_model
from derive_specs import generate_specs_and_update_params
from model_code.budget_equation import create_savings_grid
from model_code.policy_states_belief import exp_ret_age_transition_matrix
from model_code.policy_states_belief import expected_SRA_probs_estimation
from model_code.specify_model import specify_model


def generate_model_to_estimate(
    project_paths,
    start_params_all,
    load_model,
):
    # Generate model_specs
    project_specs, start_params_all = generate_specs_and_update_params(
        project_paths, start_params_all
    )

    # Execute load first step estimation data
    alpha_hat = np.loadtxt(project_paths["est_results"] + "exp_val_params.txt")
    sigma_sq_hat = np.loadtxt(project_paths["est_results"] + "var_params.txt")
    project_specs["beliefs_trans_mat"] = exp_ret_age_transition_matrix(
        options=project_specs,
        alpha_hat=alpha_hat,
        sigma_sq_hat=sigma_sq_hat,
    )

    # Generate dcegm model for project specs
    model, options = specify_model(
        project_specs=project_specs,
        model_data_path=project_paths["intermediate_data"],
        exog_trans_func=expected_SRA_probs_estimation,
        load_model=load_model,
    )
    print("Model specified.")

    return model, options, start_params_all


def process_data_for_dcegm(df, state_space_names):
    data_dict = {}
    # Now transform for dcegm
    data_dict["states"] = {name: df[name].values for name in state_space_names}
    data_dict["wealth"] = df["wealth"].values
    data_dict["choices"] = df["choice"].values
    return data_dict


def solve_estimated_model(project_paths, start_params_all, load_model, load_solution):
    solution_file = project_paths["intermediate_data"] + "est_model_solution.pkl"

    if load_solution:
        solution_est = pickle.load(open(solution_file, "rb"))
        return solution_est

    # Generate model_specs
    model, options, start_params_all = generate_model_to_estimate(
        project_paths=project_paths,
        start_params_all=start_params_all,
        load_model=load_model,
    )
    savings_grid = create_savings_grid()

    solve_func = get_solve_func_for_model(model, savings_grid, options)
    value, policy_left, policy_right, endog_grid = solve_func(start_params_all)

    solution_est = {
        "value": value,
        "policy_left": policy_left,
        "policy_right": policy_right,
        "endog_grid": endog_grid,
    }

    pickle.dump(solution_est, open(solution_file, "wb"))

    return solution_est


def visualize_em_database(db_path):
    fig = em.criterion_plot(db_path)
    fig.show()

    fig = em.params_plot(db_path)
    fig.show()
