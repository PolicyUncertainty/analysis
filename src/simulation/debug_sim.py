# %%
# Set paths of project
from set_paths import create_path_dict

paths_dict = create_path_dict()

import jax
import pickle as pkl
import numpy as np
from model_code.wealth_and_budget.budget_equation import budget_constraint

jax.config.update("jax_enable_x64", True)

from set_paths import create_path_dict

path_dict = create_path_dict()

# %%
params = pkl.load(open(path_dict["est_params"], "rb"))
from model_code.specify_model import specify_and_solve_model
from model_code.policy_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.policy_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)

solution_subj, model, params = specify_and_solve_model(
    path_dict=paths_dict,
    file_append="subj",
    params=params,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    load_model=True,
    load_solution=True,
)

# solution_00, model, params = specify_and_solve_model(
#     path_dict=paths_dict,
#     file_append="00",
#     params=params,
#     update_spec_for_policy_state=create_update_function_for_slope(0),
#     policy_state_trans_func=realized_policy_step_function,
#     load_model=True,
#     load_solution=True,
# )
period = 55
education = 0
lagged_choice = 2
experience = 40
partner_state = 0
# policy_state = 0
wealth_id = 15
savings_end_of_previous_period = model["exog_savings_grid"][wealth_id]
income_shock_previous_period = 0
job_offer = 0
options = model["options"]
choice = 2

# names: ['period', 'lagged_choice', 'experience', 'education', 'policy_state', 'job_offer', 'partner_state']

for policy_state in range(29):
    all_pol_states = model["model_structure"]["map_state_choice_to_index"][
        period,
        lagged_choice,
        experience,
        education,
        policy_state,
        job_offer,
        partner_state,
        choice,
    ]
    endog_last = solution_subj["endog_grid"][all_pol_states, wealth_id + 1]
    from dcegm.numerical_integration import quadrature_legendre

    income_shock_draws_unscaled, income_shock_weights = quadrature_legendre(
        options["model_params"]["quadrature_points_stochastic"]
    )
    wealth_last = budget_constraint(
        period,
        education,
        lagged_choice,  # d_{t-1}
        experience,
        np.array(partner_state),
        policy_state,  # current applicable SRA identifyer
        savings_end_of_previous_period,  # A_{t-1}
        income_shock_previous_period,  # epsilon_{t - 1}
        params,
        options["model_params"],
    )
    print(wealth_last)
