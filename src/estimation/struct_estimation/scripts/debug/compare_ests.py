# # %% Set paths of project
# import pickle
#
# import numpy as np
# from model_code.policy_processes.policy_states_belief import (
#     expected_SRA_probs_estimation,
# )
# from model_code.policy_processes.policy_states_belief import (
#     update_specs_exp_ret_age_trans_mat,
# )
# from model_code.specify_model import specify_and_solve_model
# from model_code.unobserved_state_weighting import create_unobserved_state_specs
# from set_paths import create_path_dict
# from specs.derive_specs import generate_derived_and_data_derived_specs
#
# path_dict = create_path_dict()
#
# specs = generate_derived_and_data_derived_specs(path_dict)
#
#
# from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
#     load_and_set_start_params,
# )
#
# params_pre = load_and_set_start_params(path_dict)
#
# est_model_start, model, params_pre = specify_and_solve_model(
#     path_dict=path_dict,
#     params=params_pre,
#     update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
#     policy_state_trans_func=expected_SRA_probs_estimation,
#     file_append="start",
#     load_model=True,
#     load_solution=True,
# )
#
# params_post = pickle.load(open(path_dict["est_results"] + "est_params_pete.pkl", "rb"))
# est_model_post, _, params_post = specify_and_solve_model(
#     path_dict=path_dict,
#     params=params_post,
#     update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
#     policy_state_trans_func=expected_SRA_probs_estimation,
#     file_append="pete",
#     load_model=True,
#     load_solution=True,
# )
#
# from estimation.struct_estimation.scripts.observed_model_fit import (
#     load_and_prep_data_for_model_fit,
#     choice_probs_for_choice_vals,
# )
#
# data_decision, states_dict = load_and_prep_data_for_model_fit(
#     path_dict, specs, params_post, model, drop_retirees=True
# )
#
# unobserved_state_specs = create_unobserved_state_specs(data_decision, model)
#
# for choice in range(specs["n_choices"]):
#     choice_vals = np.ones_like(data_decision["choice"].values) * choice
#
#     choice_probs_observations = choice_probs_for_choice_vals(
#         choice_vals=choice_vals,
#         states_dict=states_dict,
#         model=model,
#         unobserved_state_specs=unobserved_state_specs,
#         params=params_pre,
#         est_model=est_model_start,
#         use_probability_of_observed_states=False,
#     )
#
#     choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
#     data_decision[f"start_{choice}"] = choice_probs_observations
#
# data_decision["start_ll"] = choice_probs_for_choice_vals(
#     choice_vals=data_decision["choice"].values,
#     states_dict=states_dict,
#     model=model,
#     unobserved_state_specs=unobserved_state_specs,
#     params=params_pre,
#     est_model=est_model_start,
#     use_probability_of_observed_states=True,
# )
#
#
# for choice in range(specs["n_choices"]):
#     choice_vals = np.ones_like(data_decision["choice"].values) * choice
#
#     choice_probs_observations = choice_probs_for_choice_vals(
#         choice_vals=choice_vals,
#         states_dict=states_dict,
#         model=model,
#         unobserved_state_specs=unobserved_state_specs,
#         params=params_post,
#         est_model=est_model_post,
#         use_probability_of_observed_states=False,
#     )
#
#     choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
#     data_decision[f"post_{choice}"] = choice_probs_observations
#
#
# data_decision["post_ll"] = choice_probs_for_choice_vals(
#     choice_vals=data_decision["choice"].values,
#     states_dict=states_dict,
#     model=model,
#     unobserved_state_specs=unobserved_state_specs,
#     params=params_post,
#     est_model=est_model_post,
#     use_probability_of_observed_states=True,
# )
# data_decision["ll_diff"] = data_decision["start_ll"] - data_decision["post_ll"]
#
# # for choice in range(specs["n_choices"]):
# #     for informed in range(2):
# #         for job_offer in range(2):
# #             choice_vals = np.ones_like(data_decision["choice"].values) * choice
# #
# #             states_dict["informed"][:] = informed
# #             states_dict["job_offer"][:] = job_offer
# #
# #             choice_probs_post = choice_probs_for_choice_vals(
# #                 choice_vals=choice_vals,
# #                 states_dict=states_dict,
# #                 model=model,
# #                 # unobserved_state_specs=unobserved_state_specs,
# #                 params=params_post,
# #                 est_model=est_model_post,
# #                 use_probability_of_observed_states=False,
# #             )
# #             data_decision[f"post_{choice}_{informed}_{job_offer}"] = np.nan_to_num(choice_probs_post, nan=0.0)
# #
# #
# #             choice_probs_start = choice_probs_for_choice_vals(
# #                 choice_vals=choice_vals,
# #                 states_dict=states_dict,
# #                 model=model,
# #                 # unobserved_state_specs=unobserved_state_specs,
# #                 params=params_pre,
# #                 est_model=est_model_start,
# #                 use_probability_of_observed_states=False,
# #             )
# #             data_decision[f"start_{choice}_{informed}_{job_offer}"] = np.nan_to_num(choice_probs_start, nan=0.0)
#
# # from dcegm.interface import value_for_state_choice_vec
# #
# # discrete_state_names = model["model_structure"]["discrete_states_names"]
# # # ['period', 'lagged_choice', 'education', 'sex', 'informed', 'policy_state', 'job_offer', 'partner_state', 'health']
# #
# # obs_id = 160
# # states_dict = {}
# # states_dict["period"] =  data_decision.loc[obs_id, "period"]
# # states_dict["lagged_choice"] =  data_decision.loc[obs_id, "lagged_choice"]
# # states_dict["education"] =  data_decision.loc[obs_id, "education"]
# # states_dict["sex"] =  0
# # states_dict["informed"] =  0
# # states_dict["policy_state"] =  data_decision.loc[obs_id, "policy_state"]
# # states_dict["job_offer"] =  0
# # states_dict["partner_state"] =  data_decision.loc[obs_id, "partner_state"]
# # states_dict["health"] =  data_decision.loc[obs_id, "health"]
# # states_dict["choice"] =  1
# #
# # wealth_grid = np.arange(8, 100, 2)
# # value_grid_post = np.empty_like(wealth_grid, dtype=float)
# # value_grid_pre = np.empty_like(wealth_grid, dtype=float)
# #
# # for i, wealth in enumerate(wealth_grid):
# #
# #     value_grid_post[i] = value_for_state_choice_vec(endog_grid_solved=est_model_post["endog_grid"], value_solved=est_model_post["value"], params=params_post, model=model, state_choice_vec=states_dict, wealth=wealth, second_continous=data_decision.loc[obs_id, "experience"])
# #     value_grid_pre[i] = value_for_state_choice_vec(endog_grid_solved=est_model_start["endog_grid"], value_solved=est_model_start["value"], params=params_pre, model=model, state_choice_vec=states_dict, wealth=wealth, second_continous=data_decision.loc[obs_id, "experience"])
# # import matplotlib.pyplot as plt
# # plt.plot(wealth_grid, value_grid_post)
# # plt.plot(wealth_grid, value_grid_pre)
# # plt.show()
