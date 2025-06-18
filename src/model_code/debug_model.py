# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
import matplotlib.pyplot as plt
from dcegm.interfaces.inspect_solution import partially_solve

from model_code.specify_model import specify_model

model = specify_model(
    paths_dict,
    subj_unc=True,
    custom_resolution_age=None,
    sim_specs=None,
    load_model=True,
    debug_info="all",
)
# n_periods = 3

params = pkl.load(
    open(paths_dict["struct_results"] + f"est_params_msm_first.pkl", "rb")
)
# # Determine the last period we need to solve for.
# last_relevant_period = model.model_config["n_periods"] - n_periods
#
# relevant_state_choices_mask = (
#         model.model_structure["state_choice_space"][:, 0] >= last_relevant_period
# )
# relevant_state_choice_space = model.model_structure["state_choice_space"][
#     relevant_state_choices_mask
# ]
# state_choice_vec = relevant_state_choice_space[0, :]
# state = {
#     key: state_choice_vec[i] for i, key in enumerate(model.model_structure["discrete_states_names"])
# }
#
# df = model.get_child_states_and_calc_trans_probs(state, state_choice_vec[-1], params)
# breakpoint()

out_dict_0 = partially_solve(
    params=params,
    n_periods=3,
    return_candidates=True,
    income_shock_draws_unscaled=model.income_shock_draws_unscaled,
    income_shock_weights=model.income_shock_weights,
    model_config=model.model_config,
    batch_info=model.batch_info,
    model_funcs=model.model_funcs,
    model_structure=model.model_structure,
)

params["bequest_scale"] = 10

out_dict_1 = partially_solve(
    params=params,
    n_periods=3,
    return_candidates=True,
    income_shock_draws_unscaled=model.income_shock_draws_unscaled,
    income_shock_weights=model.income_shock_weights,
    model_config=model.model_config,
    batch_info=model.batch_info,
    model_funcs=model.model_funcs,
    model_structure=model.model_structure,
)
params["bequest_scale"] = 50

out_dict_2 = partially_solve(
    params=params,
    n_periods=3,
    return_candidates=True,
    income_shock_draws_unscaled=model.income_shock_draws_unscaled,
    income_shock_weights=model.income_shock_weights,
    model_config=model.model_config,
    batch_info=model.batch_info,
    model_funcs=model.model_funcs,
    model_structure=model.model_structure,
)


def plot_sols(sol_type, max_wealth_id, id_sc, candidate=False, scatter=False):
    append = "_candidates" if candidate else ""
    if scatter:
        plt.scatter(
            out_dict_0["endog_grid" + append][id_sc, 5, 1:max_wealth_id],
            out_dict_0[sol_type + append][id_sc, 5, 1:max_wealth_id],
            label="bequest scale 2.4",
        )
        plt.scatter(
            out_dict_1["endog_grid" + append][id_sc, 5, 1:max_wealth_id],
            out_dict_1[sol_type + append][id_sc, 5, 1:max_wealth_id],
            label="bequest scale 10",
        )
        plt.scatter(
            out_dict_2["endog_grid" + append][id_sc, 5, 1:max_wealth_id],
            out_dict_2[sol_type + append][id_sc, 5, 1:max_wealth_id],
            label="bequest scale 50",
        )

    else:
        plt.plot(
            out_dict_0["endog_grid" + append][id_sc, 5, 1:max_wealth_id],
            out_dict_0[sol_type + append][id_sc, 5, 1:max_wealth_id],
            label="bequest scale 2.4",
        )
        plt.plot(
            out_dict_1["endog_grid" + append][id_sc, 5, 1:max_wealth_id],
            out_dict_1[sol_type + append][id_sc, 5, 1:max_wealth_id],
            label="bequest scale 10",
        )
        plt.plot(
            out_dict_2["endog_grid" + append][id_sc, 5, 1:max_wealth_id],
            out_dict_2[sol_type + append][id_sc, 5, 1:max_wealth_id],
            label="bequest scale 50",
        )

    plt.legend()
    plt.show()


id_sc = 0
max_wealth_id = 30
sol_type = "policy"
append = ""
plot_sols("policy", 30, 0, candidate=False, scatter=False)
