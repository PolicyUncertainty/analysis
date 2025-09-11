# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
import copy

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from dcegm.interfaces.inspect_solution import partially_solve

from model_code.specify_model import specify_model
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(paths_dict)

np_health_mat = np.array(specs["health_trans_mat"])

survival_third_last_period = 1 - np_health_mat[:, :, 68, :-1, -1:]
np_health_mat[:, :, 68, :-1, -1] = 0
np_health_mat[:, :, 68, :-1, :-1] /= survival_third_last_period
specs["health_trans_mat"] = np_health_mat

model = specify_model(
    paths_dict,
    specs=specs,
    subj_unc=True,
    custom_resolution_age=None,
    sim_specs=None,
    load_model=True,
    debug_info="all",
)
n_periods = 3

params = pkl.load(
    open(paths_dict["struct_results"] + f"est_params_msm_first.pkl", "rb")
)


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


def plot_sols(
    sol_type, max_wealth_id, id_sc_0, id_sc_1, candidate=False, scatter=False
):
    append = "_candidates" if candidate else ""
    if scatter:
        plot_method = plt.scatter
    else:
        plot_method = plt.plot
    plot_method(
        out_dict_0["endog_grid" + append][id_sc_0, 5, 1:max_wealth_id],
        out_dict_0[sol_type + append][id_sc_0, 5, 1:max_wealth_id],
        label="bequest scale 2.4",
    )
    plot_method(
        out_dict_0["endog_grid" + append][id_sc_1, 5, 1:max_wealth_id],
        out_dict_0[sol_type + append][id_sc_1, 5, 1:max_wealth_id],
        label="bequest scale 10",
    )
    max_wealth = np.max(out_dict_0["endog_grid" + append][id_sc, 5, 1:max_wealth_id])
    plt.plot(
        np.arange(0, max_wealth),
        np.arange(0, max_wealth),
        label="45 degree",
    )

    plt.legend()
    plt.show()


# id_sc = 0
# max_wealth_id = 30
# sol_type = "policy"
# append = "_candidates"
plot_sols("policy", 30, 0, 25, candidate=False, scatter=True)

# params["taste_shock_scale"] = params["taste_shock_scale_men"]
# # Determine the last period we need to solve for.
# last_relevant_period = model.model_config["n_periods"] - n_periods
#
# relevant_state_choices_mask = (
#         model.model_structure["state_choice_space"][:, 0] >= last_relevant_period
# )
# relevant_state_choice_space = model.model_structure["state_choice_space"][
#     relevant_state_choices_mask
# ]
# state_choice_vec = relevant_state_choice_space[id_sc, :]
# state = {key: state_choice_vec[i] for i, key in enumerate(model.model_structure["discrete_states_names"])}
#
# rescale_idx = np.where(relevant_state_choices_mask)[0].min()
#
# df = model.get_full_child_states_by_asset_id_and_probs(state=state, choice=state_choice_vec[-1], params=params, asset_id=1, second_continuous_id=5)
