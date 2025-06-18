# %% Set paths of project
import pickle

import matplotlib.pyplot as plt

from estimation.msm.scripts.calc_moments import (
    calc_labor_supply_choice,
    calc_labor_transitions_by_age_bins,
    calc_median_wealth_by_age,
)
from estimation.msm.scripts.msm_estimation_setup import load_and_prep_data
from estimation.msm.scripts.plot_moment_fit import plot_wealth_moments
from set_paths import create_path_dict
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


model_name = "msm_first"
load_df = None
load_solution = None
load_sol_model = True


params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

iterations = {}

for bequest_scale in [2, 5, 10, 50]:
    params["bequest_scale"] = bequest_scale
    iterations[bequest_scale] = {}

    data_sim, model_solved = solve_and_simulate_scenario(
        announcement_age=None,
        path_dict=path_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        SRA_at_retirement=67,
        SRA_at_start=67,
        model_name=model_name,
        df_exists=load_df,
        solution_exists=load_solution,
        sol_model_exists=load_sol_model,
    )

    data_sim = data_sim.reset_index()

    iterations[bequest_scale]["choice_moment"] = calc_labor_supply_choice(data_sim)
    iterations[bequest_scale]["transition_moment"] = calc_labor_transitions_by_age_bins(
        data_sim
    )
    iterations[bequest_scale]["wealth_moment"] = calc_median_wealth_by_age(data_sim)

pickle.dump(
    iterations,
    open(path_dict["intermediate_data"] + f"msm_bequest_scale_iterations.pkl", "wb"),
)

# iterations = pickle.load(
#     open(path_dict["intermediate_data"] + f"msm_bequest_scale_iterations.pkl", "rb")
# )
#
# wealth_moments = []
# wealth_labels = []
# for bequest_scale, moments in iterations.items():
#     wealth_moments.append(moments["wealth_moment"])
#     wealth_labels.append(f"Bequest scale: {bequest_scale}")
#
# plot_wealth_moments(wealth_moments, wealth_labels, specs)
# plt.show()
