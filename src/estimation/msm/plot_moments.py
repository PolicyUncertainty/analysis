# %% Set paths of project
import pickle

import matplotlib.pyplot as plt

from estimation.msm.scripts.msm_estimation_setup import load_and_prep_data
from estimation.msm.scripts.plot_moment_fit import plot_moments_all_moments_for_dfs
from set_paths import create_path_dict
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


model_name = "msm_free"
load_df = True
load_solution = True
load_sol_model = True


params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

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

data_decision = load_and_prep_data(path_dict, params, model_solved)
data_decision["age"] = data_decision["period"] + specs["start_age"]

plot_moments_all_moments_for_dfs(data_decision, data_sim, specs)
plt.show()
