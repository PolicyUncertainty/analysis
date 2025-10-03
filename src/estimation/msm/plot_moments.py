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


model_name = specs["model_name"]
load_df = None
load_solution = None
load_sol_model = True

# plot_empirical_only = input("Plot empirical moments only? (y/n): ") == "y"

plot_empirical_only = False

data_decision = load_and_prep_data(path_dict)
data_decision["age"] = data_decision["period"] + specs["start_age"]

if plot_empirical_only:
    df_list = [data_decision]
    label_list = ["empirical"]
else:

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

    df_list = [data_decision, data_sim]
    label_list = ["empirical", "simulated"]


plot_moments_all_moments_for_dfs(df_list, label_list, specs)
# Get figure numbers and save
for i, fig in enumerate(plt.get_fignums()):
    fig = plt.figure(fig)
    fig.savefig(
        path_dict["plots"] + f"msm_moments_{model_name}_{i}.png",
        bbox_inches="tight",
    )
    plt.close(fig)
