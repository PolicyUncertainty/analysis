# %%
# Set paths of project
from set_paths import create_path_dict

path_dict = create_path_dict()

from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(path_dict)
# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)

import pickle
import numpy as np
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario

# %%
# Set specifications
n_agents = 10000
seeed = 123
params = pickle.load(open(path_dict["est_params"], "rb"))

# %%
###################################################################
# Uncertainty counterfactual
###################################################################


for alpha_val in np.arange(0, 0.11, 0.01):
    if np.allclose(alpha_val, 0.04):
        # Replace 0.04 with the subjective alpha from the data.
        # Sigma squared is always 0 in the simulation, as there is no uncertainty in
        # step function
        alpha_sim = np.loadtxt(path_dict["est_results"] + "exp_val_params.txt")
    else:
        alpha_sim = alpha_val

    # Create estimated model
    data_sim = solve_and_simulate_scenario(
        path_dict=path_dict,
        params=params,
        sim_alpha=alpha_sim,
        policy_exp_params=False,
        model_name="mu_fixed",
        df_exists=False,
        solution_exists=True,
        sol_model_exists=True,
        sim_model_exists=True,
    )
