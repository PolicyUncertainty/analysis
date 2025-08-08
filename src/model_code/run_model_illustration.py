# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
import numpy as np

from model_code.specify_model import specify_and_solve_model
from model_code.specify_simple_model import specify_and_solve_simple_model
from model_code.state_space.experience import scale_experience_years
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

# Set run specs
model_name = "ucl"
load_model = True
load_solution = True

params = pickle.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

model_solved = specify_and_solve_simple_model(
    path_dict=path_dict,
    file_append=model_name,
    params=params,
    load_model=load_model,
    load_solution=load_solution,
)
periods = np.arange(30, 40, dtype=int)
n_obs = len(periods)

exp = scale_experience_years(5, periods, specs["max_exp_diffs_per_period"])

# states = {
#     "period": periods,
#     "lagged_choice": np.ones_like(periods) * 3,
#     "education": np.zeros_like(periods),
#     "sex": np.zeros_like(periods),
#     "informed": np.ones_like(periods),
#     "policy_state": np.ones_like(periods) * 8,
#     "job_offer": np.ones_like(periods),
#     "partner_state": np.zeros_like(periods),
#     "health": np.zeros_like(periods),
#     "experience": exp
# }

states = {
    "period": periods,
    "lagged_choice": np.ones_like(periods) * 3,
    "education": np.zeros_like(periods),
    "sex": np.zeros_like(periods),
    "informed": np.ones_like(periods),
    "policy_state": np.ones_like(periods) * 8,
    "job_offer": np.ones_like(periods),
    "partner_state": np.zeros_like(periods),
    "health": np.zeros_like(periods),
    "experience": exp,
}

fig, ax = plt.subplots(figsize=(10, 6))
for asset in np.arange(1, 10):
    # for exp in np.arange(0.1, 1.0, 0.1):
    states["assets_begin_of_period"] = np.ones_like(periods, dtype=float) * asset

    choice_probs = model_solved.choice_probabilites_for_states(states=states)
    ax.plot(
        periods + 30,
        np.nan_to_num(choice_probs[:, 0], nan=0.0),
        label=f"Asset: {asset}",
    )
    ax.legend()

plt.show()
