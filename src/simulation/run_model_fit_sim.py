# %% Set paths of project
import pickle

import matplotlib.pyplot as plt

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


model_name = "disability_try"
load_df = None
load_solution = None
load_sim_model = True
load_sol_model = True


# params = pickle.load(
#     open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
# )

params = pickle.load(open(path_dict["struct_results"] + f"est_params_new.pkl", "rb"))
params["mu_men"] = params["mu"]
params["mu_women"] = params["mu"]
params["job_finding_logit_period_men"] = params["job_finding_logit_age_men"]
params["job_finding_logit_period_women"] = params["job_finding_logit_age_women"]
params["job_finding_logit_const_men"] -= params["job_finding_logit_age_men"] * 30
params["job_finding_logit_const_women"] -= params["job_finding_logit_age_women"] * 30

for s in ["men", "women"]:
    for edu in ["low", "high"]:
        for health in ["bad", "good"]:
            params[f"disutil_ft_work_{edu}_{health}_{s}"] = params[
                f"disutil_ft_work_{health}_{s}"
            ]

        params[f"disutil_unemployed_{edu}_{s}"] = params[f"disutil_unemployed_{s}"]


for edu in ["low", "high"]:
    for health in ["bad", "good"]:
        params[f"disutil_pt_work_{edu}_{health}_women"] = params[
            f"disutil_pt_work_{health}_women"
        ]


from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_start = load_and_set_start_params(path_dict)
params["disability_logit_const"] = params_start["disability_logit_const"]
params["disability_logit_period"] = params_start["disability_logit_period"]
params["disability_logit_high_educ"] = params_start["disability_logit_high_educ"]

# loop over params_start keys and check if they are all in params
for key in params_start.keys():
    if key not in params:
        print(f"{key} not in params")


which_plots = input(
    "Which plots do you want to show?\n \n"
    " - [a]ll\n"
    " - [c]hoices\n"
    " - [w]ealth\n"
    " - [i]ncome\n"
    " - [s]tates\n"
    " - [wc]hoices and wealth\n"
)

from simulation.figures.simulated_model_fit import (
    plot_choice_shares_single,
    plot_quantiles,
    plot_states,
)

if which_plots in ["a", "c", "wc"]:
    plot_choice_shares_single(
        path_dict=path_dict,
        specs=specs,
        params=params,
        file_name="sim_choices",
        model_name=model_name,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After running, we can set all to true
    load_df = True if load_df is not None else load_df
    load_solution = True if load_solution is not None else load_solution
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "w", "wc"]:
    plot_quantiles(
        path_dict=path_dict,
        specs=specs,
        params=params,
        model_name=model_name,
        quantiles=[0.5],
        sim_col_name="wealth_at_beginning",
        obs_col_name="adjusted_wealth",
        file_name="average_wealth",
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After running, we can set all to true
    load_df = True if load_df is not None else load_df
    load_solution = True if load_solution is not None else load_solution
    load_sim_model = True
    load_sol_model = True

if which_plots in ["a", "i"]:
    plot_quantiles(
        path_dict=path_dict,
        specs=specs,
        params=params,
        model_name=model_name,
        quantiles=[0.5],
        sim_col_name="total_income",
        obs_col_name="hh_net_income",
        file_name=None,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )
    # After running, we can set all to true
    load_df = True if load_df is not None else load_df
    load_solution = True if load_solution is not None else load_solution
    load_sim_model = True
    load_sol_model = True


if which_plots in ["a", "s"]:
    plot_states(
        path_dict,
        specs,
        params,
        model_name,
        load_df=load_df,
        load_solution=load_solution,
        load_sol_model=load_sol_model,
        load_sim_model=load_sim_model,
    )

plt.show()
