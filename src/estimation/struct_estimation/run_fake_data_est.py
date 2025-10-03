# Set paths of project
import pickle as pkl

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
import pickle as pkl

import yaml

from estimation.struct_estimation.scripts.estimate_setup import (
    estimate_model,
    generate_print_func,
)
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from specs.derive_specs import generate_derived_and_data_derived_specs

params_to_estimate_names = [
    # "mu_men",
    # Men Full-time - 4 parameters
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    "disutil_unemployed_good_men",
    "disutil_unemployed_bad_men",
    "disutil_partner_retired_men",
    "SRA_firing_logit_intercept_men_low",
    "SRA_firing_logit_intercept_men_high",
    # Taste shock men - 1 parameter
    # "taste_shock_scale_men",
    # "bequest_scale",
    # # Men job finding - 3 parameters
    # "job_finding_logit_const_men",
    # "job_finding_logit_high_educ_men",
    # "job_finding_logit_good_health_men",
    # "job_finding_logit_age_men",
    # "job_finding_logit_age_above_55_men",
    # # Disability probability men - 3 parameters
    # "disability_logit_const_men",
    # "disability_logit_high_educ_men",
    # "disability_logit_age_men",
    # "disability_logit_age_above_55_men",
    # # "mu_women",
    # # Women Full-time - 4 parameters
    # "disutil_ft_work_good_women",
    # "disutil_ft_work_bad_women",
    # # # Women Part-time - 4 parameters
    # "disutil_pt_work_good_women",
    # "disutil_pt_work_bad_women",
    # # # Women Unemployment - 2 parameters
    # "disutil_unemployed_good_women",
    # "disutil_unemployed_bad_women",
    # # # Children - 2 parameters
    # "disutil_children_ft_work_high",
    # "disutil_children_ft_work_low",
    # # # Taste shock women - 1 parameter
    # # # "taste_shock_scale_women",
    # # # Women job finding - 3 parameters
    # # "job_finding_logit_const_women",
    # # "job_finding_logit_high_educ_women",
    # # "job_finding_logit_good_health_women",
    # # "job_finding_logit_age_women",
    # # "job_finding_logit_age_above_55_women",
    # # # Disability probability women - 3 parameters
    # "disability_logit_const_women",
    # "disability_logit_age_women",
    # "disability_logit_age_above_55_women",
    # "disability_logit_high_educ_women",
]

model_name = "start_lower_fake"

print(f"Running fake estimation for params: {model_name}", flush=True)

LOAD_SOL_MODEL = False
LOAD_SOLUTION = False
LOAD_DF = False
SAVE_RESULTS = False

# Load start params
start_params_all = load_and_set_start_params(paths_dict)
params = start_params_all.copy()
# params = pkl.load(
#     open(paths_dict["struct_results"] + f"est_params_women_3_it.pkl", "rb")
# )

specs = generate_derived_and_data_derived_specs(paths_dict)
print_function = generate_print_func(params_to_estimate_names, specs)

print("True parameters are:", flush=True)
print_function(params)

# Simulate baseline with subjective belief
data_sim, _ = solve_and_simulate_scenario(
    announcement_age=None,
    path_dict=paths_dict,
    params=params,
    subj_unc=True,
    custom_resolution_age=None,
    SRA_at_retirement=67,
    SRA_at_start=67,
    model_name=model_name,
    df_exists=LOAD_DF,
    solution_exists=LOAD_SOLUTION,
    sol_model_exists=LOAD_SOL_MODEL,
    men_only=True,
)
# Assume unobserved informed, health and job offers
data_fake = data_sim.copy()
data_fake.reset_index(inplace=True)
data_fake = data_fake[data_fake["sex"] == 0]

# data_fake["informed"] = -99
# data_fake.loc[data_fake["health"].isin([1, 2]), "health"] = -99
# data_fake.loc[(data_fake["choice"] == 0) & (data_fake["period"] < 33), "health"] = 2
# data_fake.loc[data_fake["period"] == 0, "lagged_health"] = data_fake.loc[
#     data_fake["period"] == 0, "health"
# ]
#
# data_fake["job_offer"] = -99
# data_fake.loc[data_fake["choice"].isin([2, 3]), "job_offer"] = 1

# Alter params by setting start values to lower bounds
lower_bounds_all = yaml.safe_load(
    open(paths_dict["start_params_and_bounds"] + "lower_bounds.yaml", "rb")
)
for name in params_to_estimate_names:
    params[name] = lower_bounds_all[name]

# Run estimation
estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    start_params_all=start_params_all,
    load_model=LOAD_SOL_MODEL,
    use_weights=False,
    last_estimate=None,
    save_results=SAVE_RESULTS,
    use_observed_data=False,
    sim_data=data_fake,
    print_women_examples=False,
    print_men_examples=False,
    men_only=True,
    scale_opt=True,
    multistart=True,
)
print(estimation_results)

# %% Set paths of project
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)


create_fit_plots(
    path_dict=paths_dict,
    specs=specs,
    model_name=model_name,
    load_sol_model=True,
    load_solution=None,
    load_data_from_sol=False,
)


# %%
