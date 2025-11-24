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
from estimation.struct_estimation.start_params_and_bounds.param_lists import (  # men_disutil_firing,
    men_disability_params,
    men_disutil_params,
    men_job_offer_params,
    men_taste,
    women_disability_params,
    women_disutil_params,
    women_job_offer_params,
    women_taste,
)
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from model_code.transform_data_from_model import create_informed_probability
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from specs.derive_specs import generate_derived_and_data_derived_specs

model_name = "sra_partner_fake_fixed_women"
params = pkl.load(
    open(paths_dict["struct_results"] + f"est_params_sra_partner_fixed.pkl", "rb")
)


params_to_estimate_names = (
    women_disutil_params
    + women_disability_params
    + women_job_offer_params
    # + women_taste
)
sex_type = "women"
edu_type = "all"
util_type = "add"
old_sample_only = False

LOAD_LAST_ESTIMATE = False
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False
LOAD_DF = None
LOAD_SOLUTION = None

print(f"Running fake estimation for params: {model_name}", flush=True)

# Load start params
start_params_all = load_and_set_start_params(paths_dict)

# Alter params by setting start values to lower bounds
lower_bounds_all = yaml.safe_load(
    open(paths_dict["start_params_and_bounds"] + "lower_bounds.yaml", "rb")
)
upper_bounds_all = yaml.safe_load(
    open(paths_dict["start_params_and_bounds"] + "upper_bounds.yaml", "rb")
)
for name in params_to_estimate_names:
    if "job_finding" in name:
        start_params_all[name] = (upper_bounds_all[name] + lower_bounds_all[name]) / 2
    else:
        start_params_all[name] = lower_bounds_all[name]

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
)

# Assume unobserved informed, health and job offers
data_fake = data_sim.copy()
data_fake.reset_index(inplace=True)


# Make informed, health and job offers unobserved
data_fake["informed"] = -99
data_fake["health"] = data_fake["health"].astype(int)
data_fake.loc[data_fake["health"].isin([1, 2]), "health"] = -99
data_fake["job_offer"] = -99

# Assign state values which we know for unobserved variables
ret_mask = data_fake["choice"] == 0
below_63_mask = data_fake["period"] < 33
data_fake.loc[ret_mask & below_63_mask, "health"] = 2
data_fake.loc[ret_mask & ~below_63_mask, "informed"] = 1

data_fake.loc[data_fake["choice"].isin([2, 3]), "job_offer"] = 1

# Add weighting var for estimation
data_fake.loc[data_fake["period"] == 0, "lagged_health"] = data_fake.loc[
    data_fake["period"] == 0, "health"
]
data_fake = create_informed_probability(df=data_fake, specs=specs)
# Run estimation
estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    start_params_all=start_params_all,
    load_model=LOAD_SOL_MODEL,
    use_weights=USE_WEIGHTS,
    last_estimate=None,
    save_results=SAVE_RESULTS,
    sex_type=sex_type,
    edu_type=edu_type,
    util_type=util_type,
    old_only=old_sample_only,
    print_men_examples=True,
    print_women_examples=True,
    slow_version=False,
    scale_opt=True,
    multistart=True,
    sim_data=data_fake,
)
print(estimation_results)

# # %% Set paths of project
# from specs.derive_specs import generate_derived_and_data_derived_specs

# path_dict = create_path_dict()
# specs = generate_derived_and_data_derived_specs(path_dict)


# create_fit_plots(
#     path_dict=paths_dict,
#     specs=specs,
#     model_name=model_name,
#     load_sol_model=True,
#     load_solution=None,
#     load_data_from_sol=False,
# )


# %%
