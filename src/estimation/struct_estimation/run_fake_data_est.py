# Set paths of project
import pickle as pkl

import numpy as np
import pandas as pd
import yaml

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.struct_estimation.scripts.estimate_setup import (
    estimate_model,
    generate_print_func,
)
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
from model_code.transform_data_from_model import create_informed_probability
from simulation.sim_tools.simulate_scenario import solve_and_simulate_scenario
from specs.derive_specs import generate_derived_and_data_derived_specs

seed = 123
model_name = "fake_summer_women"
# True params. They must be a complete params vector of the current model, as all
# params which are not estimated are kept at these values.
true_params_name = "summer_women"
params = pkl.load(
    open(paths_dict["struct_results"] + f"est_params_{true_params_name}.pkl", "rb")
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

LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False
LOAD_DF = None
LOAD_SOLUTION = None
# Relative scale of the perturbation of the start values around the true params.
# Params with a true value close to zero are perturbed on an absolute scale of
# START_PERTURBATION * 0.1.
START_PERTURBATION = 0.25
# Draw the same number of observations per sex, education and period as in the
# observed estimation sample.
MATCH_OBSERVED_SAMPLE = True

print(f"Running fake estimation for params: {model_name}", flush=True)

specs = generate_derived_and_data_derived_specs(paths_dict)
print_function = generate_print_func(params_to_estimate_names, specs)

print("True parameters are:", flush=True)
print_function(params)

# All params which are not estimated are kept at their true values. The estimated
# params start at a random perturbation of the true values within the bounds.
lower_bounds_all = yaml.safe_load(
    open(paths_dict["start_params_and_bounds"] + "lower_bounds.yaml", "rb")
)
upper_bounds_all = yaml.safe_load(
    open(paths_dict["start_params_and_bounds"] + "upper_bounds.yaml", "rb")
)
rng = np.random.default_rng(seed)
start_params_all = params.copy()
for name in params_to_estimate_names:
    true_value = float(params[name])
    scale = START_PERTURBATION * max(abs(true_value), 0.1)
    start_params_all[name] = float(
        np.clip(
            true_value + scale * rng.standard_normal(),
            lower_bounds_all[name],
            upper_bounds_all[name],
        )
    )
# estimate_model overwrites the start params it gets with the estimates.
start_values = {name: start_params_all[name] for name in params_to_estimate_names}

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
    seed=seed,
)

data_fake = data_sim.copy()
data_fake.reset_index(drop=True, inplace=True)
data_fake.sort_values(["agent", "period"], inplace=True)

good_health_var = specs["good_health_var"]
bad_health_var = specs["bad_health_var"]
disabled_health_var = specs["disabled_health_var"]
working_choices = [2, 3]

# Lagged health as in the observed data, where the survey distinguishes only good
# and bad health. The first period has no lag in the simulation, so we use the
# health of the period itself.
surveyed_health = data_fake["health"].replace(disabled_health_var, bad_health_var)
data_fake["lagged_health"] = surveyed_health.groupby(data_fake["agent"]).shift(1)
first_period = data_fake["lagged_health"].isna()
data_fake.loc[first_period, "lagged_health"] = surveyed_health[first_period]
data_fake["lagged_health"] = data_fake["lagged_health"].astype(int)

# Make informed, health and job offers unobserved following the rules of the
# observed data (see process_data/structural_sample_scripts and
# process_data/soep_vars/job_hire_and_fire.py).
age = data_fake["period"] + specs["start_age"]

# Informed: only known for retirees from 63 on.
data_fake["informed"] = -99
data_fake.loc[(data_fake["choice"] == 0) & (age >= 63), "informed"] = 1

# Health: bad and disabled are not distinguished, except for fresh retirees below
# the long insured retirement age, who must be disabled.
fresh_early_retired = (
    (data_fake["choice"] == 0)
    & (data_fake["lagged_choice"] != 0)
    & (age < specs["min_long_insured_age"])
)
n_not_disabled = (
    fresh_early_retired & (data_fake["health"] != disabled_health_var)
).sum()
if n_not_disabled > 0:
    raise ValueError(
        f"{n_not_disabled} fresh retirees below the long insured retirement age are "
        f"not disabled in the simulation, but the observed data assigns them "
        f"disability."
    )
data_fake["health"] = data_fake["health"].astype(int)
data_fake.loc[
    data_fake["health"].isin([bad_health_var, disabled_health_var]), "health"
] = -99
data_fake.loc[fresh_early_retired, "health"] = disabled_health_var

# Job offer: observed for everybody working and for everybody who worked last
# period and stopped (fired or not).
working = data_fake["choice"].isin(working_choices)
was_working = data_fake["lagged_choice"].isin(working_choices)
true_job_offer = data_fake["job_offer"].copy()
data_fake["job_offer"] = -99
data_fake.loc[working, "job_offer"] = 1
stopped_working = was_working & ~working
data_fake.loc[stopped_working, "job_offer"] = true_job_offer[stopped_working]

if MATCH_OBSERVED_SAMPLE:
    # Sample, for every sex, education and period, as many observations as in the
    # observed estimation sample. Already retired observations are dropped as in
    # the estimation.
    cell_vars = ["sex", "education", "period"]
    data_obs = pd.read_csv(paths_dict["struct_est_sample"])
    data_obs = data_obs[data_obs["lagged_choice"] != 0]
    n_obs_cells = data_obs.groupby(cell_vars).size()

    data_fake = data_fake[data_fake["lagged_choice"] != 0]
    sampled = []
    for cell, df_cell in data_fake.groupby(cell_vars):
        n_obs = n_obs_cells.get(cell, 0)
        if n_obs == 0:
            continue
        if n_obs > len(df_cell):
            print(
                f"Cell {cell} has {n_obs} observed but only {len(df_cell)} "
                f"simulated observations. Using all simulated ones.",
                flush=True,
            )
        sampled.append(df_cell.sample(n=min(n_obs, len(df_cell)), random_state=seed))
    data_fake = pd.concat(sampled)
    print(f"Fake estimation sample has {len(data_fake)} observations.", flush=True)

data_fake = create_informed_probability(df=data_fake, specs=specs)

# Run estimation
estimation_results, est_params_all = estimate_model(
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
    use_observed_data=False,
    sim_data=data_fake,
    reference_params=params,
)
print(estimation_results)

# Compare estimates with the true params
std_errors = pkl.load(
    open(paths_dict["struct_results"] + f"std_errors_{model_name}.pkl", "rb")
)
recovery = pd.DataFrame(
    {
        "true": {name: float(params[name]) for name in params_to_estimate_names},
        "start": start_values,
        "estimate": {
            name: float(est_params_all[name]) for name in params_to_estimate_names
        },
        "std_error": {
            name: float(std_errors[name]) for name in params_to_estimate_names
        },
    }
)
recovery["t_vs_true"] = (recovery["estimate"] - recovery["true"]) / recovery[
    "std_error"
]
recovery.to_csv(paths_dict["struct_results"] + f"recovery_{model_name}.csv")
print(recovery.to_string(), flush=True)
