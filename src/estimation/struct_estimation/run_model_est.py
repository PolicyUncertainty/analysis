# Set paths of project
import pickle as pkl

from estimation.struct_estimation.start_params_and_bounds.param_lists import (
    men_disability_params,
    men_disutil_params,
    men_job_finding_params,
    men_SRA_firing,
    men_taste,
    wealth_params,
    women_disability_params,
    women_disutil_params,
    women_job_offer_params,
    women_SRA_firing,
    women_taste,
)
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)
from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)

params_to_estimate_names = (
    wealth_params
    + men_disutil_params
    + men_SRA_firing
    + men_taste
    + men_job_finding_params
    + men_disability_params
    + women_SRA_firing
    + women_taste
    + women_disutil_params
    + women_job_offer_params
    + women_disability_params
)


model_name = "men_3"

print(f"Running estimation for model: {model_name}", flush=True)

LOAD_LAST_ESTIMATE = False
LOAD_SOL_MODEL = True
SAVE_RESULTS = True
USE_WEIGHTS = False

if LOAD_LAST_ESTIMATE:
    last_estimate = pkl.load(
        open(paths_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )
else:
    last_estimate = None

# Load start params
start_params_all = load_and_set_start_params(paths_dict)

# Run estimation
estimation_results = estimate_model(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=model_name,
    start_params_all=start_params_all,
    load_model=LOAD_SOL_MODEL,
    use_weights=USE_WEIGHTS,
    last_estimate=last_estimate,
    save_results=SAVE_RESULTS,
)
print(estimation_results)

# %% Set paths of project
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(paths_dict)


create_fit_plots(
    path_dict=paths_dict,
    specs=specs,
    model_name=model_name,
    load_sol_model=True,
    load_solution=None,
    load_data_from_sol=False,
)


# %%
