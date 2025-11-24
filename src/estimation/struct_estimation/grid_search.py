# Set paths of project
import pickle as pkl
import sys

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

paths_dict = create_path_dict(define_user=False)
import itertools

import numpy as np
import pandas as pd

from estimation.struct_estimation.scripts.estimate_setup import estimate_model
from estimation.struct_estimation.scripts.initialize_ll_function_only import (
    initialize_est_class,
)
from estimation.struct_estimation.scripts.observed_model_fit import create_fit_plots
from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
    load_and_set_start_params,
)
from model_code.specify_model import specify_model
from model_code.transform_data_from_model import load_scale_and_correct_data

LOAD_SOL_MODEL = True
SEX_TYPE = "women"

# Load start params
start_params_all = load_and_set_start_params(paths_dict)
specs = generate_derived_and_data_derived_specs(paths_dict)

# Define parameter groups (excluding health-specific parameters)
param_groups = {
    "ft_work": ["disutil_ft_work_good_women", "disutil_ft_work_bad_women"],
    "pt_work": ["disutil_pt_work_good_women", "disutil_pt_work_bad_women"],
    "unemployed": ["disutil_unemployed_good_women", "disutil_unemployed_bad_women"],
    "partner_retired": ["disutil_partner_retired_women"],
    "children": ["disutil_children_ft_work_low", "disutil_children_ft_work_high"],
}

# Define grid values
grid_values = [0.1, 0.5, 0.9]

# Test name for output files
test_name = "women_disutil_grid_search"

# Initialize estimation class
params_to_estimate_names = []
for group_params in param_groups.values():
    params_to_estimate_names.extend(group_params)

est_class = initialize_est_class(
    paths_dict,
    params_to_estimate_names=params_to_estimate_names,
    file_append=test_name,
    load_model=LOAD_SOL_MODEL,
    start_params_all=start_params_all,
    use_weights=False,
    save_results=True,
    print_men_examples=True,
    print_women_examples=True,
    use_observed_data=True,
    sim_data=None,
    old_only=False,
    sex_type=SEX_TYPE,
    edu_type="all",
    util_type="add",
)

# Load model and data
model = specify_model(
    path_dict=paths_dict,
    specs=specs,
    subj_unc=True,
    custom_resolution_age=None,
    load_model=LOAD_SOL_MODEL,
    sim_specs=None,
)

# Create all combinations for grid search
# Each group gets 3 values, so we have 3^5 = 243 combinations total
group_combinations = list(itertools.product(grid_values, repeat=len(param_groups)))

print(f"Running grid search with {len(group_combinations)} combinations...")
print(f"Parameter groups: {list(param_groups.keys())}")
print(f"Grid values: {grid_values}")

# Store results
results_list = []

for i, combination in enumerate(group_combinations):
    print(f"Running combination {i + 1}/{len(group_combinations)}: {combination}")

    # Create parameter dictionary for this combination
    param_dict = {}

    # Assign values to each group
    for group_idx, (group_name, group_params) in enumerate(param_groups.items()):
        group_value = combination[group_idx]

        # For health-specific parameters, keep the health differential
        if group_name in ["ft_work", "pt_work", "unemployed", "children"]:
            # Keep current health differential - good health gets the grid value,
            # bad health gets a slightly different value to maintain ranking
            param_dict[group_params[0]] = group_value  # good health
            param_dict[group_params[1]] = group_value  # bad health (slightly lower)
        else:
            # For non-health specific parameters, assign same value to all in group
            for param in group_params:
                param_dict[param] = group_value

    # Calculate likelihood for this parameter combination
    ll_value = np.sum(est_class.ll_func(param_dict))

    # Store results
    result_row = {
        "combination_id": i,
        "ft_work_value": combination[0],
        "pt_work_value": combination[1],
        "unemployed_value": combination[2],
        "partner_retired_value": combination[3],
        "children_value": combination[4],
        "log_likelihood": ll_value,
    }

    # Add individual parameter values for reference
    for param_name, param_value in param_dict.items():
        result_row[param_name] = param_value

    results_list.append(result_row)

    print(f"  Log-likelihood: {ll_value:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results_list)

# Save results
results_df.to_csv(f"{test_name}_results.csv", index=False)
print(f"\nGrid search complete! Results saved to {test_name}_results.csv")

# Print summary of best combinations
print("\nTop 10 parameter combinations by log-likelihood:")
top_results = results_df.nlargest(10, "log_likelihood")
print(
    top_results[
        [
            "combination_id",
            "ft_work_value",
            "pt_work_value",
            "unemployed_value",
            "partner_retired_value",
            "children_value",
            "log_likelihood",
        ]
    ].to_string(index=False)
)

# Save top results separately
top_results.to_csv(f"{test_name}_top_10.csv", index=False)
print(f"\nTop 10 results saved to {test_name}_top_10.csv")
