import pickle as pkl

import jax.numpy as jnp
import numpy as np
import yaml

from model_code.stochastic_processes.math_funcs import inv_logit_formula, logit_formula
from specs.belief_specs import add_belief_process_specs
from specs.experience_pp_specs import add_experience_and_pp_specs
from specs.family_specs import (
    predict_children_by_state,
    read_in_partner_transition_specs,
)
from specs.health_specs import process_health_labels, read_in_health_transition_specs
from specs.income_specs import add_income_specs


def generate_derived_and_data_derived_specs(path_dict, load_precomputed=False):
    """This function reads in specs and adds derived and data estimated specs."""
    specs = read_and_derive_specs(path_dict["specs"])
    # family transitions
    specs["children_by_state"], specs["max_children"] = predict_children_by_state(
        path_dict, specs
    )

    # Add belief process specs (both SRA and ERP parameters)
    specs = add_belief_process_specs(specs, path_dict)

    # Add income specs
    specs = add_income_specs(specs, path_dict)

    # Add experience specs
    specs = add_experience_and_pp_specs(specs, path_dict, load_precomputed)

    # Read in family transitions
    (
        specs["partner_trans_mat"],
        specs["n_partner_states"],
    ) = read_in_partner_transition_specs(path_dict, specs)

    # Read in health transition matrix
    specs["health_trans_mat"] = read_in_health_transition_specs(path_dict, specs)

    job_sep_probs = pkl.load(
        open(path_dict["first_step_results"] + "job_sep_probs.pkl", "rb")
    )
    specs["job_sep_probs"] = job_sep_probs
    specs["log_job_sep_probs"] = inv_logit_formula(job_sep_probs)
    return specs


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # Number of education types and choices from labels
    specs["n_education_types"] = len(specs["education_labels"])
    specs["education_grid"] = np.arange(specs["n_education_types"], dtype=int)
    specs["n_sexes"] = len(specs["sex_labels"])
    specs["sex_grid"] = np.arange(specs["n_sexes"], dtype=int)
    specs["n_choices"] = len(specs["choice_labels"])

    # Process information from health labels
    specs = process_health_labels(specs)

    # Partner states
    specs["n_partner_states"] = len(specs["partner_labels"])

    # you can retire from min retirement age until max retirement age
    specs["n_policy_states"] = (
        int(((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1) + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )
    specs["life_exp"] = np.array(specs["life_exp"])

    specs["income_tax_brackets"] = np.array(specs["income_tax_brackets"])
    specs["linear_income_tax_rates"] = np.array(specs["linear_income_tax_rates"])
    specs["quadratic_income_tax_rates"] = np.array(specs["quadratic_income_tax_rates"])
    specs["intercepts_income_tax"] = np.array(specs["intercepts_income_tax"])
    return specs
