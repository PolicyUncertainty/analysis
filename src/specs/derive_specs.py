import jax.numpy as jnp
import numpy as np
import yaml
from specs.experience_specs import create_max_experience
from specs.family_specs import predict_children_by_state
from specs.family_specs import read_in_partner_transition_specs
from specs.health_specs import read_in_health_transition_specs
from specs.income_specs import add_income_specs
from specs.informed_specs import add_informed_process_specs


def generate_derived_and_data_derived_specs(path_dict, load_precomputed=False):
    """This function reads in specs and adds derived and data estimated specs."""
    specs = read_and_derive_specs(path_dict["specs"])

    # Add income specs
    specs = add_income_specs(specs, path_dict)

    # family transitions
    specs["children_by_state"] = predict_children_by_state(path_dict, specs)

    # Read in family transitions
    (
        specs["partner_trans_mat"],
        specs["n_partner_states"],
    ) = read_in_partner_transition_specs(path_dict, specs)

    # Read in health transition matrix
    specs["health_trans_mat"] = read_in_health_transition_specs(path_dict, specs)

    # Set initial experience
    specs["max_init_experience"], specs["max_experience"] = create_max_experience(
        path_dict, specs, load_precomputed
    )

    specs["job_sep_probs"] = jnp.asarray(
        np.loadtxt(path_dict["est_results"] + "job_sep_probs.csv", delimiter=",")
    )

    # Add informed process specs
    specs = add_informed_process_specs(specs, path_dict)

    return specs


def read_and_derive_specs(spec_path):
    specs = yaml.safe_load(open(spec_path))

    # Number of periods in model
    specs["n_periods"] = specs["end_age"] - specs["start_age"] + 1
    # Number of education types and choices from labels
    specs["n_education_types"] = len(specs["education_labels"])
    specs["n_sexes"] = len(specs["sex_labels"])
    specs["n_choices"] = len(specs["choice_labels"])
    specs["n_health_states"] = len(specs["health_labels"])
    # you can retire from min retirement age until max retirement age
    specs["n_policy_states"] = (
        int(((specs["max_SRA"] - specs["min_SRA"]) / specs["SRA_grid_size"]) + 1) + 1
    )
    specs["SRA_values_policy_states"] = np.arange(
        specs["min_SRA"],
        specs["max_SRA"] + specs["SRA_grid_size"],
        specs["SRA_grid_size"],
    )
    return specs
