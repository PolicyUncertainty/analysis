# Description: This file estimates the parameters of the partner transition matrix using the SOEP panel data.
# For each sex, education level and age bin (10 years), we estimate P(partner_state | lagged_partner_state) non-parametrically.

import numpy as np
import pandas as pd
from model_code.derive_specs import read_and_derive_specs


def estimate_partner_transitions(paths_dict, specs):
    """Estimate the partner state transition matrix."""
    transition_data = prepare_transition_data(paths_dict, specs)

    sexes = transition_data["sex"].unique()
    edu_levels = transition_data["education"].unique()
    age_bins = transition_data["age_bin"].unique()
    states = transition_data["partner_state"].unique()

    cov_list = ["sex", "education", "age_bin", "lagged_partner_state"]
    trans_mat_df = transition_data.groupby(cov_list)["partner_state"].value_counts(normalize=True)
    
    out_file_path = paths_dict["est_results"] + "partner_transition_matrix.csv"
    trans_mat_df.to_csv(out_file_path)

    return trans_mat_df


def prepare_transition_data(paths_dict, specs):
    """Prepare the data for the transition estimation."""
    # load
    end_age = specs["end_age"]
    transition_data = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )
    # modify
    transition_data = transition_data[transition_data["age"] <= end_age]
    transition_data["age_bin"] = np.floor(transition_data["age"] / 10) * 10
    return transition_data


def calculate_nb_children(path_dict, specs):
    """Calculate the number of children in the household for each individual conditional on sex, education and age bin."""
    specs = read_and_derive_specs(path_dict["specs"])
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    # load data, filter, create age bins
    df = pd.read_pickle(path_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl")
    df = df[df["age"] >= start_age]
    df = df[df["age"] <= end_age]
    df["age_bin"] = np.floor(df["age"] / 5) * 5
    df["period"] = df["age"] - start_age

    # calculate average hours worked by partner by age, sex and education
    cov_list = ["sex", "education", "age"]        
    nb_children = df.groupby(cov_list)["children"].mean()

    # save to csv
    out_file_path = path_dict["est_results"] + f"nb_children.csv"
    nb_children.to_csv(out_file_path)
    breakpoint()
    return nb_children