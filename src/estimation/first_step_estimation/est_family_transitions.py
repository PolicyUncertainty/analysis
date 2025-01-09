# Description: This file estimates the parameters of the partner transition matrix using the SOEP panel data.
# For each sex, education level and age bin (10 years), we estimate P(partner_state | lagged_partner_state) non-parametrically.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)
from specs.derive_specs import read_and_derive_specs
from statsmodels.discrete.discrete_model import MNLogit


def estimate_partner_transitions(paths_dict, specs, load_data):
    """Estimate the partner state transition matrix."""
    df_trans = create_partner_transition_sample(paths_dict, specs, load_data=load_data)
    df_trans = df_trans[df_trans["age"] <= 90]
    df_trans["age_sq"] = df_trans["age"] ** 2

    ages = np.arange(specs["start_age"], specs["end_age"] + 1)

    result_index = pd.MultiIndex.from_product(
        [
            specs["sex_labels"],
            specs["education_labels"],
            ages,
            specs["partner_labels"],
            specs["partner_labels"],
        ],
        names=["sex", "education", "age", "partner_state", "lead_partner_state"],
    )

    result_df = pd.DataFrame(index=result_index, columns=["probability"], data=0.0)

    split_age = 45
    for model_num in range(2):
        if model_num == 0:
            df_model = df_trans[df_trans["age"] < split_age]
            df_model = df_model[df_model["lead_partner_state"] < 2]
            df_model = df_model[df_model["partner_state"] < 2]
            start_age = specs["start_age"]
            end_age = split_age - 1
            possible_partner_labels = specs["partner_labels"][:2]
        else:
            df_model = df_trans[df_trans["age"] >= split_age]
            start_age = split_age
            end_age = specs["end_age"]
            possible_partner_labels = specs["partner_labels"]

        for partner_state, partner_state_label in enumerate(possible_partner_labels):
            for sex_var, sex_label in enumerate(specs["sex_labels"]):
                for edu_var, edu_label in enumerate(specs["education_labels"]):
                    mask = (
                        (df_model["sex"] == sex_var)
                        & (df_model["education"] == edu_var)
                        & (df_model["partner_state"] == partner_state)
                    )
                    df_restr = df_model[mask]
                    model = MNLogit(
                        endog=df_restr["lead_partner_state"],
                        exog=sm.add_constant(df_restr[["age"]]),
                    )
                    result = model.fit()

                    ages = np.arange(start_age, end_age + 1)
                    predicted_probs = result.predict(exog=sm.add_constant(ages))

                    for age_idx, age in enumerate(ages):
                        for lead_partner_state, lead_partner_state_label in enumerate(
                            possible_partner_labels
                        ):
                            result_df.loc[
                                (
                                    sex_label,
                                    edu_label,
                                    age,
                                    partner_state_label,
                                    lead_partner_state_label,
                                ),
                                "probability",
                            ] = predicted_probs[age_idx, lead_partner_state]

    out_file_path = paths_dict["est_results"] + "partner_transition_matrix.csv"
    result_df.to_csv(out_file_path)

    return result_df


def prepare_transition_data(paths_dict, specs):
    """Prepare the data for the transition estimation."""
    # load
    end_age = specs["end_age"]

    # # modify
    # transition_data = transition_data[transition_data["age"] <= end_age]
    # transition_data["age_bin"] = np.floor(transition_data["age"] / 10) * 10
    return transition_data


def estimate_nb_children(paths_dict, specs):
    """Estimate the number of children in the household for each individual conditional
    on sex, education and age bin."""
    # load data, filter, create period and has_partner state
    df = pd.read_pickle(
        paths_dict["intermediate_data"] + "partner_transition_estimation_sample.pkl"
    )

    start_age = specs["start_age"]
    end_age = specs["end_age"]

    df = df[df["age"] >= start_age]

    # Filter out individuals below 60 for better estimation(we should set this in specs)
    df = df[df["age"] <= 60]
    df["period"] = df["age"] - start_age
    df["period_sq"] = df["period"] ** 2
    df["has_partner"] = (df["partner_state"] > 0).astype(int)
    # estimate OLS for each combination of sex, education and has_partner

    edu_states = list(range(specs["n_education_types"]))
    sexes = [0, 1]
    partner_states = [0, 1]

    sub_group_names = ["sex", "education", "has_partner"]

    multiindex = pd.MultiIndex.from_product(
        [sexes, edu_states, partner_states],
        names=sub_group_names,
    )

    columns = ["const", "period", "period_sq"]
    estimates = pd.DataFrame(index=multiindex, columns=columns)
    for sex in sexes:
        for education in edu_states:
            for has_partner in partner_states:
                df_reduced = df[
                    (df["sex"] == sex)
                    & (df["education"] == education)
                    & (df["has_partner"] == has_partner)
                ]
                X = df_reduced[columns[1:]]
                X = sm.add_constant(X)
                Y = df_reduced["children"]
                model = sm.OLS(Y, X).fit()
                estimates.loc[(sex, education, has_partner), columns] = model.params

    out_file_path = paths_dict["est_results"] + "nb_children_estimates.csv"
    estimates.to_csv(out_file_path)
    # plot results
    return estimates
