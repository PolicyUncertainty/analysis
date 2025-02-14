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
    est_data = create_partner_transition_sample(paths_dict, specs, load_data=load_data)

    # modify
    est_data = est_data[est_data["age"] <= specs["end_age"]]

    # Create a transition matrix for the partner state
    type_list = ["sex", "education"]
    cov_list = ["age", "partner_state_1.0", "partner_state_2.0"]
    # Create dummies for partner_state
    est_data_dummies = pd.get_dummies(est_data, columns=["partner_state"])
    est_data = pd.concat(
        [est_data, est_data_dummies[["partner_state_1.0", "partner_state_2.0"]]], axis=1
    )

    # Get dataframe of cartesian products of cov variables
    all_ages = np.arange(specs["start_age"], specs["end_age"] + 1)
    dummies = [False, True]
    full_index = pd.MultiIndex.from_product(
        [
            all_ages,
            dummies,
            dummies,
        ],
        names=["age", "partner_state_1.0", "partner_state_2.0"],
    )
    df_to_predict = pd.DataFrame(index=full_index).reset_index()
    # Filter out True, True
    mask = (df_to_predict["partner_state_1.0"] == True) & (
        df_to_predict["partner_state_2.0"] == True
    )
    df_to_predict = df_to_predict[~mask]

    full_index = pd.MultiIndex.from_product(
        [
            specs["sex_labels"],
            specs["education_labels"],
            all_ages,
            specs["partner_labels"],
            specs["partner_labels"],
        ],
        names=["sex", "education", "age", "partner_state", "lead_partner_state"],
    )
    full_df = pd.Series(index=full_index, data=np.nan, name="proportion")

    for sex, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            df_reduced = est_data[
                (est_data["sex"] == sex) & (est_data["education"] == edu_var)
            ]
            X = df_reduced[cov_list].astype(float)
            X = sm.add_constant(X)
            Y = df_reduced["lead_partner_state"]
            model = MNLogit(Y, X).fit()
            # Add prediction
            df_edu_predict = df_to_predict.copy()
            X_predict = sm.add_constant(df_edu_predict[cov_list].astype(float))
            df_edu_predict[[0, 1, 2]] = model.predict(X_predict)
            # Add to full_df
            df_edu_predict.set_index(
                ["age", "partner_state_1.0", "partner_state_2.0"], inplace=True
            )
            for current_partner_state, current_partner_label in enumerate(
                specs["partner_labels"]
            ):
                for lead_partner_state, lead_partner_label in enumerate(
                    specs["partner_labels"]
                ):
                    dummy_state_1 = current_partner_state == 1
                    dummy_state_2 = current_partner_state == 2
                    full_df.loc[
                        (
                            sex_label,
                            edu_label,
                            all_ages,
                            current_partner_label,
                            lead_partner_label,
                        )
                    ] = df_edu_predict.loc[
                        (all_ages, dummy_state_1, dummy_state_2), lead_partner_state
                    ].values

    if full_df.isna().any():
        raise ValueError("There should not be a None")

    out_file_path = paths_dict["est_results"] + "partner_transition_matrix.csv"
    full_df.to_csv(out_file_path)


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
