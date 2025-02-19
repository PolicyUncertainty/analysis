# Description: This file estimates the parameters of the partner transition matrix using the SOEP panel data.
# For each sex, education level and age bin (10 years), we estimate P(partner_state | lagged_partner_state) non-parametrically.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from export_results.figures.color_map import JET_COLOR_MAP
from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)
from specs.derive_specs import read_and_derive_specs
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.discrete.discrete_model import MNLogit


def estimate_partner_transitions(paths_dict, specs, load_data):
    """Estimate the partner state transition matrix."""
    est_data = create_partner_transition_sample(paths_dict, specs, load_data=load_data)

    # modify
    est_data = est_data[est_data["age"] < specs["end_age"]]

    # Create covariance list
    cov_list = [
        "age_partner_state_0",
        "age_partner_state_1",
        "age_partner_state_2",
        "const_partner_state_0",
        "const_partner_state_1",
        "const_partner_state_2",
    ]

    # Create dummies for partner_state
    est_data_dummies = pd.get_dummies(est_data, columns=["partner_state"])
    # Interact age with partner_state
    est_data["age_partner_state_0"] = (
        est_data["age"] * est_data_dummies["partner_state_0.0"]
    )
    est_data["age_partner_state_1"] = (
        est_data["age"] * est_data_dummies["partner_state_1.0"]
    )
    est_data["age_partner_state_2"] = (
        est_data["age"] * est_data_dummies["partner_state_2.0"]
    )

    # Interact constant with partner_state
    est_data["const_partner_state_0"] = est_data_dummies["partner_state_0.0"].astype(
        float
    )
    est_data["const_partner_state_1"] = est_data_dummies["partner_state_1.0"].astype(
        float
    )
    est_data["const_partner_state_2"] = est_data_dummies["partner_state_2.0"].astype(
        float
    )

    # Get dataframe of cartesian products of cov variables
    all_ages = np.arange(specs["start_age"], specs["end_age"])

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

    fig, axs = plt.subplots(
        nrows=specs["n_partner_states"], ncols=specs["n_partner_states"]
    )
    col_count = 0
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            df_reduced = est_data[
                (est_data["sex"] == sex_var) & (est_data["education"] == edu_var)
            ].copy()
            X = df_reduced[cov_list].astype(float)
            Y = df_reduced["lead_partner_state"]
            model = MNLogit(Y, X).fit()

            df_reduced[[0, 1, 2]] = model.predict(X)
            for current_partner_state, current_partner_label in enumerate(
                specs["partner_labels"]
            ):
                df_state = df_reduced[
                    df_reduced["partner_state"] == current_partner_state
                ].copy()

                for lead_partner_state, lead_partner_label in enumerate(
                    specs["partner_labels"]
                ):
                    # Select lead partner state probs and forward fill missing values
                    trans_probs = (
                        df_state.groupby("age")[lead_partner_state]
                        .mean()
                        .reindex(all_ages, method="nearest")
                    )
                    full_df.loc[
                        (
                            sex_label,
                            edu_label,
                            all_ages,
                            current_partner_label,
                            lead_partner_label,
                        )
                    ] = trans_probs.values

                    axs[current_partner_state, lead_partner_state].plot(
                        all_ages,
                        trans_probs,
                        label=f"{sex_label}; {edu_label}",
                        color=JET_COLOR_MAP[col_count],
                    )
                    axs[current_partner_state, lead_partner_state].plot(
                        df_reduced.groupby(["age", "partner_state"])[
                            "lead_partner_state"
                        ]
                        .value_counts(normalize=True)
                        .loc[(slice(None), current_partner_state, lead_partner_state)],
                        color=JET_COLOR_MAP[col_count],
                        linestyle="--",
                    )
                    axs[current_partner_state, lead_partner_state].legend()
            col_count += 1

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
