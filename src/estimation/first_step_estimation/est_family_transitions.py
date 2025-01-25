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
    est_data["age_bin"] = np.floor(est_data["age"] / 10) * 10

    # Create a transition matrix for the partner state
    type_list = ["sex", "education"]
    cov_list = ["age", "partner_state_1.0", "partner_state_2.0"]
    # Create dummies for partner_state
    est_data = pd.get_dummies(est_data, columns=["partner_state"])

    trans_mat_df = est_data.groupby(cov_list)["lead_partner_state"].value_counts(
        normalize=True
    )
    # Fo a multinominal logit model with lead_partner_state as dependent variable and cov list plus constant as
    # independent variables. # Condition the models of each type from type_list
    #
    # for sex in range(specs["n_sexes"]):
    #     for edu_var in range(specs["n_education_types"]):
    #         df_reduced = est_data[
    #             (est_data["sex"] == sex) & (est_data["education"] == edu_var)
    #         ]
    #         X = df_reduced[cov_list].astype(float)
    #         X = sm.add_constant(X)
    #         Y = df_reduced["lead_partner_state"]
    #         model = MNLogit(Y, X).fit()
    #         # Add prediction
    #         df_reduced[[0, 1, 2]] = model.predict(X).copy()
    #         breakpoint()

    full_index = pd.MultiIndex.from_product(
        [
            range(specs["n_sexes"]),
            range(specs["n_education_types"]),
            est_data["age_bin"].unique().tolist(),
            range(specs["n_partner_states"]),
            range(specs["n_partner_states"]),
        ],
        names=cov_list + ["lead_partner_state"],
    )
    full_df = pd.Series(index=full_index, data=0.0, name="proportion")
    full_df.update(trans_mat_df)

    out_file_path = paths_dict["est_results"] + "partner_transition_matrix.csv"
    full_df.to_csv(out_file_path)

    return trans_mat_df


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
