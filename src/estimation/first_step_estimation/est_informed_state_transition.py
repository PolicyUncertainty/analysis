from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from export_results.figures.color_map import JET_COLOR_MAP


def calibrate_uninformed_hazard_rate(paths, specs):
    """This functions calibrates the hazard rate of becoming informed for the uninformed
    individuals with method of (weighted) moments.

    The hazard rate is assumed to be constant but can be changed to be a function of
    age.

    """

    out_file_path_rates = paths["est_results"] + "uninformed_hazard_rate.csv"
    out_file_shares = paths["est_results"] + "predicted_shares.csv"
    out_file_path_belief = paths["est_results"] + "uninformed_average_belief.csv"

    df = open_and_filter_dataset(paths, specs)

    # Classify informed individuals
    df["informed"] = df["belief_pens_deduct"] <= specs["informed_threshhold"]
    params = pd.DataFrame(columns=specs["education_labels"])
    uninformed_beliefs = pd.DataFrame(
        index=["erp_uninformed_belief", "erp_uninformed_belief_sem"],
        columns=specs["education_labels"],
    )
    predicted_shares = pd.DataFrame(columns=specs["education_labels"])
    edu_moments = pd.DataFrame(columns=specs["education_labels"])

    # calibrate hazard rate for each education group
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        df_restricted = df[df["education"] == edu_val]

        # First estimate the average erp belief of the uninformed individuals
        avg_belief_uninformed, sem_uninformed = beliefs_of_uninformed(df_restricted)
        uninformed_beliefs.loc["erp_uninformed_belief", edu_label] = (
            avg_belief_uninformed
        )
        uninformed_beliefs.loc["erp_uninformed_belief_sem", edu_label] = sem_uninformed

        # Then estimate the initial share and hazard rate
        observed_informed_shares, weights = generate_observed_informed_shares(
            df_restricted
        )
        params[edu_label] = fit_moments(
            moments=observed_informed_shares, weights=weights
        )
        # Save moments for plotting
        edu_moments[edu_label] = observed_informed_shares

        # Predict shares
        initial_age = df_restricted["age"].min()
        ages_to_predict = np.arange(initial_age, specs["max_ret_age"] + 1)
        predicted_shares[edu_label] = predicted_shares_by_age(
            params[edu_label].values, ages_to_predict
        )

    # store and plot results
    params.to_csv(out_file_path_rates)
    uninformed_beliefs.to_csv(out_file_path_belief)
    predicted_shares.to_csv(out_file_shares)
    plot_predicted_vs_actual(
        path_dict=paths,
        predicted_shares=predicted_shares,
        observed_shares=edu_moments,
        specs=specs,
    )


def open_and_filter_dataset(paths, specs):
    soep_is = paths["soep_is"]
    relevant_cols = [
        "belief_pens_deduct",
        "age",
        "fweights",
        "education",
    ]
    df = pd.read_stata(soep_is)[relevant_cols].astype(float)

    # recode education
    df["education"] = df["education"].replace({1: 0, 2: 0, 3: 1})

    # Age as int
    df["age"] = df["age"].astype(int)
    # Restrict dataset to relevant age range and filter invalid beliefs
    df = df[df["belief_pens_deduct"] >= 0]
    df = df[df["age"] <= specs["max_ret_age"]]

    return df


def beliefs_of_uninformed(df):
    """This function saves the average ERP belief of the uninformed individuals in the
    dataset."""
    df_u = df[df["informed"] == 0]
    normalized_fweights = (df_u["fweights"] / df_u["fweights"].sum()) * df_u.shape[0]
    weighted_beliefs = df_u["belief_pens_deduct"] * normalized_fweights
    weighted_belief_uninformed = weighted_beliefs.mean()
    sem = weighted_beliefs.sem()

    return weighted_belief_uninformed, sem


def generate_observed_informed_shares(df):
    sum_fweights = df.groupby("age")["fweights"].sum()
    informed_sum_fweights = pd.Series(index=sum_fweights.index, data=0, dtype=float)
    informed_sum_fweights.update(
        df[df["informed"] == 1].groupby("age")["fweights"].sum()
    )
    informed_by_age = informed_sum_fweights / sum_fweights
    weights = sum_fweights / sum_fweights.sum()
    return informed_by_age, weights


def fit_moments(moments, weights):
    params_guess = np.array([0.1, 0.01])
    partial_obj = partial(objective_function, moments=moments, weights=weights)
    result = minimize(fun=partial_obj, x0=params_guess, method="BFGS")
    params = pd.Series(index=["initial_informed_share", "hazard_rate"], data=result.x)
    return params


def objective_function(params, moments, weights):
    observed_ages = moments.index.values
    predicted_shares = predicted_shares_by_age(
        params=params, ages_to_predict=observed_ages
    )
    return (((predicted_shares - moments) ** 2) * weights).sum()


def predicted_shares_by_age(params, ages_to_predict):
    age_span = np.arange(ages_to_predict.min(), ages_to_predict.max() + 1)
    # The next line could be more complicated with age specific hazard rates
    # For now we use constant
    hazard_rate = params[1]
    predicted_hazard_rate = hazard_rate * np.ones_like(age_span, dtype=float)

    informed_shares = np.zeros_like(age_span, dtype=float)
    initial_informed_share = params[0]
    informed_shares[0] = initial_informed_share
    uninformed_shares = 1 - informed_shares

    for period in range(1, len(age_span)):
        uninformed_shares[period] = uninformed_shares[period - 1] * (
            1 - predicted_hazard_rate[period - 1]
        )
        informed_shares[period] = 1 - uninformed_shares[period]

    relevant_shares = pd.Series(index=age_span, data=informed_shares).loc[
        ages_to_predict
    ]
    return relevant_shares


def plot_predicted_vs_actual(path_dict, predicted_shares, observed_shares, specs):
    fig, ax = plt.subplots(figsize=(12, 8))
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        ax.plot(
            observed_shares[edu_label].rolling(window=3).mean(),
            label=f"Obs. {edu_label}",
            marker="o",
            linestyle="None",
            markersize=4,
            color=JET_COLOR_MAP[edu_val],
        )
        ax.plot(
            predicted_shares[edu_label],
            color=JET_COLOR_MAP[edu_val],
            label=f"Est. {edu_label}",
        )
    # Set labels
    ax.set_xlabel("Age")
    ax.set_ylabel("Share Informed")
    ax.legend()
    fig.savefig(path_dict["plots"] + "informed_shares.png")
    plt.show()
