import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import partial


def calibrate_uninformed_hazard_rate(paths, options):
    """ This functions calibrates the hazard rate of becoming informed for the uninformed individuals with method of (weighted) moments.
    The hazard rate is assumed to be constant but can be changed to be a function of age."""
    out_file_path_rates = paths["intermediate_data"] + "uninformed_hazard_rate.pkl"
    out_file_path_belief = paths["intermediate_data"] + "uninformed_average_belief.pkl"

    df = open_dataset(paths)
    df = restrict_dataset(df, options)
    df = classify_informed(df, options)
    params = pd.DataFrame(columns = df["education"].unique())
    uninformed_beliefs = pd.DataFrame(columns = df["education"].unique())
    # calibrate hazard rate for each education group
    for edu in df["education"].unique():
        df_restricted = df[df["education"] == edu]
        avg_belief_uninformed = save_beliefs_of_uninformed(df_restricted, out_file_path_belief)
        moments, weights = generate_moments(df_restricted)
        initial_guess = [0.01]
        calibrated_params = fit_moments(moments, weights, initial_guess)
        params[edu] = calibrated_params
        uninformed_beliefs[edu] = [avg_belief_uninformed]
    # store and plot results
    params.to_pickle(out_file_path_rates)
    uninformed_beliefs.to_pickle(out_file_path_belief)
    plot_predicted_vs_actual(df, params)
    return calibrated_params, avg_belief_uninformed

def open_dataset(paths):
    soep_is = paths["soep_is"]
    relevant_cols = [
        "belief_pens_deduct",
        "age",
        "fweights",
        "education",
    ]
    df = pd.read_stata(soep_is)[relevant_cols].astype(float)

    # recode education
    df["education"] = df["education"].replace({1: 0, 2: 0, 3:1})
    return df

def restrict_dataset(df, options):
    df = df[df["belief_pens_deduct"] >= 0]
    df = df[df["age"] <= options["max_ret_age"]]
    return df

def classify_informed(df, options):
    informed_threshhold = options["informed_threshhold"]
    df["informed"] = df["belief_pens_deduct"] <= informed_threshhold
    return df

def save_beliefs_of_uninformed(df, out_file_path):
    """ This function saves the average ERP belief of the uninformed individuals in the dataset."""
    df_u = df[df["informed"] == 0]
    weighted_belief_uninformed = (df_u['belief_pens_deduct'] * df_u['fweights']).sum() / df_u['fweights'].sum()
    return weighted_belief_uninformed

def generate_moments(df):
    sum_fweights = df.groupby("age")["fweights"].sum()
    informed_sum_fweights = pd.Series(index = sum_fweights.index, data = 0, dtype=float)
    informed_sum_fweights.update(df[df["informed"] == 1].groupby("age")["fweights"].sum())
    informed_by_age = informed_sum_fweights / sum_fweights
    weights = sum_fweights / sum_fweights.sum()
    return informed_by_age, weights

def fit_moments(moments, weights, initial_guess):
    partial_obj = partial(objective_function, moments=moments, weights=weights)
    #breakpoint()
    result = minimize(fun=partial_obj, x0=initial_guess, tol=1e-16)
    params = result.x
    return pd.Series(params)

def objective_function(params, moments, weights):
    observed_ages = moments.index
    predicted_hazard_rate = hazard_rate(params, observed_ages)
    predicted_shares = predicted_shares_by_age(predicted_hazard_rate, observed_ages)
    return (((predicted_shares - moments)**2)*weights).sum()

def hazard_rate(params, observed_ages):
    # this can be changed to be a function of age
    max_age = int(observed_ages.max()) + 1
    ages = np.arange(max_age)
    #hazard_rate = params[0] + ages * params[1]
    hazard_rate = params[0] * np.ones(max_age)
    return hazard_rate

def predicted_shares_by_age(predicted_hazard_rate, observed_ages):
    # assumption: at age 0, no one is informed
    stay_uninformed_rate = 1 - predicted_hazard_rate
    max_age = int(observed_ages.max()) + 1
    uninformed_shares = np.ones(max_age)
    informed_shares = np.zeros(max_age)
    for age in range(1, max_age):
        uninformed_shares[age] = uninformed_shares[age - 1] * stay_uninformed_rate[age]
        informed_shares[age] = 1 - uninformed_shares[age]
    
    observed_ages = observed_ages.astype(int)
    relevant_shares = pd.Series(informed_shares).loc[observed_ages].values
    return relevant_shares

def plot_predicted_vs_actual(df, params):
    for edu in df["education"].unique():
        df_restricted = df[df["education"] == edu]
        moments, weights = generate_moments(df_restricted)
        predicted_informed_shares = predicted_shares_by_age(hazard_rate(params[edu], moments.index), moments.index)
        predicted_informed_shares = pd.Series(predicted_informed_shares, index=moments.index)
        plt.plot(moments, label="Actual_edu" + str(edu.astype(int)), marker="o", linestyle="None", markersize=4)
        plt.plot(predicted_informed_shares, label="Predicted_edu" + str(edu.astype(int)))
    plt.xlabel("Age")
    plt.ylabel("Share Informed")
    plt.legend()
    plt.show()