import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import partial


def calibrate_uninformed_hazard_rate(paths, options, load_data=False):
    out_file_path = paths["intermediate_data"] + "uninformed_hazard_rate.pkl"

    if load_data:
        calibrated_params = pd.read_pickle(out_file_path)
        return calibrated_params

    df = open_dataset(paths)
    df = restrict_dataset(df, options)
    df = classify_informed(df, options)
    moments = generate_moments(df)
    # hazard rate = constant + age * slope
    initial_guess = [0.01]
    calibrated_params = fit_moments(moments, initial_guess)
    calibrated_params.to_pickle(out_file_path)
    plot_predicted_vs_actual(moments, calibrated_params)
    return calibrated_params

def open_dataset(paths):
    soep_is = paths["soep_is"]
    relevant_cols = [
        "belief_pens_deduct",
        "age",
        "fweights",
    ]
    df = pd.read_stata(soep_is)[relevant_cols].astype(float)
    return df

def restrict_dataset(df, options):
    df = df[df["belief_pens_deduct"] >= 0]
    df = df[df["age"] <= options["max_ret_age"]]
    return df

def classify_informed(df, options):
    informed_threshhold = options["informed_threshhold"]
    df["informed"] = df["belief_pens_deduct"] <= informed_threshhold
    return df

def generate_moments(df):
    informed_by_age = df.groupby("age")["informed"].mean()
    return informed_by_age


def fit_moments(moments, initial_guess):
    #breakpoint()
    partial_obj = partial(objective_function, moments=moments)
    result = minimize(fun=partial_obj, x0=initial_guess, tol=1e-16)
    params = result.x
    return pd.Series(params)

def objective_function(params, moments):
    observed_ages = moments.index
    predicted_hazard_rate = hazard_rate(params, observed_ages)
    predicted_shares = predicted_shares_by_age(predicted_hazard_rate, observed_ages)
    return (predicted_shares - moments).sum()**2 

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

def plot_predicted_vs_actual(moments, params):
    predicted_informed_shares = predicted_shares_by_age(hazard_rate(params, moments.index), moments.index)
    predicted_informed_shares = pd.Series(predicted_informed_shares, index=moments.index)
    plt.plot(moments, label="Actual")
    plt.plot(predicted_informed_shares, label="Predicted")
    plt.legend()
    plt.show()