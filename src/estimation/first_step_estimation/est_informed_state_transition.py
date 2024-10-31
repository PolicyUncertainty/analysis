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
    initial_guess = [0.1, 0.01]
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
    return pd.Series(params, index=["constant", "slope"])

def objective_function(params, moments):
    observed_ages = moments.index
    predicted_hazard_rate = hazard_rate(params, observed_ages)
    predicted_shares = predicted_shares_by_age(predicted_hazard_rate, observed_ages)
    return (predicted_shares - moments).sum()**2 

def hazard_rate(params, observed_ages):
    hazard_rate = params[0] + observed_ages * params[1]
    return hazard_rate

def predicted_shares_by_age(predicted_hazard_rate, observed_ages):
    # assumption: at age 0, no one is informed
    shares = np.zeros(len(observed_ages))
    breakpoint()
    for i, age in enumerate(observed_ages):
        age = int(age)
        shares[i] = predicted_hazard_rate[age]**(age)
    return 

def plot_predicted_vs_actual(moments, params):
    predicted_hazard_rate = hazard_rate(params, moments.index)
    plt.plot(moments, label="Actual")
    plt.plot(predicted_hazard_rate, label="Predicted")
    plt.legend()
    plt.show()