import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from process_data.first_step_sample_scripts.create_credited_periods_est_sample import create_credited_periods_est_sample
import matplotlib.pyplot as plt

# delete later
from set_paths import create_path_dict
paths_dict = create_path_dict()

def calibrate_credited_periods(paths, load_data=False, plot_results=False):
    df = create_credited_periods_est_sample(paths_dict, load_data=load_data)

    # create missing variables
    df["const"] = 1
    df["experience_men"] = df["experience"] * (1-df["sex"])
    df["experience_women"] = df["experience"] * df["sex"]
    edu_states = [0, 1]
    sexes = [0, 1]
  
    
    columns = [
        #"const",
        #"experience",
        #"has_partner",
        #"sex",
        "experience_men",
        "experience_women",
        ]
    
    # sub_group_names = ["sex", "education"]
    # multiindex = pd.MultiIndex.from_product(
    #     [sexes, edu_states],
    #     names=sub_group_names,
    # estimates = pd.DataFrame(index=multiindex, columns=columns)

    #for sex in sexes:
    #    for education in edu_states:
    #        df_reduced = df[
    #            (df["sex"] == sex)
    #            & (df["education"] == education)
    #        ]
    #        X = df_reduced[columns]
    #        Y = df_reduced["credited_periods"]
    #        model = sm.OLS(Y, X).fit()
    #        estimates.loc[(sex, education), columns] = model.params
    #        print(f'sex: {sex} \n education: {education}')
    #        print(model.summary())

    X = df[columns]
    Y = df["credited_periods"]
    model = sm.OLS(Y, X).fit()
    print(model.summary())

    if plot_results:
        plot_credited_periods_vs_exp(df, model, columns)

    # save estimates to csv
    estimates = pd.DataFrame(model.params, columns=['estimate'])
    out_file_path = paths["est_results"] + "credited_periods_estimates.csv"
    estimates.to_csv(out_file_path)
    return estimates 

def plot_credited_periods_vs_exp(df, model, columns):
    """ Plot credited periods (actual + predicted) vs experience """
    df["predicted_credited_periods"] = model.predict(df[columns])
    men_mask = df["sex"]==0
    plt.scatter(df[men_mask]["experience"], df[men_mask]["credited_periods"], label="actual_men")
    plt.scatter(df[~men_mask]["experience"], df[~men_mask]["credited_periods"], label="actual_women")
    plt.scatter(df[men_mask]["experience"], df[men_mask]["predicted_credited_periods"], label="predicted_men")
    plt.scatter(df[~men_mask]["experience"], df[~men_mask]["predicted_credited_periods"], label="predicted_women")
    plt.xlabel("experience")
    plt.ylabel("credited periods")
    plt.legend()
    plt.show()