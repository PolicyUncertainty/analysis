# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# THIS IS THE LEGACY VERSION - DELETE SOON!
# NEW HOME: src/first_step_estimation/estimation/job_sep_estimation.py
# NEW PLOTTING: src/first_step_estimation/plots/job_separation_plots.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import pickle as pkl

import numpy as np
import pandas as pd
import statsmodels.api as sm

from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)


def est_job_sep(paths_dict, specs, load_data=False):
    """This function estimates the job separation probability for each age and education
    level."""
    # Get estimation sample and create age squared
    df_job = create_job_sep_sample(paths_dict, specs, load_data)

    # Estimate job separation probabilities
    job_sep_probs, job_sep_params = est_job_for_sample(df_job, specs)
    # Save results
    job_sep_params.to_csv(paths_dict["est_results"] + "job_sep_params.csv")
    pkl.dump(job_sep_probs, open(paths_dict["est_results"] + "job_sep_probs.pkl", "wb"))


def est_job_for_sample(df_job, specs):

    # Estimate job separation probabilities until max retirement age.

    df_job = df_job[df_job["age"] <= specs["max_est_age_labor"]].copy()
    df_job["good_health"] = df_job["lagged_health"] == specs["good_health_var"]
    df_job["above_50"] = df_job["age"] >= 50
    df_job["above_55"] = df_job["age"] >= 55
    df_job["above_60"] = df_job["age"] >= 60
    df_job["high_educ"] = df_job["education"] == 1
    df_job = sm.add_constant(df_job)

    logit_cols = [
        "const",
        "high_educ",
        "good_health",
        "above_50",
        "above_55",
        "above_60",
    ]

    # Create solution containers
    job_sep_params = pd.DataFrame(index=specs["sex_labels"], columns=logit_cols)

    # Loop over sexes
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        sub_mask = df_job["sex"] == sex_var
        # Filter data and estimate with OLS
        df_job_subset = df_job[sub_mask]
        # Estimate a logit model with age and age squared
        exog = df_job_subset[logit_cols].astype(float)
        model = sm.Logit(endog=df_job_subset["job_sep"].astype(float), exog=exog)
        results = model.fit()
        # Save params
        job_sep_params.loc[sex_label] = results.params
        # Calculate job sep for each age
        df_job.loc[sub_mask, "predicted_probs"] = results.predict(exog)

    # We will generate an array starting with age 0 to be able to use age as index

    all_ages = np.arange(0, specs["max_ret_age"])

    job_sep_probs = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], 2, len(all_ages)), dtype=float
    )
    predicted_ages = np.arange(specs["start_age"], specs["max_est_age_labor"] + 1)
    above_50 = predicted_ages >= 50
    above_55 = predicted_ages >= 55
    above_60 = predicted_ages >= 60

    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        params = job_sep_params.loc[sex_label]
        for edu_var in range(specs["n_education_types"]):
            for good_health in [0, 1]:
                exp_factor = (
                    params.loc["const"]
                    + params.loc["high_educ"] * edu_var
                    + params.loc["good_health"] * good_health
                    + params.loc["above_50"] * above_50
                    + params.loc["above_55"] * above_55
                    + params.loc["above_60"] * above_60
                )
                job_sep_probs_group = 1 / (1 + np.exp(-exp_factor))
                job_sep_probs[sex_var, edu_var, good_health, predicted_ages] = (
                    job_sep_probs_group
                )
                job_sep_probs[
                    sex_var, edu_var, good_health, specs["max_est_age_labor"] + 1 :
                ] = job_sep_probs_group[-1]

    return job_sep_probs, job_sep_params
