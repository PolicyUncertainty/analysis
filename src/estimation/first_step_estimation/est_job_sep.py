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
    df_job["age_sq"] = df_job["age"] ** 2

    index = pd.MultiIndex.from_product(
        [specs["sex_labels"], specs["education_labels"]],
        names=["sex", "education"],
    )
    # Create solution containers
    job_sep_params = pd.DataFrame(index=index, columns=["age", "age_sq", "const"])

    # Estimate job separation probabilities until max retirement age
    n_ages = specs["max_ret_age"] - specs["start_age"] + 1
    job_sep_probs = np.zeros(
        (specs["n_sexes"], specs["n_education_types"], n_ages), dtype=float
    )

    # Loop over sexes
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        # Loop over education levels
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            # Filter data and estimate with OLS
            df_job_edu = df_job[
                (df_job["sex"] == sex_var) & (df_job["education"] == edu_var)
            ]
            model = sm.OLS(
                endog=df_job_edu["job_sep"],
                exog=sm.add_constant(df_job_edu[["age", "age_sq"]]),
            )
            results = model.fit()
            # Save params
            job_sep_params.loc[(sex_label, edu_label), :] = results.params
            # Calculate job sep for each age
            ages = np.sort(df_job_edu["age"].unique())
            job_sep_probs_group = (
                job_sep_params.loc[(sex_label, edu_label), "const"]
                + job_sep_params.loc[(sex_label, edu_label), "age"] * ages
                + job_sep_params.loc[(sex_label, edu_label), "age_sq"] * ages**2
            )
            job_sep_probs[sex_var, edu_var, :] = job_sep_probs_group

    return job_sep_probs, job_sep_params
