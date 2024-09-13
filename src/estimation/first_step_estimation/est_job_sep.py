import numpy as np
import pandas as pd
import statsmodels.api as sm
from process_data.sample_creation_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)


def est_job_sep(paths_dict, specs, load_data=False):
    """This function estimates the job separation probability for each age and education
    level."""
    # Get estimation sample and create age squared
    df_job = create_job_sep_sample(paths_dict, specs, load_data)
    df_job["age_sq"] = df_job["age"] ** 2

    # Create solution containers
    job_sep_params = pd.DataFrame(
        index=df_job["education"].unique(), columns=["age", "age_sq", "const"]
    )
    n_edu_types = len(df_job["education"].unique())
    n_ages = len(df_job["age"].unique())
    job_sep_probs = np.zeros((n_edu_types, n_ages), dtype=float)

    # Loop over education levels
    for education in range(specs["n_education_types"]):
        # Filter data and estimate with OLS
        df_job_edu = df_job[df_job["education"] == education]
        model = sm.OLS(
            endog=df_job_edu["job_sep"],
            exog=sm.add_constant(df_job_edu[["age", "age_sq"]]),
        )
        results = model.fit()
        # Save params
        job_sep_params.loc[education] = results.params
        # Calculate job sep for each age
        ages = np.sort(df_job_edu["age"].unique())
        job_sep_probs_edu = (
            job_sep_params.loc[education, "const"]
            + job_sep_params.loc[education, "age"] * ages
            + job_sep_params.loc[education, "age_sq"] * ages**2
        )
        job_sep_probs[education, :] = job_sep_probs_edu

    # Save results
    job_sep_params.to_csv(paths_dict["est_results"] + "job_sep_params.csv")
    np.savetxt(
        paths_dict["est_results"] + "job_sep_probs.csv", job_sep_probs, delimiter=","
    )
    return job_sep_params, job_sep_probs
