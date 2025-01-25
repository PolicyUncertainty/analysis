import pandas as pd
import yaml
from specs.derive_specs import generate_derived_and_data_derived_specs
from statsmodels import api as sm


def load_and_set_start_params(path_dict):
    start_params_all = yaml.safe_load(
        open(path_dict["start_params_and_bounds"] + "start_params.yaml", "rb")
    )

    # Create start values for job offer probabilities
    struct_est_sample = pd.read_pickle(path_dict["struct_est_sample"])
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    job_offer_params = est_job_offer_params_full_obs(struct_est_sample, specs)

    # Update start values
    start_params_all.update(job_offer_params)
    return start_params_all


def est_job_offer_params_full_obs(df, specs, sex_append=["men", "women"]):
    # Filter for unemployed, because we only estimate job offer probs on them
    df_unemployed = df[df["lagged_choice"] == 1].copy()
    # Create work start indicator
    df_unemployed.loc[:, "work_start"] = (
        df_unemployed["choice"].isin([2, 3]).astype(int)
    )

    # Filter for relevant columns
    logit_df = df_unemployed[["sex", "period", "education", "work_start"]].copy()
    logit_df["age"] = logit_df["period"] + specs["start_age"]

    # logit_df["above_49"] = 0
    # logit_df.loc[logit_df["age"] > 49, "above_49"] = 1

    logit_df = logit_df[logit_df["age"] < 65]
    logit_df["intercept"] = 1

    logit_vars = [
        "intercept",
        "age",
        "education",
    ]

    job_offer_params = {}
    for sex_var, sex_append in enumerate(sex_append):
        logit_df_gender = logit_df[logit_df["sex"] == sex_var]
        logit_model = sm.Logit(
            logit_df_gender["work_start"], logit_df_gender[logit_vars]
        )
        logit_fitted = logit_model.fit()

        params = logit_fitted.params

        gender_params = {
            f"job_finding_logit_const_{sex_append}": params["intercept"],
            f"job_finding_logit_age_{sex_append}": params["age"],
            f"job_finding_logit_high_educ_{sex_append}": params["education"],
        }
        job_offer_params = {**job_offer_params, **gender_params}

    return job_offer_params
