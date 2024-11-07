import pandas as pd
import yaml
from specs.derive_specs import generate_derived_and_data_derived_specs
from statsmodels import api as sm


def load_and_set_start_params(path_dict):
    start_params_all = yaml.safe_load(open(path_dict["start_params"], "rb"))

    job_sep_params = create_job_offer_params_from_start(path_dict)
    start_params_all.update(job_sep_params)
    return start_params_all


def create_job_offer_params_from_start(path_dict):
    struct_est_sample = pd.read_pickle(path_dict["struct_est_sample"])

    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    # Filter for unemployed, because we only estimate job offer probs on them
    df_unemployed = struct_est_sample[struct_est_sample["lagged_choice"] == 1].copy()
    # Create work start indicator
    df_unemployed.loc[:, "work_start"] = (
        df_unemployed["choice"].isin([2, 3]).astype(int)
    )

    # Filter for relevant columns
    logit_df = df_unemployed[["period", "education", "work_start"]].copy()
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

    logit_model = sm.Logit(logit_df["work_start"], logit_df[logit_vars])
    logit_fitted = logit_model.fit()

    params = logit_fitted.params

    job_offer_params = {
        "job_finding_logit_const": params["intercept"],
        "job_finding_logit_age": params["age"],
        "job_finding_logit_high_educ": params["education"],
    }
    return job_offer_params
