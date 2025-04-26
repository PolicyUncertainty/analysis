from statsmodels import api as sm


def est_job_offer_params_full_obs(df, specs):
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
    for sex_var, sex_append in enumerate(["men", "women"]):
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
