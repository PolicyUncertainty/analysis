import matplotlib.pyplot as plt
import numpy as np
from statsmodels import api as sm

from model_code.stochastic_processes.job_offers import (
    calc_job_finding_prob_men,
    calc_job_finding_prob_women,
)


def est_job_offer_params_full_obs(df, specs):
    df_unemployed = df[df["lagged_choice"] == 1].copy()
    df_unemployed.loc[:, "work_start"] = (
        df_unemployed["choice"].isin([2, 3]).astype(int)
    )

    logit_df = df_unemployed[
        ["sex", "period", "education", "health", "work_start"]
    ].copy()
    logit_df["age"] = logit_df["period"] + specs["start_age"]
    logit_df["good_health"] = (logit_df["health"] == 0).astype(float)
    logit_df["age_above_55"] = (logit_df["age"] >= 55) * (logit_df["age"] - 55)
    logit_df = logit_df[logit_df["age"] < 65]
    logit_df["intercept"] = 1

    logit_vars = ["intercept", "education", "good_health", "age", "age_above_55"]

    job_offer_params = {}

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    for i, (sex_var, sex_append, job_func) in enumerate(
        [
            (0, "men", calc_job_finding_prob_men),
            (1, "women", calc_job_finding_prob_women),
        ]
    ):

        # Subset by gender & education
        sub_df = logit_df[(logit_df["sex"] == sex_var)].copy()

        logit_model = sm.Logit(sub_df["work_start"], sub_df[logit_vars])
        logit_fitted = logit_model.fit(disp=False)

        sub_df["predicted_probs"] = logit_fitted.predict(sub_df[logit_vars])

        params = logit_fitted.params
        gender_params = {
            f"job_finding_logit_const_{sex_append}": params["intercept"],
            f"job_finding_logit_age_{sex_append}": params["age"],
            f"job_finding_logit_high_educ_{sex_append}": params["education"],
            f"job_finding_logit_good_health_{sex_append}": params["good_health"],
            f"job_finding_logit_age_above_55_{sex_append}": 0.0,
        }
        job_offer_params.update(gender_params)

        # for j, (educ_val, educ_label) in enumerate([(0, "low"), (1, "high")]):
        #     ax = axs[sex_var, educ_val]
        #
    #         # Plot predicted vs observed by health type
    #         for h_val, h_label, color in [
    #             (1, "Good health", "blue"),
    #             (0, "Bad health", "red"),
    #         ]:
    #             tmp = sub_df[sub_df["good_health"] == h_val]
    #             pred = tmp.groupby("age")["predicted_probs"].mean()
    #             ages = np.sort(sub_df.age.unique())
    #             pred = job_func(
    #                 params=gender_params,
    #                 education=educ_val,
    #                 good_health=h_val,
    #                 age=ages,
    #             )
    #             obs = tmp.groupby("age")["work_start"].mean()
    #             ax.plot(ages, pred, label=f"Pred {h_label}", color=color)
    #             ax.plot(obs, label=f"Obs {h_label}", color=color, linestyle="--")
    #
    #         ax.set_title(f"{sex_append}, {educ_label} edu")
    #         ax.legend(fontsize=8)
    #
    # fig.suptitle(
    #     "Job Offer Probabilities by Gender, Education, and Health", fontsize=14
    # )
    # plt.tight_layout()

    return job_offer_params
