import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm

from set_styles import set_colors
JET_COLOR_MAP, LINE_STYLES = set_colors()
from process_data.first_step_sample_scripts.create_disability_pension_sample import (
    create_disability_pension_sample,
)


def est_disability_prob(paths, specs):

    logit_df = create_disability_pension_sample(paths, specs, load_data=True)

    logit_df["intercept"] = 1

    logit_vars = [
        "intercept",
        "education",
        "age",
        "age_above_55",
    ]

    logit_df["age_above_55"] = (logit_df["age"] >= 55) * (logit_df["age"] - 55)
    logit_df = logit_df[logit_df["health"] != 0]

    disability_prob_params = {}
    for sex_var, sex_append in enumerate(["men", "women"]):
        sex_mask = logit_df["sex"] == sex_var
        logit_df_gender = logit_df[sex_mask]
        logit_model = sm.Logit(
            logit_df_gender["retirement"], logit_df_gender[logit_vars]
        )
        logit_fitted = logit_model.fit()

        params = logit_fitted.params

        gender_params = {
            f"disability_logit_const_{sex_append}": params["intercept"],
            f"disability_logit_age_{sex_append}": params["age"],
            f"disability_logit_age_above_55_{sex_append}": params["age_above_55"],
            f"disability_logit_high_educ_{sex_append}": params["education"],
        }
        disability_prob_params = {**disability_prob_params, **gender_params}
        logit_df.loc[sex_mask, "predicted"] = logit_fitted.predict()

    # Plot prediction and data
    fig, axs = plt.subplots(ncols=2)
    for sex in range(2):
        ax = axs[sex]
        for edu in range(2):
            mask = (logit_df["education"] == edu) & (logit_df["sex"] == sex)

            df_edu_age_grouped = logit_df[mask].groupby("age")
            ax.plot(
                df_edu_age_grouped["retirement"].mean(),
                label=f"Data {edu}",
                color=JET_COLOR_MAP[edu],
                ls="--",
            )
            ax.plot(
                df_edu_age_grouped["predicted"].mean(),
                label=f"Predicted {edu}",
                color=JET_COLOR_MAP[edu],
            )
        ax.legend()
    plt.show()
    return disability_prob_params
