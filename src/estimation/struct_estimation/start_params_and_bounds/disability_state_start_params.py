import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm

from model_code.stochastic_processes.health_transition import (
    calc_disability_probability,
)
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
        # "low_edu",
        "education",
        # "age",
        # "age_above_55",
        # "above_50",
        "age_above_50",
        # "above_60",
    ]

    logit_df["age_above_50"] = (logit_df["age"] >= 50) * (logit_df["age"] - 50)
    # logit_df["above_50"] = (logit_df["age"] >= 50).astype(float)
    # logit_df["above_55"] = (logit_df["age"] >= 55).astype(float)
    # logit_df["above_60"] = (logit_df["age"] >= 60).astype(float)

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
            # f"disability_logit_low_educ_{sex_append}": params["low_edu"],
            f"disability_logit_const_{sex_append}": params["intercept"],
            # f"disability_logit_age_{sex_append}": params["age"],
            f"disability_logit_age_above_50_{sex_append}": params["age_above_50"],
            # f"disability_logit_above_50_{sex_append}": params["above_50"],
            # f"disability_logit_above_55_{sex_append}": params["above_55"],
            # f"disability_logit_above_60_{sex_append}": params["above_60"],
            f"disability_logit_high_educ_{sex_append}": params["education"],
        }
        disability_prob_params = {**disability_prob_params, **gender_params}
        logit_df.loc[sex_mask, "predicted"] = logit_fitted.predict()

    # # # Plot prediction and data
    # fig, axs = plt.subplots(ncols=2)
    # for sex_var, sex_label in enumerate(specs["sex_labels"]):
    #     ax = axs[sex_var]
    #     for edu in range(2):
    #         mask = (logit_df["education"] == edu) & (logit_df["sex"] == sex_var)

    #         df_edu_age_grouped = logit_df[mask].groupby("age")

    #         ages = np.arange(30, 72)
    #         periods = ages - specs["start_age"]

    #         pred = calc_disability_probability(
    #             params=disability_prob_params,
    #             sex=jnp.array(sex_var),
    #             education=jnp.array(edu),
    #             period=periods,
    #             model_specs=specs,
    #         )
    #         pred *= np.ones_like(periods)

    #         ax.plot(
    #             df_edu_age_grouped["retirement"].mean(),
    #             label=f"Data {edu}",
    #             color=JET_COLOR_MAP[edu],
    #             ls="--",
    #         )
    #         ax.plot(
    #             ages,
    #             pred,
    #             label=f"Predicted {edu}",
    #             color=JET_COLOR_MAP[edu],
    #         )
    #         ax.set_title(f"{sex_label}")
    #         ax.set_xlabel("Age")
    #     ax.legend()

    # plt.savefig("disability_prob_fit.png")
    return disability_prob_params
