import pandas as pd
from matplotlib import pyplot as plt
from statsmodels import api as sm

from export_results.figures.color_map import JET_COLOR_MAP
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
    ]

    disability_prob_params = {}
    logit_model = sm.Logit(logit_df["retirement"], logit_df[logit_vars])
    logit_fitted = logit_model.fit()

    params = logit_fitted.params

    type_params = {
        f"disability_logit_const": params["intercept"],
        f"disability_logit_age": params["age"],
        f"disability_logit_high_educ": params["education"],
    }
    disability_prob_params = {**disability_prob_params, **type_params}
    # Plot prediction and data
    # fig, ax = plt.subplots()
    # logit_df["predicted"] = logit_fitted.predict()
    # for edu in range(2):
    #
    #     df_edu_age_grouped = logit_df[logit_df["education"] == edu].groupby("age")
    #     ax.plot(
    #         df_edu_age_grouped["retirement"].mean(),
    #         label=f"Data {edu}",
    #         color=JET_COLOR_MAP[edu],
    #         ls="--",
    #     )
    #     ax.plot(
    #         df_edu_age_grouped["predicted"].mean(),
    #         label=f"Predicted {edu}",
    #         color=JET_COLOR_MAP[edu],
    #
    #     )
    # ax.legend()
    # plt.show()
    return disability_prob_params
