# Description: This file estimates the parameters of the MONTHLY wage equation using the SOEP data.
# We estimate the following equation:
# wage = beta_0 + beta_1 * full_time_exp + beta_2 * full_time_exp^2 + individual_FE + time_FE + epsilon
import numpy as np
import pandas as pd
from linearmodels.panel.model import PanelOLS
from model_code.derive_specs import read_and_derive_specs


def estimate_wage_parameters(paths):
    specs = read_and_derive_specs(paths["specs"])
    # unpack path to SOEP core
    soep_c38 = paths["soep_c38"]

    # unpack options
    start_year = specs["start_year"]
    end_year = specs["end_year"]
    exp_cap = (specs["exp_cap"],)
    truncation_percentiles = [
        specs["wage_trunc_low_perc"],
        specs["wage_trunc_high_perc"],
    ]

    # get relevant data (sex, employment status, gross income, full time experience, education) from SOEP core
    pgen_df = pd.read_stata(
        f"{soep_c38}/pgen.dta",
        columns=["syear", "hid", "pid", "pgemplst", "pglabgro", "pgexpft", "pgpsbil"],
        convert_categoricals=False,
    )
    ppathl_df = pd.read_stata(
        f"{soep_c38}/ppathl.dta", columns=["pid", "hid", "syear", "sex", "gebjahr"]
    )
    merged_df = pd.merge(
        pgen_df,
        ppathl_df[["pid", "hid", "syear", "sex", "gebjahr"]],
        on=["pid", "hid", "syear"],
        how="inner",
    )

    # drop negative wages
    merged_df = merged_df[merged_df["pglabgro"] >= 0]

    # restrict sample
    merged_df = merged_df[merged_df["sex"] == "[1] maennlich"]  # only men
    merged_df = merged_df[
        (merged_df["syear"] >= start_year) & (merged_df["syear"] <= end_year)
    ]
    merged_df = merged_df[merged_df["pgemplst"] == 1]  # only full time
    merged_df = merged_df[
        (merged_df["pgexpft"] >= 0) & (merged_df["pgexpft"] <= exp_cap)
    ]
    pglabgro_percentiles = merged_df["pglabgro"].quantile(truncation_percentiles)
    merged_df = merged_df[
        (merged_df["pglabgro"] >= pglabgro_percentiles.iloc[0])
        & (merged_df["pglabgro"] <= pglabgro_percentiles.iloc[1])
    ]

    # Prepare estimation
    merged_df["year"] = merged_df["syear"].astype("category")
    merged_df = merged_df.set_index(["pid", "syear"])
    merged_df = merged_df.rename(
        columns={"pgexpft": "full_time_exp", "pglabgro": "wage"}
    )
    merged_df["full_time_exp_sq"] = merged_df["full_time_exp"] ** 2
    merged_df["constant"] = np.ones(len(merged_df))

    # save df to stata, keeping index as columns
    merged_df.reset_index().to_stata(paths["intermediate_data"] + "wage_data_old.dta", write_index=False)
    
    # estimate parametric regression, save parameters
    model = PanelOLS(
        dependent=merged_df["wage"] / specs["wealth_unit"],
        exog=merged_df[["constant", "full_time_exp", "full_time_exp_sq", "year"]],
        entity_effects=True,
        # time_effects=True,
    )
    fitted_model = model.fit(
        cov_type="clustered", cluster_entity=True, cluster_time=True
    )
    coefficients = fitted_model.params[0:3]

    # show model summary
    print(fitted_model)

    # model.fit().std_errors
    coefficients.loc["income_shock_std"] = np.sqrt(
        model.fit().resid_ss / (merged_df.shape[0] - 14763)
    )

    print("Estimated wage equation coefficients:\n{}".format(coefficients.to_string()))

    # Export regression coefficients
    coefficients.to_csv(paths["est_results"] + "wage_eq_params.csv")
    return coefficients
