# This script estimates the parameters of the truncated normal distribution of subjective retirement age expectations
# based on the observed CDF values from the SOEP-IS dataset. The parameters are estimated for each respondent individually.
# Then, the expected values of the individual CDF are regressed on birth year to obtain the slope of the policy expectation process.
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import truncnorm

from process_data.structural_sample_scripts.policy_state import (
    create_SRA_by_gebjahr,
)


def estimate_truncated_normal(df, paths, options, load_data=False):
    out_file_path = paths["intermediate_data"] + "df_soep_is_truncated_normals.pkl"

    if load_data:
        df = pd.read_pickle(out_file_path)
        return df


    # unpack options, put into new dict
    function_spec = {
        "ll": options["lower_limit"],
        "ul": options["upper_limit"],
        "first_cdf_point": options["first_cdf_point"],
        "second_cdf_point": options["second_cdf_point"],
    }

    df["time_to_ret"] = df["exp_pens_uptake"] - df["age"]

    # estimate loc and scale of truncated normal (as well as mean and var), save
    df = estimate_truncated_normal_parameters(df, function_spec)
    df.to_pickle(out_file_path)
    return df


def estimate_truncated_normal_parameters(df, function_spec):
    # add columns for mu and sigma (parameters of the truncated normal distribution), expected value and variance of the distribution,
    # and squared differences between observed and predicted CDF values
    df["mu"] = np.nan
    df["sigma"] = np.nan
    df["ex_val"] = np.nan
    df["var"] = np.nan
    df["error_1"] = np.nan
    df["error_2"] = np.nan

    # loop over respondents and estimate parameters.
    # 1&2 special case: if expected ret age is 67 or 68 with 100% certainty, set expected value to 67 or 68, respectively, and variance to 0.
    # 3 special case: if expected ret age is 69 with 100% certainty, set expected value to mean of last interval and variance to variance of uniform distribution in last interval
    # 4 all other cases: use scipy.optimize.root to find the parameters that minimize the squared differences between observed and predicted CDF values
    for index, row in df.iterrows():
        if not np.isnan(row["pol_unc_stat_ret_age_67"]):
            if row["pol_unc_stat_ret_age_67"] == 100:
                row["pol_unc_stat_ret_age_67"] = 99.5
                row["pol_unc_stat_ret_age_68"] = 0.4
                row["pol_unc_stat_ret_age_69"] = 0.1
            elif row["pol_unc_stat_ret_age_68"] == 100:
                row["pol_unc_stat_ret_age_67"] = 0.25
                row["pol_unc_stat_ret_age_68"] = 99.5
                row["pol_unc_stat_ret_age_69"] = 0.25
            elif row["pol_unc_stat_ret_age_69"] == 100:
                # Assume that there was just enough mass in the last interval that the
                # respondent rounded up to 100. Assign the rest to 68.
                row["pol_unc_stat_ret_age_67"] = 0.1
                row["pol_unc_stat_ret_age_68"] = 0.4
                row["pol_unc_stat_ret_age_69"] = 99.5

            # observed CDF values
            function_spec["observed_cdf_67_5"] = (
                row["pol_unc_stat_ret_age_67"] / 100
            )  # CDF(67.5)
            function_spec["observed_cdf_68_5"] = (
                row["pol_unc_stat_ret_age_68"] + row["pol_unc_stat_ret_age_67"]
            ) / 100  # CDF(68.5)

            # initial guess for mean and variance
            initial_guess = np.array([68, 2])
            partial_obj = partial(objective, function_spec=function_spec)

            # perform optimization
            result = root(fun=partial_obj, x0=initial_guess, tol=1e-16)

            # collect results
            loc, scale = result.x
            df.at[index, "mu"] = loc
            df.at[index, "sigma"] = scale
            df.at[index, "error_1"], df.at[index, "error_2"] = result.fun

            # calculate expected value
            a, b = (function_spec["ll"] - loc) / scale, (
                function_spec["ul"] - loc
            ) / scale
            exval, var = truncnorm.stats(a, b, loc=loc, scale=scale, moments="mv")
            df.at[index, "ex_val"] = exval
            df.at[index, "var"] = var
    return df


# objective function to be minimized (squared differences between observed and predicted CDF values)
def objective(params, function_spec):
    mu, sigma = params
    a, b = (function_spec["ll"] - mu) / sigma, (function_spec["ul"] - mu) / sigma

    # Calculate CDF values from the truncated normal distribution
    predicted_cdf_1 = truncnorm.cdf(
        x=function_spec["first_cdf_point"], a=a, b=b, loc=mu, scale=sigma
    )
    predicted_cdf_2 = truncnorm.cdf(
        x=function_spec["second_cdf_point"], a=a, b=b, loc=mu, scale=sigma
    )
    # Calculate the squared differences
    error_1 = (function_spec["observed_cdf_67_5"] - predicted_cdf_1) ** 2
    error_2 = (function_spec["observed_cdf_68_5"] - predicted_cdf_2) ** 2
    # Sum of squared differences
    # total_error = error_1 + error_2
    return np.array([error_1, error_2])

