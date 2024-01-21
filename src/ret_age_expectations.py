# This script estimates the parameters of the truncated normal distribution of subjective retirement age expectations
# based on the observed CDF values from the SOEP-IS dataset. The parameters are estimated for each respondent individually.
# Then, the expected values of the individual CDF are regressed on birth year to obtain the slope of the policy expectation process.


import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize, root
from functools import partial
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

def estimate_policy_expectation_parameters(paths, options, load_data=False):
    if load_data:
        coefficients = pd.read_csv("output/policy_expectation_params.csv")
        coefficients = coefficients.iloc[:,[1]]
        return coefficients

    # unpack path to SOEP-IS
    soep_is = paths["soep_is"]

    # unpack options, put into new dict 
    function_spec = {
        "ll": options["lower_limit"],
        "ul": options["upper_limit"],
        "first_cdf_point": options["first_cdf_point"],
        "second_cdf_point": options["second_cdf_point"],
    }

    # load dataset (policy uncertainty questions from SOEP-IS)
    relevant_cols = ["pol_unc_stat_ret_age_67", "pol_unc_stat_ret_age_68", "pol_unc_stat_ret_age_69", "gebjahr"]
    df = pd.read_stata(soep_is)[relevant_cols].astype(float)
    df.rename({"gebjahr": "birth_year"}, axis=1, inplace=True)

    # estimate params of truncated normal, as well as mean and var
    df = estimate_truncated_normal_parameters(df, function_spec)
    
    # exclude people born before 1947 and people born after 2000
    df_analysis = df[df["pol_unc_stat_ret_age_67"].notnull()]
    df_analysis = df_analysis[df_analysis["birth_year"] >= options["min_birth_year"]]
    df_analysis = df_analysis[df_analysis["birth_year"] <= options["max_birth_year"]]

    # plot means by birth year
    df_analysis.groupby("birth_year")["ex_val"].mean().plot()

    # regress expected value on birth year minus minimum birth year, save OLS results
    df_analysis["birth_year"] = df_analysis["birth_year"] - options["min_birth_year"]
    exog_1 = np.array([np.ones(df_analysis.shape[0]), df_analysis["birth_year"].values]).T 
    model = OLS(exog=exog_1, endog=df_analysis["ex_val"].values)
    #model.fit().summary()
    coefficients = pd.DataFrame(model.fit().params)
    print("Estimated regression equation: E[ret age] = {} + {} * (birth year - {})".format(coefficients.iloc[0,0], coefficients.iloc[1,0], options["min_birth_year"]))
    coefficients.to_csv("output/policy_expectation_params.csv")
    return coefficients


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
            # observed CDF values
            function_spec["observed_cdf_67_5"] = row["pol_unc_stat_ret_age_67"] / 100  # CDF(67.5)
            function_spec["observed_cdf_68_5"] = (
                row["pol_unc_stat_ret_age_68"] + row["pol_unc_stat_ret_age_67"]
            ) / 100  # CDF(68.5)

            if row["pol_unc_stat_ret_age_67"] == 100:
                df.at[index, "ex_val"] = 67
                df.at[index, "var"] = 0
            elif row["pol_unc_stat_ret_age_68"] == 100:
                df.at[index, "ex_val"] = 68
                df.at[index, "var"] = 0
            elif row["pol_unc_stat_ret_age_69"] == 100:
                # Mean of last interval (which goes from 68.5 to specified upper limit, e.g. 80)
                df.at[index, "ex_val"] = 68.5 +  (function_spec["ul"] - 68.5) / 2
                # Variance of uniform distribution between start of last interval and upper limit
                df.at[index, "var"] = ((80 - 68.5) ** 2) / 12
            else:
                # initial guess for mean and variance
                initial_guess = [68, 1]
                partial_obj = partial(objective, function_spec=function_spec)

                # perform optimization
                result = root(fun=partial_obj, x0=initial_guess)

                # collect results
                mean, stdev = result.x
                df.at[index, "mu"] = mean
                df.at[index, "sigma"] = stdev
                df.at[index, "error_1"], df.at[index, "error_2"] = result.fun

                # calculate expected value
                a, b = (function_spec["ll"] - mean) / stdev, (function_spec["ul"] - mean) / stdev
                exval, var = truncnorm.stats(a, b, loc=mean, scale=stdev, moments="mv")
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