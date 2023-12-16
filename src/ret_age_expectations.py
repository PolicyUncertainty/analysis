# %%
import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.optimize import minimize, root
from functools import partial
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

# %%
# open dataset
relevant_cols = ["pol_unc_stat_ret_age_67", "pol_unc_stat_ret_age_68", "pol_unc_stat_ret_age_69", "gebjahr"]
df = pd.read_stata("../../data/dataset_main_SOEP_IS.dta")[relevant_cols].astype(float)
df.rename({"gebjahr": "birth_year"}, axis=1, inplace=True)

# %%
# add columns for mean and stdev
df["mu"] = np.nan
df["sigma"] = np.nan
df["ex_val"] = np.nan
df["var"] = np.nan
df["error_1"] = np.nan
df["error_2"] = np.nan

# %%
# das wird später noch eleganter
options = {
    "ll": 66.5,
    "ul": 80,
    "first_cdf_point": 67.5,
    "second_cdf_point": 68.5,
}

# %%
# objective function to be minimized
def objective(params, options):
    mu, sigma = params
    # mu = params
    # sigma = 1 #TEST
    a, b = (options["ll"] - mu) / sigma, (options["ul"] - mu) / sigma

    # Calculate CDF values from the truncated normal distribution
    predicted_cdf_1 = truncnorm.cdf(
        x=options["first_cdf_point"], a=a, b=b, loc=mu, scale=sigma
    )
    predicted_cdf_2 = truncnorm.cdf(
        x=options["second_cdf_point"], a=a, b=b, loc=mu, scale=sigma
    )
    # Calculate the squared differences
    error_1 = (options["observed_cdf_67_5"] - predicted_cdf_1) ** 2
    error_2 = (options["observed_cdf_68_5"] - predicted_cdf_2) ** 2
    # Sum of squared differences
    # total_error = error_1 + error_2
    return np.array([error_1, error_2])

# %%
for index, row in df.iterrows():
    if not np.isnan(row["pol_unc_stat_ret_age_67"]):
        # observed CDF values
        options["observed_cdf_67_5"] = row["pol_unc_stat_ret_age_67"] / 100  # CDF(67.5)
        options["observed_cdf_68_5"] = (
            row["pol_unc_stat_ret_age_68"] + row["pol_unc_stat_ret_age_67"]
        ) / 100  # CDF(68.5)

        if row["pol_unc_stat_ret_age_67"] == 100:
            df.at[index, "ex_val"] = 67
            df.at[index, "var"] = 0
        elif row["pol_unc_stat_ret_age_68"] == 100:
            df.at[index, "ex_val"] = 68
            df.at[index, "var"] = 0
        elif row["pol_unc_stat_ret_age_69"] == 100:
            # Mean of last interval
            df.at[index, "ex_val"] = 74.25
            # Variance of uniform distribution between start of last
            # interval and upper limit
            df.at[index, "var"] = ((80 - 68.5) ** 2) / 12
        else:
            # initial guess for mean and variance
            initial_guess = [68, 1]
            # initial_guess = 68
            partial_obj = partial(objective, options=options)

            # perform optimization
            # result = minimize(partial_obj, initial_guess, method='BFGS')
            result = root(fun=partial_obj, x0=initial_guess)

            # collect results
            mean, stdev = result.x
            # mean =  result.x #TEST
            # stdev = 1 #TEST
            df.at[index, "mu"] = mean
            df.at[index, "sigma"] = stdev
            df.at[index, "error_1"], df.at[index, "error_2"] = result.fun

            # calculate expected value
            a, b = (options["ll"] - mean) / stdev, (options["ul"] - mean) / stdev
            exval, var = truncnorm.stats(a, b, loc=mean, scale=stdev, moments="mv")
            df.at[index, "ex_val"] = exval
            df.at[index, "var"] = var

    # print(index)

# %%
mean = df[["ex_val"]]
mean.rename({"ex_val":"SRA_mean"}).to_stata("../../../stylized_ltc_ret/data/SRA_mean.dta")

# %%
df_analysis = df[df["pol_unc_stat_ret_age_67"].notnull()]

# %%
df_analysis.groupby("birth_year")["ex_val"].mean().plot()

# %%
exog_1 = np.array([np.ones(df_analysis.shape[0]), df_analysis["birth_year"].values]).T

# %%
OLS(exog=exog_1, endog=df_analysis["var"].values).fit().summary()

# %%
OLS(exog=exog_1, endog=df_analysis["ex_val"].values).fit().summary()

# %%



