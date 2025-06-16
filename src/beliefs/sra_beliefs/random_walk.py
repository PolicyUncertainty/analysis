import numpy as np
import pandas as pd
from statsmodels import api as sm


def est_SRA_params(paths, df=None):
    # load (if df is None) and filter data
    if df is None:
        df = pd.read_csv(
            paths["intermediate_data"] + "beliefs/soep_is_truncated_normals.csv"
        )
    df = filter_df(df)
    # estimate expected SRA increase and variance
    alpha_hat, alpha_hat_std_err = est_expected_SRA(paths, df)
    sigma_sq_hat, sigma_sq_hat_std_err = estimate_expected_SRA_variance(paths, df)
    columns = ["parameter", "estimate", "std_error"]
    results_df = pd.DataFrame(
        columns=columns,
        data=[
            ["alpha", alpha_hat[0], alpha_hat_std_err[0]],
            ["sigma_sq", sigma_sq_hat[0], sigma_sq_hat_std_err[0]],
        ],
    )
    return results_df


def est_expected_SRA(paths, df=None, print_summary=False):
    # set up regression
    x_var = "time_to_ret"
    weights = "fweights"
    df["exp_SRA_increase"] = df["ex_val"] - df["current_SRA"]
    y_var = "exp_SRA_increase"
    # regress expected SRA increase on time to retirement without constant
    model = sm.WLS(
        exog=df[x_var].values,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    if print_summary:
        print(model.fit().summary())
        print(
            f"Estimated regression equation: E[ret age change] = "
            f"{alpha_hat[0]} * (Time to retirement)"
        )
    alpha_hat = model.fit().params
    alpha_hat_std_err = model.fit().bse
    return alpha_hat, alpha_hat_std_err


def estimate_expected_SRA_variance(paths, df=None, print_summary=False):
    # set up regression
    x_var = "time_to_ret"
    weights = "fweights"
    y_var = "var"
    # save dataset to stata
    df.to_stata(paths["intermediate_data"] + "temp_bruno.dta", write_index=False)
    # regress estimated variance on time to retirement without constant
    model = sm.WLS(
        exog=df[x_var].values,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    if print_summary:
        print(model.fit().summary())
        print(
            f"Estimated regression equation: E[ret age variance] = "
            f"{sigma_sq_hat[0]} * (Time to retirement)"
        )
    sigma_sq_hat = model.fit().params
    sigma_sq_hat_std_err = model.fit().bse
    return sigma_sq_hat, sigma_sq_hat_std_err


def estimate_expected_SRA_variance_by_taking_average(paths, df=None):
    if df is None:
        df = pd.read_pickle(paths["intermediate_data"] + "policy_expect_data.pkl")

    # truncate data: remove birth years outside before 1964
    df = filter_df(df)

    # divide estimated variances by time to retirement
    sigma_sq_hat = df["var"] / df["time_to_ret"]

    # weight and take average
    sigma_sq_hat = (sigma_sq_hat * df["fweights"]).sum() / df["fweights"].sum()
    sigma_sq_hat = np.array([sigma_sq_hat])

    sigma_sq_hat_std_err = np.std(df["var"] / df["time_to_ret"]) / np.sqrt(len(df))

    return sigma_sq_hat, sigma_sq_hat_std_err


def filter_df(df):
    """Drop observations of people born before 1964 and drop missing subjective expectation parameters."""
    df = df[df["gebjahr"] >= 1964]
    df = df.dropna(subset=["ex_val", "var", "fweights", "time_to_ret"])
    print(
        f"Filtered data: {len(df)} observations remaining after dropping birth years before 1964, and people with missing values in subjective expectation parameters."
    )
    return df
