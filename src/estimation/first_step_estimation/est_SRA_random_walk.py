import numpy as np
from statsmodels import api as sm


def est_expected_SRA(paths, df):
    x_var = "time_to_ret"
    weights = "fweights"

    # truncate data: remove birth years outside before 1964
    df = df.query("birth_year >= 1964")

    # calculate current policy state for each observation
    df["exp_SRA_increase"] = df["ex_val"] - df["current_SRA"]

    y_var = "exp_SRA_increase"
    # y_var = "ex_val"

    model = sm.WLS(
        exog=df[x_var].values,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    print(model.fit().summary())
    alpha_hat = model.fit().params

    print(
        f"Estimated regression equation: E[ret age change] = "
        f"{alpha_hat[0]} * (Time to retirement)"
    )

    np.savetxt(paths["est_results"] + "exp_val_params.txt", alpha_hat, delimiter=",")
    return alpha_hat


def estimate_expected_SRA_variance(paths, df):
    # truncate data: remove birth years outside before 1964
    df = df.query("birth_year >= 1964")

    # divide estimated variances by time to retirement
    sigma_sq_hat = df["var"] / df["time_to_ret"]

    # weight and take average
    sigma_sq_hat = (sigma_sq_hat * df["fweights"]).sum() / df["fweights"].sum()
    sigma_sq_hat = np.array([sigma_sq_hat])

    # regress variance on time to retirement without constant

    # exog_1 = np.array([np.ones(df.shape[0]), df[x_var].values]).T
    #
    # model = sm.WLS(
    #     exog=df[x_var].values,
    #     endog=df[y_var].values,
    #     weights=df[weights].values,
    # )
    # print(model.fit().summary())
    # coefficients = model.fit().params

    np.savetxt(paths["est_results"] + "var_params.txt", sigma_sq_hat, delimiter=",")
    return sigma_sq_hat
