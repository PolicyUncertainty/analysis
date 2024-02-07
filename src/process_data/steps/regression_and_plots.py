import numpy as np
from matplotlib import pyplot as plt
from statsmodels import api as sm


def gen_exp_val_params_and_plot(paths, df, load_data=False):
    output_file = paths["project_path"] + "output/exp_val_params.txt"
    x_var = "time_to_ret"
    y_var = "ex_val"
    weights = "fweights"

    if load_data:
        return np.loadtxt(output_file)

    exog_1 = np.array([np.ones(df.shape[0]), df[x_var].values]).T
    model = sm.WLS(
        exog=exog_1,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    print(model.fit().summary())
    coefficients = model.fit().params
    fig, ax = plt.subplots()
    ax.scatter(df[x_var], df[y_var], s=(df[weights] / df[weights].sum()) * 5000)
    ax.plot(
        df[x_var],
        coefficients[0] + coefficients[1] * df[x_var],
        "r",
    )
    ax.set_xlabel("Time to retirement")
    ax.set_ylabel(
        "Expected retirement age",
    )
    plt.savefig(paths["project_path"] + "output/exp_val_plot.png")
    plt.show()
    print(
        f"Estimated regression equation: E[ret age] = {coefficients[0]} + "
        f"{coefficients[1]} * (Time to retirement)"
    )

    # save relevant parameter
    alpha_hat = coefficients[1:]


    np.savetxt(output_file, alpha_hat, delimiter=",")
    return alpha_hat


def gen_var_params_and_plot(paths, df, load_data=False):
    output_file = paths["project_path"] + "output/var_params.txt"
    x_var = "time_to_ret"
    y_var = "var"
    weights = "fweights"


    if load_data:
        return np.loadtxt(output_file)

    # divide estimated variances by time to retirement
    sigma_sq_hat = df["var"] / df["time_to_ret"]

    # weight and take average
    sigma_sq_hat = (sigma_sq_hat * df["fweights"]).sum() / df["fweights"].sum()
    sigma_sq_hat = np.array([sigma_sq_hat])

    # this is a mostly useless regression that is only used to generate a plot to 
    # illustrate why we need to improve the model

    exog_1 = np.array([np.ones(df.shape[0]), df[x_var].values]).T
    model = sm.WLS(
        exog=exog_1,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    print(model.fit().summary())
    coefficients = model.fit().params
    fig, ax = plt.subplots()
    ax.scatter(df[x_var], df[y_var], s=(df[weights] / df[weights].sum()) * 5000)
    ax.plot(
        df[x_var],
        coefficients[0] + coefficients[1] * df[x_var],
        "r",
    )
    ax.set_xlabel("Time to retirement")
    ax.set_ylabel(
        "Expected retirement age",
    )
    plt.savefig(paths["project_path"] + "output/var_plot.png")
    plt.show()

    np.savetxt(output_file, sigma_sq_hat, delimiter=",")
    return sigma_sq_hat
