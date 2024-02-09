import numpy as np
from matplotlib import pyplot as plt
from process_data.steps.gather_decision_data import create_policy_state
from statsmodels import api as sm


def gen_exp_val_params_and_plot(paths, df, load_data=False):
    output_file = paths["project_path"] + "output/exp_val_params.txt"
    x_var = "time_to_ret"
    weights = "fweights"
    # calculate current policy state for each observation
    df["current_SRA"] = create_policy_state(df["birth_year"])
    df["exp_SRA_increase"] = df["ex_val"] - df["current_SRA"]
    y_var = "exp_SRA_increase"
    # y_var = "ex_val"

    if load_data:
        return np.loadtxt(output_file)

    model = sm.WLS(
        exog=df[x_var].values,
        endog=df[y_var].values,
        weights=df[weights].values,
    )
    print(model.fit().summary())
    alpha_hat = model.fit().params
    fig, ax = plt.subplots()
    ax.scatter(df[x_var], df[y_var], s=(df[weights] / df[weights].sum()) * 5000)
    ax.plot(
        df[x_var],
        alpha_hat[0] * df[x_var],
        "r",
    )
    ax.set_xlabel("Time to retirement")
    ax.set_ylabel(
        "Expected retirement age",
    )
    plt.savefig(paths["project_path"] + "output/exp_val_plot.png")
    plt.show()
    print(
        f"Estimated regression equation: E[ret age change] = "
        f"{alpha_hat[0]} * (Time to retirement)"
    )

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

    fig, ax = plt.subplots()
    ax.scatter(df[x_var], df[y_var], s=(df[weights] / df[weights].sum()) * 5000)
    ax.plot(
        df[x_var],
        sigma_sq_hat[0] * df[x_var],
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
