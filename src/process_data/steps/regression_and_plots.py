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
    np.savetxt(output_file, coefficients, delimiter=",")
    return coefficients


def gen_var_params_and_plot(paths, df, load_data=False):
    output_file = paths["project_path"] + "output/var_params.txt"
    x_var = "time_to_ret"
    y_var = "var"
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
    plt.savefig(paths["project_path"] + "output/var_plot.png")
    plt.show()

    np.savetxt(output_file, coefficients, delimiter=",")
    return coefficients
