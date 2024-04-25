def create_utility_functions():
    return {
        "utility": utility_func,
        "inverse_marginal_utility": inverse_marginal,
        "marginal_utility": marg_utility,
    }


def create_final_period_utility_functions():
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def utility_func(consumption, params):
    # Reading parameters
    mu = params["mu"]
    # Select which dis-utility to use. Retirement is 0 as baseline
    # Compute utility
    utility = (consumption ** (1 - mu) - 1) / (1 - mu)
    return utility


def marg_utility(consumption, params):
    mu = params["mu"]
    marg_util = consumption**-mu
    return marg_util


def inverse_marginal(marginal_utility, params):
    mu = params["mu"]
    return marginal_utility ** (-1 / mu)


def utility_final_consume_all(
    resources,
    params,
):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (resources ** (1 - mu) / (1 - mu))


def marginal_utility_final_consume_all(resources, params):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (resources**-mu)
