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


def utility_func(consumption, choice, params):
    # Reading parameters
    mu = params["mu"]
    dis_util_work = params["dis_util_work"]
    dis_util_unemployed = params["dis_util_unemployed"]
    # Check which choice we have
    is_working = choice == 1
    is_unemployed = choice == 0
    # Select which dis-utility to use. Retirement is 0 as baseline
    dis_utility = dis_util_work * is_working + dis_util_unemployed * is_unemployed
    # Compute utility
    utility = consumption ** (1 - mu) / (1 - mu) - dis_utility
    return utility


def marg_utility(consumption, params):
    mu = params["mu"]
    marg_util = consumption**-mu
    return marg_util


def inverse_marginal(marginal_utility, params):
    mu = params["mu"]
    return marginal_utility ** (-1 / mu)


def utility_final_consume_all(
    choice,
    resources,
    params,
    options,
):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (resources ** (1 - mu) / (1 - mu))


def marginal_utility_final_consume_all(choice, resources, params, options):
    mu = params["mu"]
    bequest_scale = params["bequest_scale"]
    return bequest_scale * (resources**-mu)
