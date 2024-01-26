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
    mu = params["mu"]
    delta = params["delta"]
    is_working = choice == 1
    utility = consumption ** (1 - mu) / (1 - mu) - delta * is_working
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
    return utility_func(consumption=resources, choice=choice, params=params)


def marginal_utility_final_consume_all(choice, resources, params, options):
    return marg_utility(consumption=resources, params=params)
