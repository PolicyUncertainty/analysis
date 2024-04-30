def create_final_period_utility_functions():
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


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
