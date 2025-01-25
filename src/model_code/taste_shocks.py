def shock_function_dict():
    return {
        "taste_shock_scale_per_state": taste_shock_for_sexes,
    }


def taste_shock_for_sexes(sex, params):
    return (
        params["taste_shock_scale_men"] * (1 - sex)
        + params["taste_shock_scale_women"] * sex
    )
