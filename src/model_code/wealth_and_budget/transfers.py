def calc_child_benefits(sex, education, has_partner_int, period, model_specs):
    """Calculate the child benefits."""
    nb_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    return nb_children * model_specs["monthly_child_benefits"] * 12


def calc_unemployment_benefits(
    assets, sex, education, has_partner_int, period, model_specs
):
    # Unemployment benefits means test
    means_test = assets < model_specs["unemployment_wealth_thresh"]

    # Unemployment benefits for children living in the household
    nb_children = model_specs["children_by_state"][
        sex, education, has_partner_int, period
    ]
    unemployment_benefits_children = (
        nb_children * model_specs["annual_child_unemployment_benefits"]
    )

    own_unemployemnt_benefits = (
        model_specs["annual_unemployment_benefits"]
        + model_specs["annual_unemployment_benefits_housing"]
    )

    partner_unemployment_benefits = has_partner_int * (
        model_specs["annual_unemployment_benefits"]
        + model_specs["annual_unemployment_benefits_housing"] * 0.5
    )

    # Total unemployment benefits
    total_unemployment_benefits = (
        own_unemployemnt_benefits
        + partner_unemployment_benefits
        + unemployment_benefits_children
    )

    # reduced benefits for savings slightly above threshold
    reduced_benefits_threshhold = (
        model_specs["unemployment_wealth_thresh"] + total_unemployment_benefits
    )
    reduced_benefits_means_test = (1 - means_test) * (
        assets < reduced_benefits_threshhold
    )
    reduced_benefits = reduced_benefits_threshhold - assets

    unemployment_benefits = (
        means_test * total_unemployment_benefits
        + reduced_benefits_means_test * reduced_benefits
    )
    return unemployment_benefits, own_unemployemnt_benefits
