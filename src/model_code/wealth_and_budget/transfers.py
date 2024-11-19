def calc_child_benefits(education, has_partner_int, period, options):
    """Calculate the child benefits."""
    nb_children = options["children_by_state"][0, education, has_partner_int, period]
    return nb_children * options["monthly_child_benefits"] * 12


def calc_unemployment_benefits(savings, education, has_partner_int, period, options):
    # Unemployment benefits
    means_test = savings < options["unemployment_wealth_thresh"]

    # Unemployment benefits for children living in the household
    nb_children = options["children_by_state"][0, education, has_partner_int, period]
    unemployment_benefits_children = (
        nb_children * options["annual_child_unemployment_benefits"]
    )

    # Unemployment benefits for adults living in the household
    unemployment_benefits_adults = (1 + has_partner_int) * options[
        "annual_unemployment_benefits"
    ]
    # For housing, second adult gets only half
    unemployment_benefits_housing = (1 + 0.5 * has_partner_int) * options[
        "annual_unemployment_benefits_housing"
    ]

    # Total unemployment benefits
    unemployment_benefits = means_test * (
        unemployment_benefits_adults
        + unemployment_benefits_children
        + unemployment_benefits_housing
    )

    return unemployment_benefits
