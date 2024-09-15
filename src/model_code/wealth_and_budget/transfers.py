def calc_child_benefits(education, has_partner, period, options):
    """Calculate the child benefits."""
    nb_children = options["children_by_state"][0, education, has_partner, period]
    return nb_children * options["child_benefit"] * 12


def calc_unemployment_benefits(savings, education, has_partner_int, period, options):
    # Unemployment benefits
    means_test = savings < options["unemployment_wealth_thresh"]

    # Unemployment benefits for children living in the household
    nb_children = options["children_by_state"][0, education, has_partner_int, period]
    unemployment_benefits_children = (
        nb_children * options["child_unemployment_benefits"] * 12
    )

    # Unemployment benefits for adults living in the household
    nb_adults = 1 + has_partner_int
    unemployment_benefits_adults = nb_adults * options["unemployment_benefits"] * 12

    # Total unemployment benefits
    unemployment_benefits = means_test * (
        unemployment_benefits_adults + unemployment_benefits_children
    )
    return unemployment_benefits
