def calc_pension_annuity_value(
    pension_payments,
    payment_years,
    interest_rate,
):
    # calculate perpetuity of getting pension income after SRA, then subtract perpetuity after death
    discount_factor = 1 / (1 + interest_rate)
    annuity_factor = 1 / (1 - discount_factor)
    pension_annuity_value = (
        pension_payments * annuity_factor
        - pension_payments * annuity_factor * discount_factor**payment_years
    )
    return pension_annuity_value
