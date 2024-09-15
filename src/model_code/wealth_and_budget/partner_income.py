from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_pensioneer
from model_code.wealth_and_budget.tax_and_ssc import calc_after_ssc_income_worker


def calc_partner_income_after_ssc(partner_state, options, education, period):
    """Calculate the partner income after deduction of ssc."""
    partner_wage = options["partner_wage"][education, period] * 12
    partner_wage_after_ssc = calc_after_ssc_income_worker(
        gross_wage=partner_wage, options=options
    )
    partner_pension_after_ssc = calc_after_ssc_income_pensioneer(
        gross_pesnion=options["partner_pension"][education], options=options
    )
    working_age_partner = partner_state == 1
    retired_partner = partner_state == 2
    partner_income = (
        working_age_partner * partner_wage_after_ssc
        + retired_partner * partner_pension_after_ssc
    )
    return partner_income
