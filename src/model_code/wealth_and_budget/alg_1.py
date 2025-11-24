from model_code.wealth_and_budget.tax_and_ssc import calc_inc_tax_for_single_income
from model_code.wealth_and_budget.wages import calc_labor_income_after_ssc


def calc_potential_alg_1(
    sex,
    education,
    age,
    experience_years,
    model_specs,
):
    labor_income_after_ssc_pt, _ = calc_labor_income_after_ssc(
        lagged_choice=2,
        experience_years=experience_years,
        age=age,
        education=education,
        sex=sex,
        income_shock=0.0,
        model_specs=model_specs,
    )

    labor_income_after_ssc_ft, _ = calc_labor_income_after_ssc(
        lagged_choice=3,
        experience_years=experience_years,
        age=age,
        education=education,
        sex=sex,
        income_shock=0.0,
        model_specs=model_specs,
    )
    # Full time weight is 1 for men and 0.5 for women
    is_men = sex == 0
    ft_weight = is_men + (1 - is_men) * 0.5
    labor_income_after_ssc = (
        labor_income_after_ssc_pt * (1 - ft_weight)
        + labor_income_after_ssc_ft * ft_weight
    )
    # Apply tax function to get net salary
    inc_tax_single = calc_inc_tax_for_single_income(
        gross_income=labor_income_after_ssc, model_specs=model_specs
    )
    net_labor_income = labor_income_after_ssc - inc_tax_single
    # Apply replacement rate of 0.67
    potential_alg_1 = net_labor_income * 0.67
    return potential_alg_1
