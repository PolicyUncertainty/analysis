import matplotlib.pyplot as plt
import numpy as np

from model_code.wealth_and_budget.pension_payments import calc_pensions_after_ssc
from specs.derive_specs import generate_derived_and_data_derived_specs


def plot_pension_npv_by_age(paths, edu=0):
    raise ValueError("Not maintained right now.")
    # Generate derived specs
    specs = generate_derived_and_data_derived_specs(paths)
    # Select highes pension point value
    pension_point_value = specs["pension_point_value"][edu, -1]
    start_age = specs["start_age"]
    end_age = specs["end_age"]
    discount_rate = 0.03

    # calculate net periodic retirement income (assumption: working until 67)
    pension_factor = 1
    experience = 67 - start_age
    retirement_income_gross = pension_point_value * experience * pension_factor * 12
    retirement_income_net = calc_pensions_after_ssc(retirement_income_gross, specs)

    # calculate net present value of retirement income at age 67
    npv_67 = retirement_income_net / discount_rate - (
        retirement_income_net / discount_rate
    ) / (1 + discount_rate) ** (end_age - 67)

    # calculate net present value of retirement income at different ages
    npv_by_age = np.full(67 - start_age + 1, npv_67)
    discount_factor_by_age = np.power(1 + discount_rate, np.arange(67 - start_age + 1))
    npv_by_age = npv_by_age / discount_factor_by_age
    npv_by_age_reversed = npv_by_age[::-1]
    plt.plot(np.arange(start_age, 68), npv_by_age_reversed)
    plt.xlabel("Age")
    plt.ylabel("NPV of retirement income (1000 €)")
