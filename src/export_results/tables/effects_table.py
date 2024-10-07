import numpy as np
import pandas as pd
from export_results.tables.cv import calc_compensated_variation
from specs.derive_specs import generate_derived_and_data_derived_specs

"""TODO: ROunding everywhere instead of cutting off decimals"""


def create_effects_table(df_base, df_cf, params, path_dict, scenario_name):
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    sol_table = pd.DataFrame()
    # Savings
    savings_base = int(calc_savings_increase(df_base) * specs["wealth_unit"])
    savings_cf = int(calc_savings_increase(df_cf) * specs["wealth_unit"])
    sol_table.loc["savings", "baseline"] = f"{savings_base}"
    sol_table.loc["savings", "counterfactual"] = f"{savings_cf}"
    savings_diff = savings_cf - savings_base
    sol_table.loc["savings", "diff"] = f"{savings_diff}"

    perc_change_savings = savings_diff / savings_base * 100
    if savings_diff > 0:
        sol_table.loc["savings", "perc_change"] = f"+{perc_change_savings:.2f}\%"
    else:
        sol_table.loc["savings", "perc_change"] = f"{perc_change_savings:.2f}\%"

    # Labor supply
    labor_supply_base = calc_labor_supply_diff(df_base) * 100
    labor_supply_cf = calc_labor_supply_diff(df_cf) * 100
    sol_table.loc["labor supply", "baseline"] = f"{labor_supply_base:.2f}\%"
    sol_table.loc["labor supply", "counterfactual"] = f"{labor_supply_cf:.2f}\%"
    labor_supply_diff = labor_supply_cf - labor_supply_base
    sol_table.loc["labor supply", "diff"] = f"{labor_supply_diff:.2f}\%"

    perc_change_labor_supply = labor_supply_diff / labor_supply_base * 100

    if labor_supply_diff > 0:
        sol_table.loc[
            "labor supply", "perc_change"
        ] = f"+{perc_change_labor_supply:.2f}\%"
    else:
        sol_table.loc[
            "labor supply", "perc_change"
        ] = f"{perc_change_labor_supply:.2f}\%"

    # Average retirement age
    average_retirement_age_base = av_ret_age(df_base)
    average_retirement_age_cf = av_ret_age(df_cf)
    sol_table.loc[
        "average retirement age", "baseline"
    ] = f"{average_retirement_age_base:.2f}"
    sol_table.loc[
        "average retirement age", "counterfactual"
    ] = f"{average_retirement_age_cf:.2f}"
    average_retirement_age_diff = (
        average_retirement_age_cf - average_retirement_age_base
    )
    av_ret_age_diff_month = average_retirement_age_diff * 12

    sol_table.loc[
        "average retirement age", "diff"
    ] = f"{av_ret_age_diff_month:.2f} months"

    av_ret_age_perc_change = (
        average_retirement_age_diff / average_retirement_age_base * 100
    )

    if average_retirement_age_diff > 0:
        sol_table.loc[
            "average retirement age", "perc_change"
        ] = f"+{av_ret_age_perc_change:.2f}\%"
    else:
        sol_table.loc[
            "average retirement age", "perc_change"
        ] = f"{av_ret_age_perc_change:.2f}\%"

    # Consumption variation
    cv = np.round(
        calc_compensated_variation(df_base, df_cf, params, specs) * 100, decimals=2
    )
    sol_table.loc["compensated variation", "diff"] = f"{cv}\%"
    print(cv)

    # for edu_val, edu_label in enumerate(specs["education_labels"]):
    #     df_base_edu = df_base[df_base["education"] == edu_val]
    #     df_cf_edu = df_cf[df_cf["education"] == edu_val]
    #
    #     savings_increase_edu = calc_savings_increase(df_base_edu, df_cf_edu)
    #     av_ret_age_diff_edu, av_ret_age_base_edu = cacl_av_ret_age_diff_months(
    #         df_base_edu, df_cf_edu
    #     )
    #     labor_supply_diff_edu = calc_labor_supply_diff(df_base_edu, df_cf_edu)
    #     cv_edu = calc_compensated_variation(df_base_edu, df_cf_edu, params, specs)
    #
    #     sol_table.loc[edu_label] = [
    #         savings_increase_edu,
    #         av_ret_age_diff_edu,
    #         av_ret_age_base_edu,
    #         labor_supply_diff_edu,
    #         cv_edu,
    #     ]

    sol_table.to_latex(path_dict["tables"] + f"effects_table_{scenario_name}.tex")
    sol_table.to_csv(path_dict["tables"] + f"effects_table_{scenario_name}.csv")


def calc_savings_increase(df):
    savings_bef_60 = (df.groupby("age")["savings_dec"].mean().loc[slice(30, 60)]).mean()

    return savings_bef_60


def av_ret_age(df_base):
    av_ret_age = df_base[(df_base["choice"] == 2) & (df_base["lagged_choice"] != 2)][
        "age"
    ].mean()

    return av_ret_age


def calc_labor_supply_diff(df):
    labor_suppl = (
        df.groupby("age")["choice"].value_counts(normalize=True).loc[(slice(30, 60), 1)]
    ).mean()

    return labor_suppl
