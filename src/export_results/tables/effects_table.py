import pandas as pd
from export_results.tables.cv import calc_compensated_variation
from specs.derive_specs import generate_derived_and_data_derived_specs


def create_effects_table(df_base, df_cf, params, path_dict, scenario_name):
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)

    savings_increase_all = calc_savings_increase(df_base, df_cf)
    av_ret_age_diff_all = cacl_av_ret_age_diff_months(df_base, df_cf)
    labor_supply_diff_all = calc_labor_supply_diff(df_base, df_cf)
    cv = calc_compensated_variation(df_base, df_cf, params)

    sol_table = pd.DataFrame(
        index=["all"],
        columns=[
            "savings_increase",
            "av_ret_age_diff_months",
            "labor_supply_diff_ppp",
            "cv_cons_increase",
        ],
        data=[[savings_increase_all, av_ret_age_diff_all, labor_supply_diff_all, cv]],
    )
    for edu_val, edu_label in enumerate(specs["education_labels"]):
        df_base_edu = df_base[df_base["education"] == edu_val]
        df_cf_edu = df_cf[df_cf["education"] == edu_val]

        savings_increase_edu = calc_savings_increase(df_base_edu, df_cf_edu)
        av_ret_age_diff_edu = cacl_av_ret_age_diff_months(df_base_edu, df_cf_edu)
        labor_supply_diff_edu = calc_labor_supply_diff(df_base_edu, df_cf_edu)
        cv_edu = calc_compensated_variation(df_base_edu, df_cf_edu, params)

        sol_table.loc[edu_label] = [
            savings_increase_edu,
            av_ret_age_diff_edu,
            labor_supply_diff_edu,
            cv_edu,
        ]

    sol_table.to_latex(path_dict["tables"] + f"effects_table_{scenario_name}.tex")
    sol_table.to_csv(path_dict["tables"] + f"effects_table_{scenario_name}.csv")


def calc_savings_increase(df_base, df_cf):
    base_med_savings_bef_60 = (
        df_base.groupby("age")["savings_dec"].median().loc[slice(30, 60)]
    )
    cf_med_savings_bef_60 = (
        df_cf.groupby("age")["savings_dec"].median().loc[slice(30, 60)]
    )
    return (cf_med_savings_bef_60 / base_med_savings_bef_60).mean()


def cacl_av_ret_age_diff_months(df_base, df_cf):
    av_ret_age_base = df_base[
        (df_base["choice"] == 2) & (df_base["lagged_choice"] != 2)
    ]["age"].mean()
    av_ret_age_cf = df_cf[(df_cf["choice"] == 2) & (df_cf["lagged_choice"] != 2)][
        "age"
    ].mean()

    return (av_ret_age_cf - av_ret_age_base) * 12


def calc_labor_supply_diff(df_base, df_cf):
    labor_suppl_base = (
        df_base.groupby("age")["choice"]
        .value_counts(normalize=True)
        .loc[(slice(30, 60), 1)]
    )
    labor_supply_cf = (
        df_cf.groupby("age")["choice"]
        .value_counts(normalize=True)
        .loc[(slice(30, 60), 1)]
    )

    return (labor_supply_cf - labor_suppl_base).mean()
