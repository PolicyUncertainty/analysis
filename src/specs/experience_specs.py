import jax.numpy as jnp
import numpy as np
import pandas as pd

from model_code.pension_system.experience_stock import (
    calc_experience_years_for_pension_adjustment,
)


def create_max_experience(path_dict, specs, load_precomputed):
    load_precomputed = True
    # Initial experience
    if load_precomputed:
        max_exp_diff_period_working = np.loadtxt(
            path_dict["first_step_incomes"] + "max_exp_diff_period_working.txt",
            dtype=float,
        )
        max_exp_retirement = np.loadtxt(
            path_dict["first_step_incomes"] + "max_exp_retirement.txt", dtype=float
        )
    else:
        # max initial experience
        data_decision = pd.read_csv(path_dict["struct_est_sample"])
        max_exp_diff_period_working = (
            (data_decision["experience"] - data_decision["period"]).max().astype(float)
        )

        np.savetxt(
            path_dict["first_step_incomes"] + "max_exp_diff_period_working.txt",
            [max_exp_diff_period_working],
        )

        # Now calculate maximum experience bonus accross sexes and add them.
        # We can ensure with that, that the rescaled experience is always between  0 and 1.
        # First, age of fresh retirement (so age after retirement is chosen) is min_SRA + 1 and last age of fresh
        # retirement can be max_ret_age + 1
        ret_periods = (
            np.arange(specs["min_SRA"] + 1, specs["max_ret_age"] + 2)
            - specs["start_age"]
        )
        max_exp_across_periods = np.zeros(
            (specs["n_sexes"], specs["n_education_types"], len(ret_periods), 2),
            dtype=float,
        )

        for sex_var in range(specs["n_sexes"]):
            for edu_var in range(specs["n_education_types"]):
                for i, period in enumerate(ret_periods):
                    for health_id, health in enumerate([1, 2]):
                        max_exp_period = period + max_exp_diff_period_working

                        # The largest bonus can be obtained by being informed and working after the
                        # longest after the SRA.
                        new_exp = calc_experience_years_for_pension_adjustment(
                            period=period,
                            sex=sex_var,
                            experience_years=max_exp_period,
                            education=edu_var,
                            policy_state=0,
                            informed=1,
                            health=health,
                            model_specs=specs,
                        )
                        max_exp_across_periods[sex_var, edu_var, i, health_id] = new_exp

        # Get the maximum experience diff across periods
        max_exp_retirement = max_exp_across_periods.max()

        np.savetxt(
            path_dict["first_step_incomes"] + "max_exp_retirement.txt",
            [max_exp_retirement],
        )

    # Calculate the maximum experience one can have in a working state.
    max_exp_working = (
        specs["max_ret_age"] - specs["start_age"] + max_exp_diff_period_working
    )
    # Now span for each period the maximum experience for working periods.
    max_exps_period_working = jnp.arange(
        max_exp_diff_period_working, max_exp_working + 1
    )

    specs["max_exps_period_working"] = max_exps_period_working
    specs["max_exp_retirement"] = max_exp_retirement

    return specs


def add_very_long_insured_specs(specs, path_dict):
    """This function adds experience thresholds to be eligible for very long insured retirement path.
    We scale the 45 year of credited periods threshold by a sex specific multiplicator of experience.

    """
    exp_factor_for_credited_periods = pd.read_csv(
        path_dict["est_results"] + "credited_periods_estimates.csv", index_col=0
    )

    exp_thresholds = np.zeros((2,), dtype=float)
    for sex_var, sex_label in enumerate(["men", "women"]):
        exp_thresholds[sex_var] = (
            45
            / exp_factor_for_credited_periods.loc[f"experience_{sex_label}", "estimate"]
        )

    specs["experience_threshold_very_long_insured"] = jnp.asarray(exp_thresholds)
    return specs
