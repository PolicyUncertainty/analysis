import jax.numpy as jnp
import numpy as np
import pandas as pd

from model_code.pension_system.experience_stock import (
    calc_experience_years_for_pension_adjustment,
)


def create_max_experience(path_dict, specs, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_exp_diffs_per_period = np.loadtxt(
            path_dict["first_step_results"] + "max_exp_diffs_per_period.txt"
        )
    else:
        # max initial experience
        data_decision = pd.read_csv(path_dict["struct_est_sample"])
        max_exp_diff_data = (
            data_decision["experience"] - data_decision["period"]
        ).max()

        max_exp_diffs_per_period = np.zeros(specs["n_periods"], dtype=float)

        # Assign up to min_SRA period the maximum experience difference from the data
        min_SRA_period = specs["min_SRA"] - specs["start_age"]
        max_exp_diffs_per_period[: min_SRA_period + 1] = max_exp_diff_data

        # Now calculate maximum experience bonus accross sexes and add them.
        # We can ensure with that, that the rescaled experience is always between  0 and 1.
        # First, age of fresh retirement is min_SRA + 1 and last age of fresh retirement
        # can be max_ret_age + 1
        ret_periods = (
            np.arange(specs["min_SRA"] + 1, specs["max_ret_age"] + 2)
            - specs["start_age"]
        )
        max_exp_diff_to_periods = np.zeros(
            (specs["n_sexes"], specs["n_education_types"], len(ret_periods), 2),
            dtype=float,
        )

        for sex_var in range(specs["n_sexes"]):
            for edu_var in range(specs["n_education_types"]):
                for i, period in enumerate(ret_periods):
                    for health_id, health in enumerate([1, 2]):
                        max_exp_period = period + max_exp_diff_data

                        # The largest bonus can be obtained by beiing informed and working after the
                        # longest after the SRA.
                        new_exp = calc_experience_years_for_pension_adjustment(
                            period=period,
                            sex=sex_var,
                            experience_years=max_exp_period,
                            education=edu_var,
                            policy_state=0,
                            informed=1,
                            health=health,
                            options=specs,
                        )
                        max_exp_diff_to_periods[sex_var, edu_var, i, health_id] = (
                            new_exp - period
                        )

        # Get the maximum experience diff across periods
        max_across_ret_periods = max_exp_diff_to_periods.max()
        # Assign the maximum experience diff to the respective periods
        # max_exp_diffs_per_period[ret_periods] = max_across_ret_periods
        #
        # # Get the maximum over all and assign to the rest of the periods,
        # # always minus 1
        # max_exp_diff = max_exp_diff_to_periods.max()
        # last_periods = np.arange(ret_periods[-1] + 1, specs["n_periods"])
        # exp_diff_reduction = last_periods - ret_periods[-1]
        # max_exp_diffs_per_period[last_periods] = max_exp_diff - exp_diff_reduction
        total_max = np.maximum(max_across_ret_periods, max_exp_diff_data)
        max_exp_diffs_per_period[:] = total_max

        np.savetxt(
            path_dict["first_step_results"] + "max_exp_diffs_per_period.txt",
            max_exp_diffs_per_period,
        )

    return jnp.asarray(max_exp_diffs_per_period)


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
