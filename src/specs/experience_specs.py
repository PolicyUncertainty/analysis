import numpy as np
import pandas as pd
from model_code.state_space import calc_experience_years_for_pension_adjustment


def create_max_experience(path_dict, specs, load_precomputed):
    # Initial experience
    if load_precomputed:
        max_init_experience = int(
            np.loadtxt(path_dict["first_step_results"] + "max_init_exp.txt")
        )
    else:
        # max initial experience
        data_decision = pd.read_pickle(
            path_dict["intermediate_data"] + "structural_estimation_sample.pkl"
        )
        max_exp_first_period = (
            data_decision["experience"] - data_decision["period"]
        ).max()
        # Now calculate maximum experience bonus accross sexes and add them.
        # We can ensure with that, that the rescaled experience is always between  0 and 1.
        # Last age of fresh retirement can be max_ret_age + 1
        ages = np.arange(specs["min_SRA"], specs["max_ret_age"] + 2)
        max_exp_diff_to_periods = np.zeros(
            (specs["n_sexes"], specs["n_education_types"], len(ages)), dtype=float
        )

        for sex_var in range(specs["n_sexes"]):
            for edu_var in range(specs["n_education_types"]):
                for i, age in enumerate(ages):
                    period = age - specs["start_age"]
                    max_exp_period = period + max_exp_first_period
                    new_exp = calc_experience_years_for_pension_adjustment(
                        period=period,
                        sex=sex_var,
                        experience_years=max_exp_period,
                        education=edu_var,
                        policy_state=0,
                        informed=1,
                        options=specs,
                    )
                    max_exp_diff_to_periods[sex_var, edu_var, i] = new_exp - period

        max_exp_diff = max_exp_diff_to_periods.max()
        max_init_experience = np.maximum(max_exp_diff, max_exp_first_period)

        np.savetxt(
            path_dict["first_step_results"] + "max_init_exp.txt", [max_init_experience]
        )

    return max_init_experience
