import numpy as np
import pandas as pd
from dcegm.asset_correction import adjust_observed_assets
from dcegm.likelihood import (
    create_choice_prob_function,
)

from first_step_estimation.estimation.partner_wage_estimation import (
    create_deflate_factor,
)
from model_code.state_space.experience import scale_experience_years
from model_code.unobserved_state_weighting import create_unobserved_state_specs
from process_data.structural_sample_scripts.create_structural_est_sample import (
    CORE_TYPE_DICT,
)


def calc_choice_probs_for_df(df, params, model_solved):
    states_dict = create_states_dict(df=df, model_class=model_solved)

    model_specs = model_solved.model_specs

    unobserved_state_specs = create_unobserved_state_specs(df)

    df["prob_choice"] = np.nan
    for choice in range(model_specs["n_choices"]):
        choice_vals = np.ones_like(df["choice"].values) * choice

        choice_probs_observations = choice_probs_for_choice_vals(
            choice_vals=choice_vals,
            states_dict=states_dict,
            model_solved=model_solved,
            unobserved_state_specs=unobserved_state_specs,
            params=params,
            use_probability_of_observed_states=False,
        )

        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        df[f"choice_{choice}"] = choice_probs_observations
        df.loc[df["choice"] == choice, "prob_choice"] = choice_probs_observations[
            df["choice"] == choice
        ]
    return df


def calc_unobserved_choice_probs_for_df(df, params, model_solved):
    states_dict = create_states_dict(df=df, model_class=model_solved)

    model_specs = model_solved.model_specs

    unobserved_state_specs = create_unobserved_state_specs(df)

    # Create names using the pattern: variable_value_variable_value_variable_value
    state_names = {
        (0, 1, 0): "job_offer_0_health_1_informed_0",
        (0, 1, 1): "job_offer_0_health_1_informed_1",
        (0, 2, 0): "job_offer_0_health_2_informed_0",
        (0, 2, 1): "job_offer_0_health_2_informed_1",
        (1, 1, 0): "job_offer_1_health_1_informed_0",
        (1, 1, 1): "job_offer_1_health_1_informed_1",
        (1, 2, 0): "job_offer_1_health_2_informed_0",
        (1, 2, 1): "job_offer_1_health_2_informed_1",
    }
    # Iterate over dictionary and select corresponding job_offer, health, informed
    for key, state_name in state_names.items():
        job_offer_val, health_val, informed_val = key
        states_dict["job_offer"][
            ~unobserved_state_specs["observed_bools_states"]["job_offer"]
        ] = job_offer_val
        states_dict["health"][
            ~unobserved_state_specs["observed_bools_states"]["health"]
        ] = health_val
        states_dict["informed"][
            ~unobserved_state_specs["observed_bools_states"]["informed"]
        ] = informed_val

        for choice in range(model_specs["n_choices"]):
            choice_vals = np.ones_like(df["choice"].values) * choice

            choice_probs_observations = choice_probs_for_choice_vals(
                choice_vals=choice_vals,
                states_dict=states_dict,
                model_solved=model_solved,
                params=params,
                use_probability_of_observed_states=False,
            )

            choice_probs_observations = np.nan_to_num(
                choice_probs_observations, nan=0.0
            )
            df[f"{state_name}_choice_{choice}"] = choice_probs_observations

    return df


def choice_probs_for_choice_vals(
    choice_vals,
    states_dict,
    model_solved,
    params,
    unobserved_state_specs=None,
    use_probability_of_observed_states=False,
):
    choice_prob_func = create_choice_prob_function(
        model_structure=model_solved.model_structure,
        model_specs=model_solved.model_specs,
        model_funcs=model_solved.model_funcs,
        model_config=model_solved.model_config,
        observed_states=states_dict,
        observed_choices=choice_vals,
        unobserved_state_specs=unobserved_state_specs,
        use_probability_of_observed_states=use_probability_of_observed_states,
        return_weight_func=False,
    )

    choice_probs_observations = choice_prob_func(
        value_in=model_solved.value,
        endog_grid_in=model_solved.endog_grid,
        params_in=params,
    )
    return choice_probs_observations


def load_scale_and_correct_data(path_dict, model_class):
    # Load data
    data_decision = pd.read_csv(path_dict["struct_est_sample"])
    data_decision = data_decision.astype(CORE_TYPE_DICT)

    model_specs = model_class.model_specs

    data_decision["age"] = data_decision["period"] + model_specs["start_age"]

    data_decision["experience_years"] = data_decision["experience"].values

    # Transform experience
    data_decision["experience"] = scale_experience_years(
        experience_years=data_decision["experience"].values,
        period=data_decision["period"].values,
        is_retired=data_decision["lagged_choice"].values == 0,
        model_specs=model_specs,
    )

    # We can adjust wealth outside, as it does not depend on estimated parameters. We assign wealth as assets at the
    # beginning of the period. It will be overwritten in the functions below.
    data_decision["assets_begin_of_period"] = (
        data_decision["wealth"] / model_specs["wealth_unit"]
    )
    states_dict = create_states_dict(df=data_decision, model_class=model_class)

    out = adjust_observed_assets(
        observed_states_dict=states_dict,
        params={},
        model_class=model_class,
        aux_outs=True,
    )
    # data_decision = create_deflate_factor(path_dict, model_specs, data_decision)
    # data_decision["gross_retirement_income"] = (
    #     out[1]["gross_retirement_income"]
    #     * model_class.model_specs["wealth_unit"]
    #     / data_decision["deflate_factor"]
    # )
    #
    # import matplotlib.pyplot as plt
    #
    # data_decision.groupby("age")["gross_retirement_income"].mean().plot()
    # data_decision.groupby("age")["last_year_pension"].mean().plot()
    # plt.show()
    data_decision["assets_begin_of_period"] = out[0]

    data_decision = create_informed_probability(data_decision, model_specs)
    return data_decision


def create_states_dict(df, model_class):
    """
    Create a dictionary of states from a DataFrame.
    """
    states_dict = {
        name: df[name].values.copy()
        for name in model_class.model_structure["discrete_states_names"]
    }
    states_dict["experience"] = df["experience"].values
    states_dict["assets_begin_of_period"] = df["assets_begin_of_period"].values
    return states_dict


def create_informed_probability(df, specs):
    """Predict informed probability by education and age using hazard rate model."""
    df = df.copy()
    probs = np.zeros(len(df))

    for edu_val, edu_label in enumerate(specs["education_labels"]):
        mask = df["education"] == edu_val
        if not mask.any():
            continue

        ages = df.loc[mask, "age"].values
        initial_informed_share = specs["initial_informed_share"][edu_val]
        hazard_rate = specs["informed_hazard_rate"][edu_val]

        # Calculate periods since initial age (minimum age in belief data, i.e., 17)
        periods = ages - 17

        # Calculate uninformed probability
        initial_uninformed_share = 1 - initial_informed_share
        prob_uninformed = initial_uninformed_share * (1 - hazard_rate) ** periods
        prob_informed = 1 - prob_uninformed

        probs[mask] = prob_informed

    df["probability_informed"] = probs
    return df


# from model_code.state_space.experience import calculate_pension_size_for_experience_and_type

def correct_wealth_to_include_non_pension_retirement_income(
    df,
    model_specs,
):
    df = df.copy()
    # Correct wealth to include non-pension retirement income
    # We assume that non-pension retirement income is 30% of pension wealth.
    # Thus, we compute for every individual their pension (function of experience, education, sex) and compute the
    # net present value assuming they expect to retire at the SRA and live until life expectancy (from specs).

    df["current_pension"] = calculate_pension_size_for_experience_and_type(df, model_specs)
    life_expectancy = model_specs["life_expectancy"][df["sex"].values][df["education"].values]
    df["years_between_SRA_and_death"] = (
        life_expectancy - df["policy_state_value"].values
    )
    df["years_until_SRA"] = df["policy_state_value"].values - df["age"].values

    # calculate perpetuity of getting pension income after SRA, then subtract perpetuity after death
    discount_factor = 1 / (1 + model_specs["interest_rate"])
    annuity_factor = 1 / (1 - discount_factor)
    df["pension_annuity_value"] = (
        df["current_pension"] * annuity_factor - df["current_pension"] * annuity_factor * discount_factor ** df["years_between_SRA_and_death"]
    )
    df["pension_wealth"] = df["pension_annuity_value"] * discount_factor ** df["years_until_SRA"] 
    df["wealth_correction"] = 0.3 * df["pension_wealth"]
    df["assets_begin_of_period"] += df["wealth_correction"] / model_specs["wealth_unit"]
    return df