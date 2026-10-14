import jax.numpy as jnp

from model_code.pension_system.early_retirement_paths import (
    retirement_age_long_insured,
)
from model_code.taste_shocks import taste_shock_for_sexes


def choice_set_size_correction(
    period, lagged_choice, sex, policy_state, job_offer, health, params, model_specs
):
    """The taste-shock value of the choice set's size, subtracted from every flow utility.

    With Gumbel shocks the logsum exceeds the best alternative's value by up to
    sigma * log(J), purely because J alternatives are feasible. Taking
    sigma * log(J) off every choice in a state leaves the within-state choice
    probabilities untouched -- it is a common shift -- and nets the term out of the
    continuation value. Zero wherever the choice set is a singleton, so death,
    retirement and forced unemployment are unaffected.
    """
    n_choices = n_choices_in_state(
        period=period,
        lagged_choice=lagged_choice,
        sex=sex,
        policy_state=policy_state,
        job_offer=job_offer,
        health=health,
        model_specs=model_specs,
    )
    return taste_shock_for_sexes(sex=sex, params=params) * jnp.log(n_choices)


def n_choices_in_state(
    period, lagged_choice, sex, policy_state, job_offer, health, model_specs
):
    """The size of the choice set state_specific_choice_set builds, in closed form.

    The restrictions there drop choices from the full set independently, so counting
    them needs no enumeration: retirement and unemployment contribute one choice each
    when available, and a job offer contributes part-time and full-time, minus
    part-time for men. Kept arithmetic so it holds up under jit and vectorizes;
    tested against state_specific_choice_set over the whole state space.
    """
    age = period + model_specs["start_age"]
    SRA_pol_state = model_specs["min_SRA"] + policy_state * model_specs["SRA_grid_size"]

    retirement_available = (
        age >= retirement_age_long_insured(SRA_pol_state, model_specs)
    ) | (health == model_specs["disabled_health_var"])
    unemployment_available = age < SRA_pol_state
    # An offer buys full-time for everyone and part-time only for women.
    n_work_choices = job_offer * (1 + sex)

    n_choices = (
        jnp.asarray(retirement_available, dtype=int)
        + jnp.asarray(unemployment_available, dtype=int)
        + n_work_choices
    )

    forced_to_be_retired = (
        (health == model_specs["death_health_var"])
        | (lagged_choice == 0)
        | (age >= model_specs["max_ret_age"])
    )
    return jnp.where(forced_to_be_retired, 1, n_choices)
