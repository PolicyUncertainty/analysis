import jax
import jax.numpy as jnp


def health_transition(
    sex, health, education, period, params, choice, lagged_choice, model_specs
):
    trans_mat = model_specs["health_trans_mat"]
    prob_vector = trans_mat[sex, education, period, health, :]

    # Generate conditional probability of being disabled.
    # Start by reading out health state values:
    bad_health_var = model_specs["bad_health_var"]
    disabled_health_var = model_specs["disabled_health_var"]

    # If the agent is disabled the conditional probability of being in bad health
    # and remaining in disability is 1, i.e. there is no transition from disability
    # to bad health. Only to good.
    # If you are dead, it does not matter, as you are dead(absorbing).
    cond_prob_disabled = calc_disability_probability(
        params=params,
        sex=sex,
        education=education,
        period=period,
        model_specs=model_specs,
    )
    bad_health_prob = prob_vector[bad_health_var] * (1 - cond_prob_disabled)
    disabled_health_prob = prob_vector[bad_health_var] * cond_prob_disabled

    # Now set the probabilities in the vector(jax.numpy style)
    prob_vector = prob_vector.at[disabled_health_var].set(disabled_health_prob)
    prob_vector = prob_vector.at[bad_health_var].set(bad_health_prob)

    # We need to ensure, that if the agent is disabled and chooses retirement, she has to stay one period disabled
    # and cannot transition to good/bad health. The agent can still die.
    fresh_disability_pension = (
        (lagged_choice != 0) & (health == disabled_health_var) & (choice == 0)
    )
    fresh_disability_pension_prob = construct_fresh_disability_pension_prob_vector(
        prob_vector=prob_vector, model_specs=model_specs
    )
    prob_vector = jax.lax.select(
        fresh_disability_pension,
        on_true=fresh_disability_pension_prob,
        on_false=prob_vector,
    )

    return prob_vector


def calc_disability_probability(params, sex, education, period, model_specs):
    age = model_specs["start_age"] + period

    # Calculate exp value for men and women
    exp_value_men = jnp.exp(
        params["disability_logit_const_men"]
        + params["disability_logit_age_men"] * age
        + params["disability_logit_high_educ_men"] * education
    )
    exp_value_women = jnp.exp(
        params["disability_logit_const_women"]
        + params["disability_logit_age_women"] * age
        + params["disability_logit_high_educ_women"] * education
    )
    # Now select based on sex state
    is_men = sex == 0
    exp_value = jax.lax.select(is_men, on_true=exp_value_men, on_false=exp_value_women)
    prob = exp_value / (1 + exp_value)

    return prob


def construct_fresh_disability_pension_prob_vector(prob_vector, model_specs):
    """We need to construct the probability vector for disabled individuals which choose retirement. If, they stay alive
    they need to stay with certainty in the disabled state, to be able to adapt their experience stock correctly in the
    next period.
    """
    death_health_var = model_specs["death_health_var"]
    disabled_health_var = model_specs["disabled_health_var"]
    survival_prob = 1 - prob_vector[death_health_var]
    fresh_disability_pension_prob = jnp.zeros_like(prob_vector)
    fresh_disability_pension_prob = fresh_disability_pension_prob.at[
        disabled_health_var
    ].set(survival_prob)
    fresh_disability_pension_prob = fresh_disability_pension_prob.at[
        death_health_var
    ].set(prob_vector[death_health_var])
    return fresh_disability_pension_prob
