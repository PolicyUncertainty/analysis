import jax.numpy as jnp


def health_transition(sex, health, education, period, params, options):
    trans_mat = options["health_trans_mat"]
    prob_vector = trans_mat[sex, education, period, health, :]

    not_disabled = health != options["disabled_health_var"]
    return prob_vector


def calc_disability_probability(params, education, period):
    exp_value = jnp.exp(
        params["disability_logit_const"]
        + params["disability_logit_period"] * period
        + params["disability_logit_period"] * education
    )
    prob = exp_value / (1 + exp_value)
    return prob
