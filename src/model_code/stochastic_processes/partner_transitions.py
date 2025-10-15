import jax
import jax.numpy as jnp


def partner_transition(
    model_specs, period, policy_state, education, sex, partner_state
):
    """Compute transition matrices for a particular age and sra."""

    sra = model_specs["min_SRA"] + policy_state * model_specs["SRA_grid_size"]
    age = (model_specs["start_age"] + period).astype(float)

    val_0_1 = exp_val_single_to_working_age(
        model_specs=model_specs,
        age=age,
        sex=sex,
        education=education,
    )
    # val_0_2 = exp_val_single_to_retirement(
    #     model_specs=model_specs,
    #     age=age,
    #     sex=sex,
    #     education=education,
    # )
    val_1_1 = exp_val_working_age_to_working_age(
        model_specs=model_specs,
        age=age,
        sex=sex,
        education=education,
    )
    val_1_2 = exp_val_working_age_to_retirement(
        model_specs=model_specs,
        age=age,
        sex=sex,
        education=education,
        sra=sra,
    )
    single = partner_state == 0
    working = partner_state == 1
    retired = partner_state == 2

    val_1 = single * val_0_1 + working * val_1_1

    val_2 = jax.lax.select(
        (age > 40) & working, on_true=val_1_2, on_false=jnp.ones_like(val_1_2) * jnp.nan
    )

    prob_row = jnp.array([0.0, val_1, val_2])
    max_val = jnp.nanmax(prob_row)
    exp_row = jnp.exp(prob_row - max_val)
    trans_row = exp_row / jnp.nansum(exp_row)
    trans_row = jnp.nan_to_num(trans_row, nan=0.0)

    # Construct the absorbing rows for age 75+
    absorbing_ret_row = jnp.array([0.0, 0.0, 1.0])
    absorbing_single_row = jnp.array([1.0, 0.0, 0.0])
    above_75_row = single * absorbing_single_row + (1 - single) * absorbing_ret_row

    # Construct the row for before 75
    below_75_row = retired * absorbing_ret_row + (1 - retired) * trans_row

    # Now aggregate based on age
    above_75 = age >= 75
    final_row = (1 - above_75) * below_75_row + above_75 * above_75_row
    return final_row

    # Retirement is only possible after 50
    val_0_2 = jnp.where(age > 40, val_0_2, np.nan)
    val_1_2 = np.where(age > 40, val_1_2, np.nan)

    ones = np.ones_like(age)
    zeros = np.zeros_like(age)

    # Rescale for numerical stability. row by row
    first_row = np.array([zeros, val_0_1, val_0_2]).T
    max_values = np.nanmax(first_row, axis=1, keepdims=True)
    first_row = np.exp(first_row - max_values)

    # Second row
    second_row = np.array([zeros, val_1_1, val_1_2]).T
    max_values = np.nanmax(second_row, axis=1, keepdims=True)
    second_row = np.exp(second_row - max_values)

    third_row = np.array([zeros, zeros, ones]).T
    exp_vals = np.stack([first_row, second_row, third_row], axis=1)
    trans_mat = exp_vals / np.nansum(exp_vals, axis=-1, keepdims=True)
    # Convert to nans to zeros
    trans_mat = np.nan_to_num(trans_mat, nan=0.0)

    trans_mat_absorbing = np.zeros_like(trans_mat)
    trans_mat_absorbing[:, 0, 0] = 1.0
    trans_mat_absorbing[:, 1, 2] = 1.0
    trans_mat_absorbing[:, 2, 2] = 1.0
    above_75 = age >= 75
    trans_mat = (1 - above_75)[:, None, None] * trans_mat + above_75[
        :, None, None
    ] * trans_mat_absorbing
    return trans_mat


def exp_val_single_to_retirement(model_specs, age, sex, education):
    val = (
        model_specs["const_single_to_retirement"][sex, education]
        + model_specs["age_single_to_retirement"][sex, education] * age
        + model_specs["age_squared_single_to_retirement"][sex, education]
        * (age**2 / 100)
        # + params["age_cubic_single_to_retirement"] * (age**3 / 100_000)
    )
    return val


def exp_val_working_age_to_retirement(model_specs, age, sra, sex, education):
    val = (
        model_specs["const_working_age_to_retirement"][sex, education]
        + model_specs["age_working_age_to_retirement"][sex, education] * age
        + model_specs["age_squared_working_age_to_retirement"][sex, education]
        * (age**2 / 100)
        # + params["age_cubic_working_age_to_retirement"] * (age**3 / 100_000)
        + model_specs["SRA_age_diff_effect_working_age_to_retirement"][sex, education]
        * (sra - age)
        * (age > 50)
    )
    return val


def exp_val_working_age_to_working_age(model_specs, age, sex, education):
    val = (
        model_specs["const_working_age_to_working_age"][sex, education]
        + model_specs["age_working_age_to_working_age"][sex, education] * age
        + model_specs["age_squared_working_age_to_working_age"][sex, education]
        * (age**2 / 100)
        # + params["age_cubic_working_age_to_working_age"] * (age**3 / 100_000)
    )
    return val


def exp_val_single_to_working_age(model_specs, age, sex, education):
    val = (
        model_specs["const_single_to_working_age"][sex, education]
        + model_specs["age_single_to_working_age"][sex, education] * age
        + model_specs["SRA_age_below_40_single_to_working_age"][sex, education]
        * (age < 40)
        * age
        + model_specs["age_squared_single_to_working_age"][sex, education]
        * (age**2 / 100)
        # + params["age_cubic_single_to_working_age"] * (age**3 / 100_000)
    )
    return val
