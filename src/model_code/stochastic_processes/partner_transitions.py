import jax
import jax.numpy as jnp


def partner_transition(
    model_specs, period, policy_state, education, sex, partner_state, lagged_choice
):
    """Compute transition matrices for a particular age and sra."""

    sra = model_specs["min_SRA"] + policy_state * model_specs["SRA_grid_size"]
    age = (model_specs["start_age"] + period).astype(float)
    own_retired = lagged_choice == 0

    val_0_1 = exp_val_single_to_working_age(
        model_specs=model_specs,
        age=age,
        sex=sex,
        education=education,
    )
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

    # Set val_1_2 to nan if age <= 40
    val_1_2 = jax.lax.select(
        age > 40, on_true=val_1_2, on_false=jnp.ones_like(val_1_2) * jnp.nan
    )

    # Row for single state: can transition to working age or stay single
    # val_0_2 would be nan (can't go directly from single to retired)
    single_row = jnp.array([0.0, val_0_1, jnp.nan])
    max_val_single = jnp.nanmax(single_row)
    exp_single_row = jnp.exp(single_row - max_val_single)
    single_row_normalized = exp_single_row / jnp.nansum(exp_single_row)
    single_row_normalized = jnp.nan_to_num(single_row_normalized, nan=0.0)

    # Row for working age state: two-step normalization
    # Step 1: Probability to single vs staying partnered
    step1_vals = jnp.array([0.0, val_1_1])
    max_val_step1 = jnp.nanmax(step1_vals)
    exp_step1 = jnp.exp(step1_vals - max_val_step1)
    step1_probs = exp_step1 / jnp.sum(exp_step1)
    prob_to_single = step1_probs[0]
    prob_to_partnered = step1_probs[1]

    # Step 2: Split partnered probability between working age and retirement
    partnered_vals = jnp.array([val_1_1, val_1_2])
    max_val_partnered = jnp.nanmax(partnered_vals)
    exp_partnered = jnp.exp(partnered_vals - max_val_partnered)
    partnered_probs = exp_partnered / jnp.nansum(exp_partnered)
    partnered_probs = jnp.nan_to_num(partnered_probs, nan=0.0)

    working_row_normalized = jnp.array(
        [
            prob_to_single,
            prob_to_partnered * partnered_probs[0],  # to working_age
            prob_to_partnered * partnered_probs[1],  # to retirement
        ]
    )

    # Row for retired state: absorbing
    retired_row_normalized = jnp.array([0.0, 0.0, 1.0])

    # Select appropriate row based on current partner state
    trans_row = (
        single * single_row_normalized
        + working * working_row_normalized
        + retired * retired_row_normalized
    )

    # Construct absorbing states for age 75+
    absorbing_ret_row = jnp.array([0.0, 0.0, 1.0])
    absorbing_single_row = jnp.array([1.0, 0.0, 0.0])
    above_75_row = single * absorbing_single_row + (1 - single) * absorbing_ret_row

    # Apply age-based logic
    above_75 = age >= 75
    final_row = (1 - above_75) * trans_row + above_75 * above_75_row

    # Apply own retirement status logic
    working_age_prob = final_row[1]
    final_row = final_row.at[2].set(final_row[2] + working_age_prob * own_retired)
    final_row = final_row.at[1].set(working_age_prob * (1 - own_retired))

    return final_row


def exp_val_single_to_retirement(model_specs, age, sex, education):
    val = (
        model_specs["const_single_to_retirement"][sex, education]
        + model_specs["age_single_to_retirement"][sex, education] * age
        + model_specs["age_squared_single_to_retirement"][sex, education]
        * (age**2 / 100)
        + model_specs["age_cubic_single_to_retirement"][sex, education]
        * (age**3 / 100_000)
    )
    return val


def exp_val_working_age_to_retirement(model_specs, age, sra, sex, education):
    val = (
        model_specs["const_working_age_to_retirement"][sex, education]
        + model_specs["age_working_age_to_retirement"][sex, education] * age
        + model_specs["age_squared_working_age_to_retirement"][sex, education]
        * (age**2 / 100)
        + model_specs["age_cubic_working_age_to_retirement"][sex, education]
        * (age**3 / 100_000)
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
        + model_specs["age_cubic_working_age_to_working_age"][sex, education]
        * (age**3 / 100_000)
    )
    return val


def exp_val_single_to_working_age(model_specs, age, sex, education):
    val = (
        model_specs["const_single_to_working_age"][sex, education]
        + model_specs["age_single_to_working_age"][sex, education] * age
        + model_specs["age_squared_single_to_working_age"][sex, education]
        * (age**2 / 100)
        + model_specs["age_cubic_single_to_working_age"][sex, education]
        * (age**3 / 100_000)
    )
    return val
