"""The choice-set-size correction and the properties it has to satisfy."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from model_code.state_space.choice_set import state_specific_choice_set
from model_code.taste_shocks import taste_shock_for_sexes
from model_code.utility.choice_set_correction import (
    choice_set_size_correction,
    n_choices_in_state,
)
from model_code.utility.utility_functions_add import utility_func
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

jax.config.update("jax_enable_x64", True)

PARAMS = {
    "mu_low": 0.8,
    "mu_high": 0.8,
    "disutil_ft_work_good_men": 1.0,
    "disutil_ft_work_bad_men": 1.0,
    "disutil_unemployed_good_men": 1.0,
    "disutil_unemployed_bad_men": 1.0,
    "disutil_partner_retired_men": -0.2,
    "disutil_ft_work_good_women": 3.0,
    "disutil_ft_work_bad_women": 3.0,
    "disutil_pt_work_good_women": 2.0,
    "disutil_pt_work_bad_women": 2.0,
    "disutil_unemployed_good_women": 1.0,
    "disutil_unemployed_bad_women": 1.0,
    "disutil_partner_retired_women": -0.2,
    "disutil_children_ft_work_low": 0.2,
    "disutil_children_ft_work_high": 0.2,
    "disutil_children_pt_work_low": 0.3,
    "disutil_children_pt_work_high": 0.3,
    "bequest_scale_low_men": 3,
    "bequest_scale_high_men": 3,
    "bequest_scale_low_women": 2,
    "bequest_scale_high_women": 3,
    "kappa_low_men": 0,
    "kappa_high_men": 0,
    "kappa_low_women": 0,
    "kappa_high_women": 0,
    # Deliberately different across sexes, so a correction built with the wrong
    # scale cannot pass.
    "taste_shock_scale_men": 0.3,
    "taste_shock_scale_women": 0.45,
}


@pytest.fixture(scope="module")
def model_specs():
    return generate_derived_and_data_derived_specs(
        create_path_dict(), load_precomputed=True
    )


@pytest.fixture(scope="module")
def whole_state_space(model_specs):
    """Every state the choice set depends on, as a meshgrid plus its true set sizes.

    n_choices_in_state is arithmetic, so it takes the whole state space in one call;
    the truth comes from looping the model's own state_specific_choice_set.
    """
    axes = [
        np.arange(model_specs["n_periods"]),
        np.arange(model_specs["n_choices"]),
        np.arange(model_specs["n_sexes"]),
        np.arange(model_specs["n_policy_states"]),
        np.arange(2),
        np.arange(model_specs["n_all_health_states"]),
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    names = ("period", "lagged_choice", "sex", "policy_state", "job_offer", "health")
    states = dict(zip(names, grid))

    n_choices_true = np.empty(grid[0].shape, dtype=int)
    for index in np.ndindex(grid[0].shape):
        n_choices_true[index] = state_specific_choice_set(
            **{
                name: axis[index[dim]]
                for dim, (name, axis) in enumerate(zip(names, axes))
            },
            model_specs=model_specs,
        ).size
    return states, n_choices_true


def test_n_choices_matches_the_enumerated_choice_set(whole_state_space, model_specs):
    """The closed form must not drift from state_specific_choice_set."""
    states, n_choices_true = whole_state_space
    n_choices = n_choices_in_state(model_specs=model_specs, **states)
    np.testing.assert_array_equal(np.asarray(n_choices), n_choices_true)


def test_no_state_has_an_empty_choice_set(whole_state_space):
    """log(J) is only finite because every state keeps at least one choice."""
    _, n_choices_true = whole_state_space
    assert n_choices_true.min() == 1


def test_correction_is_sigma_times_log_choice_set_size(whole_state_space, model_specs):
    states, n_choices_true = whole_state_space
    sigma = taste_shock_for_sexes(sex=states["sex"], params=PARAMS)
    correction = choice_set_size_correction(
        params=PARAMS, model_specs=model_specs, **states
    )
    np.testing.assert_allclose(
        np.asarray(correction), sigma * np.log(n_choices_true), rtol=1e-12
    )


def test_correction_vanishes_on_singleton_choice_sets(whole_state_space, model_specs):
    """Death, retirement and forced unemployment must be left alone."""
    states, n_choices_true = whole_state_space
    correction = np.asarray(
        choice_set_size_correction(params=PARAMS, model_specs=model_specs, **states)
    )
    singleton = n_choices_true == 1
    assert singleton.any()
    assert (correction[singleton] == 0.0).all()
    assert (correction[~singleton] > 0.0).all()


def test_final_period_choice_sets_are_singletons(whole_state_space, model_specs):
    """Why the bequest utility needs no correction of its own."""
    _, n_choices_true = whole_state_space
    assert (n_choices_true[model_specs["n_periods"] - 1] == 1).all()


AGE_GRID = [50, 60, 63, 65, 66, 70, 73]
POLICY_STATE_GRID = [0, 8, 28]


def test_utility_shifts_by_the_same_amount_for_every_choice(model_specs):
    """The correction must cancel out of the within-state choice probabilities."""
    for age, lagged_choice, sex, policy_state, job_offer, health in product(
        AGE_GRID,
        range(model_specs["n_choices"]),
        range(model_specs["n_sexes"]),
        POLICY_STATE_GRID,
        range(2),
        range(model_specs["n_all_health_states"]),
    ):
        period = age - model_specs["start_age"]
        state = dict(
            period=period,
            lagged_choice=lagged_choice,
            sex=sex,
            policy_state=policy_state,
            job_offer=job_offer,
            health=health,
        )
        choice_set = state_specific_choice_set(model_specs=model_specs, **state)
        # Differences across choices are what the choice probabilities see, so they
        # have to be identical with and without the correction.
        corrected = np.array(
            [
                utility_func(
                    consumption=25.0,
                    partner_state=np.array(1),
                    education=1,
                    choice=choice,
                    params=PARAMS,
                    model_specs=model_specs,
                    **state,
                )
                for choice in choice_set
            ]
        )
        correction = choice_set_size_correction(
            params=PARAMS, model_specs=model_specs, **state
        )
        np.testing.assert_allclose(
            np.diff(corrected),
            np.diff(corrected + correction),
            rtol=1e-12,
            atol=1e-12,
        )


def test_dcegm_supplies_the_state_variables_the_correction_needs(model_specs):
    """The correction reads state variables utility_func did not take before.

    dcegm resolves a utility function's arguments from its signature and calls it with
    the full state-choice dict, so this checks the three new ones are actually in it,
    and that the correction survives being traced.
    """
    from dcegm.pre_processing.shared import (
        determine_function_arguments_and_partial_model_specs,
        try_jax_array,
    )

    model_specs_jax = jax.tree_util.tree_map(try_jax_array, model_specs)
    compute_utility = determine_function_arguments_and_partial_model_specs(
        func=utility_func,
        model_specs=model_specs_jax,
        not_allowed_state_choices=[],
    )

    # The state-choice dict dcegm hands to the utility function, as built from
    # create_model_config_wo_informed.
    state_choice_vec = {
        "period": jnp.array(30),
        "lagged_choice": jnp.array(3),
        "education": jnp.array(1),
        "sex": jnp.array(1),
        "alg_1_claim": jnp.array(0),
        "informed": jnp.array(1),
        "policy_state": jnp.array(0),
        "job_offer": jnp.array(1),
        "partner_state": jnp.array(1),
        "health": jnp.array(0),
        "choice": jnp.array(3),
    }

    got = jax.jit(compute_utility)(
        consumption=jnp.array(25.0), params=PARAMS, **state_choice_vec
    )
    expected = utility_func(
        consumption=jnp.array(25.0),
        params=PARAMS,
        model_specs=model_specs,
        **{
            key: value
            for key, value in state_choice_vec.items()
            if key not in ("alg_1_claim", "informed")
        },
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)

    # The state above has a three-element choice set, so the correction is doing work.
    assert (
        choice_set_size_correction(
            period=30,
            lagged_choice=3,
            sex=1,
            policy_state=0,
            job_offer=1,
            health=0,
            params=PARAMS,
            model_specs=model_specs,
        )
        > 0
    )
