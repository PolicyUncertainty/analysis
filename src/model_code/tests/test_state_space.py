
import pytest
import numpy as np
from itertools import product

from model_code.state_space import (
    sparsity_condition,
    update_state_space,
    state_specific_choice_set,
    apply_retirement_constraint_for_SRA,
)


# tests of update state space
PERIOD_GRID = np.linspace(10, 30, 3)
LAGGED_CHOICE_SET = np.array([0, 1, 2])


@pytest.mark.parametrize(
    "period, lagged_choice",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET)),
)
def test_period_and_lagged_choice_update(period, lagged_choice):
    options = {
        "start_age": 25,
        "min_ret_age": 65,
        "exp_cap": 40,
    }
    choice = 0
    retirement_age_id = 0
    experience = 20
    next_state = update_state_space(
        period, choice, lagged_choice, retirement_age_id, experience, options
    )
    assert next_state["period"] == period + 1
    assert next_state["lagged_choice"] == choice


# tests of update state space
PERIOD_GRID = np.linspace(35, 45, 1)
LAGGED_CHOICE_SET = np.array([0, 1, 2])


@pytest.mark.parametrize(
    "period, lagged_choice",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET)),
)
def test_retirement_age_update(period, lagged_choice):
    """Test that the retirement age is updated correctly for people who choose to
    retire."""
    options = {
        "start_age": 25,
        "min_ret_age": 65,
        "exp_cap": 40,
    }
    choice = 2
    retirement_age_id = 0
    experience = 20
    next_state = update_state_space(
        period, choice, lagged_choice, retirement_age_id, experience, options
    )
    age = period + options["start_age"]
    if lagged_choice != 2:
        assert next_state["retirement_age_id"] == age - options["min_ret_age"]
    else:
        assert next_state["retirement_age_id"] == retirement_age_id


PERIOD_GRID = np.linspace(35, 45, 1)
CHOICE_SET = np.array([0, 1, 2])
EXPERIENCE_GRID = np.linspace(35, 45, 2)


@pytest.mark.parametrize(
    "period, choice, experience",
    list(product(PERIOD_GRID, CHOICE_SET, EXPERIENCE_GRID)),
)
def test_experience_update(period, choice, experience):
    options = {
        "start_age": 25,
        "min_ret_age": 65,
        "exp_cap": 40,
    }
    period = 35
    choice = 1
    lagged_choice = 1
    retirement_age_id = 0
    next_state = update_state_space(
        period, choice, lagged_choice, retirement_age_id, experience, options
    )
    if choice == 1:
        if experience < options["exp_cap"]:
            assert next_state["experience"] == experience + 1
    else:
        assert next_state["experience"] == experience


# tests of choice set

PERIOD_GRID = np.linspace(10, 30, 3)
LAGGED_CHOICE_SET = np.array([0, 1])
JOB_OFFER_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, lagged_choice, job_offer",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET, JOB_OFFER_GRID)),
)
def test_choice_set_under_63(period, lagged_choice, job_offer):
    options = {
        "start_age": 25,
        "min_SRA": 67,
        "SRA_grid_size": 0.25,
        "ret_years_before_SRA": 2,
    }
    policy_state = 0
    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        policy_state=policy_state,
        job_offer=job_offer,
        options=options,
    )
    if job_offer == 1:
        assert (choice_set == [0, 1]).all()
    else:
        assert (choice_set == [0]).all()


PERIOD_GRID = np.linspace(10, 30, 3)
LAGGED_CHOICE_SET = np.array([0, 1])
POLICY_GRID = np.linspace(0, 9, 1)
JOB_OFFER_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, lagged_choice, policy_state, job_offer",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET, POLICY_GRID, JOB_OFFER_GRID)),
)
def test_choice_set_over_63_under_72(period, lagged_choice, policy_state, job_offer):
    options = {
        "start_age": 20,
        "min_SRA": 67,
        "SRA_grid_size": 0.25,
        "max_ret_age": 72,
        "ret_years_before_SRA": 4,
    }

    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        policy_state=policy_state,
        job_offer=job_offer,
        options=options,
    )
    age = period + options["start_age"]
    SRA = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA, options)
    if age < min_ret_age_pol_state:
        # Not old enough to retire. Check if job is offered
        if job_offer == 1:
            assert (choice_set == [0, 1]).all()
        else:
            assert (choice_set == [0]).all()
    else:
        # old enough to retire. Check if job is offered
        if job_offer == 1:
            assert (choice_set == [0, 1, 2]).all()
        else:
            assert (choice_set == [0, 2]).all()


PERIOD_GRID = np.linspace(47, 55, 1)
LAGGED_CHOICE_SET = np.array([0, 1, 2])
POLICY_GRID = np.linspace(0, 9, 1)


@pytest.mark.parametrize(
    "period, lagged_choice, policy_state",
    list(product(PERIOD_GRID, LAGGED_CHOICE_SET, POLICY_GRID)),
)
def test_choice_set_over_72(period, lagged_choice, policy_state):
    options = {
        "start_age": 25,
        "min_SRA": 67,
        "SRA_grid_size": 0.25,
        "max_ret_age": 72,
        "ret_years_before_SRA": 4,
    }

    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        policy_state=policy_state,
        job_offer=0,
        options=options,
    )
    assert (choice_set == [2]).all()
