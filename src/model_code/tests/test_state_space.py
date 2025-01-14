from itertools import product

import numpy as np
import pytest
from model_code.state_space import apply_retirement_constraint_for_SRA
from model_code.state_space import state_specific_choice_set


# tests of choice set
PERIOD_GRID = np.linspace(10, 30, 3)
LAGGED_CHOICE_SET_WORKING_LIFE = np.array([1, 2, 3])
JOB_OFFER_GRID = np.array([0, 1], dtype=int)
CHOICE_SET = np.array([0, 1, 2])
SEX_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, sex, lagged_choice, job_offer",
    list(
        product(PERIOD_GRID, SEX_GRID, LAGGED_CHOICE_SET_WORKING_LIFE, JOB_OFFER_GRID)
    ),
)
def test_choice_set_under_63(period, sex, lagged_choice, job_offer):
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
        sex=sex,
        health=1,
        job_offer=job_offer,
        options=options,
    )
    if job_offer == 1:
        if sex == 0:
            assert (choice_set == [1, 3]).all()
        else:
            assert (choice_set == [1, 2, 3]).all()
    else:
        assert (choice_set == [1]).all()


PERIOD_GRID = np.linspace(25, 42, 3)
FULL_CHOICE_SET = np.array([0, 1, 2, 3])
POLICY_GRID = np.linspace(0, 9, 1)
JOB_OFFER_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "period, sex, lagged_choice, policy_state, job_offer",
    list(product(PERIOD_GRID, SEX_GRID, FULL_CHOICE_SET, POLICY_GRID, JOB_OFFER_GRID)),
)
def test_choice_set_over_63_under_72(
    period, sex, lagged_choice, policy_state, job_offer
):
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
        sex=sex,
        health=1,
        job_offer=job_offer,
        options=options,
    )
    age = period + options["start_age"]
    SRA = options["min_SRA"] + policy_state * options["SRA_grid_size"]
    min_ret_age_pol_state = apply_retirement_constraint_for_SRA(SRA, options)
    if lagged_choice == 0:
        assert (choice_set == [0]).all()
    else:
        if age < min_ret_age_pol_state:
            # Not old enough to retire. Check if job is offered
            if job_offer == 1:
                if sex == 0:
                    assert (choice_set == [1, 3]).all()
                else:
                    assert (choice_set == [1, 2, 3]).all()
            else:
                assert (choice_set == [1]).all()
        else:
            # old enough to retire. Check if job is offered
            if job_offer == 1:
                if sex == 0:
                    if SRA <= age:
                        assert (choice_set == [0, 3]).all()
                    else:
                        assert (choice_set == [0, 1, 3]).all()
                else:
                    if SRA <= age:
                        assert (choice_set == [0, 2, 3]).all()
                    else:
                        assert (choice_set == [0, 1, 2, 3]).all()
            else:
                if SRA <= age:
                    assert (choice_set == [0]).all()
                else:
                    assert (choice_set == [0, 1]).all()


PERIOD_GRID = np.linspace(47, 55, 1)
LAGGED_CHOICE_SET = np.array([0, 1, 2, 3])
POLICY_GRID = np.linspace(0, 9, 1)


@pytest.mark.parametrize(
    "period, sex, lagged_choice, policy_state",
    list(product(PERIOD_GRID, SEX_GRID, LAGGED_CHOICE_SET, POLICY_GRID)),
)
def test_choice_set_over_72(period, sex, lagged_choice, policy_state):
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
        sex=sex,
        health=1,
        job_offer=0,
        options=options,
    )
    assert (choice_set == [0]).all()
