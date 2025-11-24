from itertools import product

import numpy as np
import pytest

from model_code.pension_system.early_retirement_paths import retirement_age_long_insured
from model_code.state_space.choice_set import (
    state_specific_choice_set,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

# tests of choice set
PERIOD_GRID = np.linspace(10, 30, 3)
LAGGED_CHOICE_SET_WORKING_LIFE = np.array([1, 2, 3])
JOB_OFFER_GRID = np.array([0, 1], dtype=int)
CHOICE_SET = np.array([0, 1, 2])
SEX_GRID = np.array([0, 1], dtype=int)
HEALTH_GRID = np.array([0, 2], dtype=int)


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "period, sex, lagged_choice, job_offer, health",
    list(
        product(
            PERIOD_GRID,
            SEX_GRID,
            LAGGED_CHOICE_SET_WORKING_LIFE,
            JOB_OFFER_GRID,
            HEALTH_GRID,
        )
    ),
)
def test_choice_set_under_63(
    period, sex, lagged_choice, job_offer, health, paths_and_specs
):
    path_dict, specs = paths_and_specs

    policy_state = 0
    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        policy_state=policy_state,
        sex=sex,
        health=health,
        job_offer=job_offer,
        model_specs=specs,
    )
    if health == 2:
        if job_offer == 1:
            if sex == 0:
                assert (choice_set == [0, 1, 3]).all()
            else:
                assert (choice_set == [0, 1, 2, 3]).all()
        else:
            assert (choice_set == [0, 1]).all()
    else:
        if job_offer == 1:
            if sex == 0:
                assert (choice_set == [1, 3]).all()
            else:
                assert (choice_set == [1, 2, 3]).all()
        else:
            assert (choice_set == [1]).all()


AGE_GRID = np.linspace(63, 73, 1)
FULL_CHOICE_SET = np.array([0, 1, 2, 3])
POLICY_GRID = np.linspace(10, 25, 1)
JOB_OFFER_GRID = np.array([0, 1], dtype=int)


@pytest.mark.parametrize(
    "age, sex, lagged_choice, policy_state, job_offer, health",
    list(
        product(
            AGE_GRID,
            SEX_GRID,
            FULL_CHOICE_SET,
            POLICY_GRID,
            JOB_OFFER_GRID,
            HEALTH_GRID,
        )
    ),
)
def test_choice_set_over_63_under_72(
    age, sex, lagged_choice, policy_state, job_offer, health, paths_and_specs
):
    # Health should not make a difference
    path_dict, specs = paths_and_specs

    choice_set = state_specific_choice_set(
        period=age - specs["start_age"],
        lagged_choice=lagged_choice,
        policy_state=policy_state,
        sex=sex,
        health=health,
        job_offer=job_offer,
        model_specs=specs,
    )
    SRA = specs["min_SRA"] + policy_state * specs["SRA_grid_size"]
    ret_age_long_insured = retirement_age_long_insured(SRA, specs)
    if lagged_choice == 0:
        assert (choice_set == [0]).all()
    else:
        if age < ret_age_long_insured:
            # Not figures enough to retire. Check if job is offered
            if job_offer == 1:
                if sex == 0:
                    if health == 2:
                        assert (choice_set == [0, 1, 3]).all()
                    else:
                        assert (choice_set == [1, 3]).all()
                else:
                    if health == 2:
                        assert (choice_set == [0, 1, 2, 3]).all()
                    else:
                        assert (choice_set == [1, 2, 3]).all()
            else:
                if health == 2:
                    assert (choice_set == [0, 1]).all()
                else:
                    assert (choice_set == [1]).all()
        else:
            # figures enough to retire. Check if job is offered
            if (age >= SRA) & (age < specs["max_ret_age"]):
                if job_offer == 1:
                    if sex == 0:
                        assert (choice_set == [0, 3]).all()
                    else:
                        assert (choice_set == [0, 2, 3]).all()
                else:
                    assert (choice_set == [0]).all()
            elif age < specs["max_ret_age"]:
                if job_offer == 1:
                    if sex == 0:
                        assert (choice_set == [0, 1, 3]).all()
                    else:
                        assert (choice_set == [0, 1, 2, 3]).all()
                else:
                    assert (choice_set == [0, 1]).all()
            else:
                assert (choice_set == [0]).all()


PERIOD_GRID = np.linspace(47, 55, 1)
LAGGED_CHOICE_SET = np.array([0, 1, 2, 3])
POLICY_GRID = np.linspace(0, 9, 1)


@pytest.mark.parametrize(
    "period, sex, lagged_choice, policy_state",
    list(product(PERIOD_GRID, SEX_GRID, LAGGED_CHOICE_SET, POLICY_GRID)),
)
def test_choice_set_over_72(period, sex, lagged_choice, policy_state, paths_and_specs):

    path_dict, specs = paths_and_specs

    choice_set = state_specific_choice_set(
        period=period,
        lagged_choice=lagged_choice,
        policy_state=policy_state,
        sex=sex,
        health=1,
        job_offer=0,
        model_specs=specs,
    )
    assert (choice_set == [0]).all()
