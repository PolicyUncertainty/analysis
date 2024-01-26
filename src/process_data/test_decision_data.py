import pytest
from steps.gather_decision_data import gather_decision_data

LOAD_SAVED_DATA = True
USER = "max"

import os
analysis_path = os.path.abspath(os.getcwd() + "/../../") + "/"

import sys
sys.path.insert(0, analysis_path + "src/process_data/")


# Define paths_dict as a fixture
@pytest.fixture
def paths_dict():
    if USER == "bruno":
        return {
            # SOEP Core and SOEP RV are saved locally
            "soep_c38": "C:/Users/bruno/papers/soep/soep38",
            "soep_rv": "C:/Users/bruno/papers/soep/soep_rv",
            "project_path": analysis_path,
        }
    elif USER == "max":
        return {
            "soep_c38": "/home/maxbl/Uni/pol_uncetainty/data/soep38",
            "soep_rv": "/home/maxbl/Uni/pol_uncetainty/data/soep_rv",
            "project_path": analysis_path,
        }


# Define options as a fixture
@pytest.fixture
def options():
    min_SRA = 65
    min_ret_age = min_SRA - 4
    max_ret_age = 72
    exp_cap = 40  # maximum number of periods of exp accumulation
    start_year = 2010  # start year of estimation sample
    end_year = 2021  # end year of estimation sample
    return {
    # Set options for estimation of policy expectation process parameters
    # limits for truncation of the normal distribution
    "lower_limit": 66.5,
    "upper_limit": 80,
    # points at which the CDF is evaluated from survey data
    "first_cdf_point": 67.5,
    "second_cdf_point": 68.5,
    # cohorts for which process parameters are estimated
    "min_birth_year": 1947,
    "max_birth_year": 2000,
    # lowest policy state
    "min_policy_state": 65,
    "start_age": 25,
    "min_ret_age": min_ret_age,
    "max_ret_age": max_ret_age,
    # Set options for estimation of wage equation parameters
    "start_year": start_year,
    "end_year": end_year,
    "exp_cap": exp_cap,
    "wage_dist_truncation_percentiles": [0.01, 0.99],
}


# define policy_step_size as a fixture
@pytest.fixture
def policy_step_size():
    policy_step_size = 0.04478741131783991
    return policy_step_size


# These function tests the decision data for consistency (cf. model state space sparsity condition).


def test_decision_data_no_missing_values(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that there are no missing values for any states"""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        policy_step_size,
        load_data=load_data,
    )
    assert dec_dat["choice"].isna().sum() == 0
    assert dec_dat["period"].isna().sum() == 0
    assert dec_dat["lagged_choice"].isna().sum() == 0
    assert dec_dat["policy_state"].isna().sum() == 0
    # assert dec_dat["retirement_age_id"].isna().sum() == 0  //
    # this is the only exception because it is irrelevant for matched decisions
    assert dec_dat["experience"].isna().sum() == 0


def test_decision_data_no_ret_before_min_ret_age(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that nobody is retired before min_ret_age"""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        policy_step_size,
        load_data=load_data,
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 2, "period"].min() + options["start_age"]
        >= options["min_ret_age"]
    )


def test_decision_data_no_work_after_max_ret_age(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that there are no working after max_ret_age"""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        policy_step_size,
        load_data=load_data,
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 1, "period"].max() + options["start_age"]
        <= options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 0, "period"].max() + options["start_age"]
        <= options["max_ret_age"]
    )


def test_decision_data_exp_cap(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that experience is smaller or equal to age and exp_cap"""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        policy_step_size,
        load_data=load_data,
    )
    assert dec_dat["experience"].max() <= options["exp_cap"]
    assert dec_dat["experience"].max() <= dec_dat["period"].max()


def test_decision_data_retirement_is_absorbing(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that retirement is absorbing"""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        policy_step_size,
        load_data=load_data,
    )
    assert dec_dat.loc[dec_dat["lagged_choice"] == 2, "choice"].unique() == 2
