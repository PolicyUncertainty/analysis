import os
import sys

import pytest


LOAD_SAVED_DATA = True
USER = "max"

import yaml
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "src")

from process_data.gather_decision_data import gather_decision_data

# As we do not keep our data in github these tests can only be run locally
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


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
    return yaml.safe_load(open(analysis_path + "src/spec.yaml"))


# define policy_step_size as a fixture
@pytest.fixture
def policy_step_size():
    policy_step_size = 0.04478741131783991
    return policy_step_size


# These function tests the decision data for consistency (cf. model state space sparsity condition).


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_no_missing_values(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that there are no missing values for any states."""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        load_data=load_data,
    )
    assert dec_dat["choice"].isna().sum() == 0
    assert dec_dat["period"].isna().sum() == 0
    assert dec_dat["lagged_choice"].isna().sum() == 0
    assert dec_dat["policy_state"].isna().sum() == 0
    # assert dec_dat["retirement_age_id"].isna().sum() == 0  //
    # this is the only exception because it is irrelevant for matched decisions
    assert dec_dat["experience"].isna().sum() == 0
    assert dec_dat["wealth"].isna().sum() == 0


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_no_ret_before_min_ret_age(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that nobody is retired before min_ret_age."""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        load_data=load_data,
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 2, "period"].min() + options["start_age"]
        >= options["min_ret_age"]
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_no_work_after_max_ret_age(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that there are no working after max_ret_age."""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
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


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_exp_cap(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that experience is smaller or equal to age and exp_cap."""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        load_data=load_data,
    )
    assert dec_dat["experience"].max() <= options["exp_cap"]
    assert dec_dat["experience"].max() <= dec_dat["period"].max()


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_retirement_is_absorbing(
    paths_dict, options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that retirement is absorbing."""
    dec_dat = gather_decision_data(
        paths_dict,
        options,
        load_data=load_data,
    )
    assert dec_dat.loc[dec_dat["lagged_choice"] == 2, "choice"].unique() == 2
