import os

import pytest


LOAD_SAVED_DATA = True

# from process_data.derive_datasets import gather_decision_data
from process_data.structural_sample_scripts.create_structural_est_sample import (
    create_structural_est_sample,
)
from set_paths import create_path_dict

# As we do not keep our data in github these tests can only be run locally
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


# These function tests the decision data for consistency (cf. model state space sparsity condition).

from specs.derive_specs import generate_derived_and_data_derived_specs


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_no_missing_values(load_data=True):
    """This functions asserts that there are no missing values for any states."""
    paths_dict = create_path_dict()

    dec_dat = create_structural_est_sample(
        paths_dict,
        specs=None,
        load_data=load_data,
    )
    assert dec_dat["choice"].isna().sum() == 0
    assert dec_dat["period"].isna().sum() == 0
    assert dec_dat["lagged_choice"].isna().sum() == 0
    assert dec_dat["policy_state"].isna().sum() == 0
    # this is the only exception because it is irrelevant for matched decisions
    assert dec_dat["experience"].isna().sum() == 0
    assert dec_dat["wealth"].isna().sum() == 0


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_no_ret_before_min_ret_age(load_data=LOAD_SAVED_DATA):
    """This functions asserts that nobody is retired before min_ret_age."""
    paths_dict = create_path_dict()
    options = generate_derived_and_data_derived_specs(paths_dict, load_precomputed=True)

    dec_dat = create_structural_est_sample(
        paths_dict,
        specs=None,
        load_data=load_data,
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 0, "period"].min() + options["start_age"]
        >= options["min_ret_age"]
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_no_work_after_max_ret_age(load_data=LOAD_SAVED_DATA):
    """This functions asserts that there are no working after max_ret_age."""
    paths_dict = create_path_dict()
    options = generate_derived_and_data_derived_specs(paths_dict, load_precomputed=True)

    dec_dat = create_structural_est_sample(
        paths_dict,
        specs=None,
        load_data=load_data,
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 1, "period"].max() + options["start_age"]
        < options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 2, "period"].max() + options["start_age"]
        < options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 3, "period"].max() + options["start_age"]
        < options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["lagged_choice"] == 1, "period"].max()
        + options["start_age"]
        <= options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["lagged_choice"] == 2, "period"].max()
        + options["start_age"]
        <= options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["lagged_choice"] == 2, "period"].max()
        + options["start_age"]
        <= options["max_ret_age"]
    )


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Test doesn't work in Github Actions.")
def test_decision_data_retirement_is_absorbing(load_data=LOAD_SAVED_DATA):
    """This functions asserts that retirement is absorbing."""
    paths_dict = create_path_dict()

    dec_dat = create_structural_est_sample(
        paths_dict,
        specs=None,
        load_data=load_data,
    )
    assert dec_dat.loc[dec_dat["lagged_choice"] == 0, "choice"].unique() == 0
