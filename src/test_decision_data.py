import pytest
from gather_decision_data import gather_decision_data

LOAD_SAVED_DATA = True
USER = "bruno"


# Define paths_dict as a fixture
@pytest.fixture
def paths_dict():
    if USER == "bruno":
        return {
            # SOEP Core and SOEP RV are saved locally
            "soep_c38": "C:/Users/bruno/papers/soep/soep38",
            "soep_rv": "C:/Users/bruno/papers/soep/soep_rv",
        }
    elif USER == "max":
        return {
            "soep_c38": "/home/maxbl/Uni/pol_uncetainty/data/soep38",
            "soep_rv": "/home/maxbl/Uni/pol_uncetainty/data/soep_rv",
        }


# Define data_options as a fixture
@pytest.fixture
def data_options():
    min_SRA = 65
    min_ret_age = min_SRA - 4
    max_ret_age = 72
    exp_cap = 40  # maximum number of periods of exp accumulation
    start_year = 2010  # start year of estimation sample
    end_year = 2021  # end year of estimation sample
    return {
        "start_year": start_year,
        "end_year": end_year,
        "start_age": 25,
        "min_ret_age": min_ret_age,
        "max_ret_age": max_ret_age,
        "exp_cap": exp_cap,
    }

# define policy_step_size as a fixture
@pytest.fixture
def policy_step_size():
    policy_step_size = 0.04478741131783991
    return policy_step_size

# These function tests the decision data for consistency (cf. model state space sparsity condition).


def test_decision_data_no_missing_values(
    paths_dict, data_options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that there are no missing values for any states"""
    dec_dat = gather_decision_data(paths_dict, data_options, policy_step_size, load_data=load_data)
    assert dec_dat["choice"].isna().sum() == 0
    assert dec_dat["period"].isna().sum() == 0
    assert dec_dat["lagged_choice"].isna().sum() == 0
    assert dec_dat["policy_state"].isna().sum() == 0
    # assert dec_dat["retirement_age_id"].isna().sum() == 0  // this is the only exception because it is irrelevant for matched decisions
    assert dec_dat["experience"].isna().sum() == 0


def test_decision_data_no_ret_before_min_ret_age(
    paths_dict, data_options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that nobody is retired before min_ret_age"""
    dec_dat = gather_decision_data(paths_dict, data_options, policy_step_size, load_data=load_data)
    assert (
        dec_dat.loc[dec_dat["choice"] == 2, "period"].min() + data_options["start_age"]
        >= data_options["min_ret_age"]
    )


def test_decision_data_no_work_after_max_ret_age(
    paths_dict, data_options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that there are no working after max_ret_age"""
    dec_dat = gather_decision_data(paths_dict, data_options, policy_step_size, load_data=load_data)
    assert (
        dec_dat.loc[dec_dat["choice"] == 1, "period"].max() + data_options["start_age"]
        <= data_options["max_ret_age"]
    )
    assert (
        dec_dat.loc[dec_dat["choice"] == 0, "period"].max() + data_options["start_age"]
        <= data_options["max_ret_age"]
    )


def test_decision_data_exp_cap(paths_dict, data_options, policy_step_size, load_data=LOAD_SAVED_DATA):
    """This functions asserts that experience is smaller or equal to age and exp_cap"""
    dec_dat = gather_decision_data(paths_dict, data_options, policy_step_size, load_data=load_data)
    assert dec_dat["experience"].max() <= data_options["exp_cap"]
    assert dec_dat["experience"].max() <= dec_dat["period"].max()


def test_decision_data_retirement_is_absorbing(
    paths_dict, data_options, policy_step_size, load_data=LOAD_SAVED_DATA
):
    """This functions asserts that retirement is absorbing"""
    dec_dat = gather_decision_data(paths_dict, data_options, policy_step_size, load_data=load_data)
    assert dec_dat.loc[dec_dat["lagged_choice"] == 2, "choice"].unique() == 2
