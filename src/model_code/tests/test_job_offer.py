from itertools import product

import numpy as np
import pytest

from model_code.stochastic_processes.job_offers import job_offer_process_transition
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

EDU_GRID = [0, 1]
PERIOD_GRID = np.arange(0, 20, 2, dtype=int)
LOGIT_PARAM_GRID = np.arange(0.1, 0.9, 0.2)
CHOICE_GRID = [2, 3]
SEX_GRID = [0, 1]
HEALTH_GRID = [0, 1]


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "education, sex, health, period, logit_param, choice",
    list(
        product(
            EDU_GRID, SEX_GRID, HEALTH_GRID, PERIOD_GRID, LOGIT_PARAM_GRID, CHOICE_GRID
        )
    ),
)
def test_job_destruction(
    education, sex, health, period, logit_param, choice, paths_and_specs
):
    # Test job destruction probs.
    path_dict, model_specs = paths_and_specs

    age = model_specs["start_age"] + period

    params = {}
    for append in ["men", "women"]:
        gender_params = {
            f"job_finding_logit_const_{append}": logit_param,
            # f"job_finding_logit_age_{append}": logit_param,
            f"job_finding_logit_high_educ_{append}": logit_param,
            f"job_finding_logit_good_health_{append}": logit_param,
            f"job_finding_logit_above_50_{append}": logit_param,
            f"job_finding_logit_above_55_{append}": logit_param,
            f"job_finding_logit_above_60_{append}": logit_param,
        }
        params = {**params, **gender_params}

    if choice > 1:
        job_dest_prob = model_specs["job_sep_probs"][sex, education, age]
        full_probs = np.array([job_dest_prob, 1 - job_dest_prob])
    else:
        append = "men" if sex == 0 else "women"
        good_health = health == model_specs["good_health_var"]

        exp_value = np.exp(
            params[f"job_finding_logit_const_{append}"]
            # + params[f"job_finding_logit_age_{str}"] * age
            + params[f"job_finding_logit_high_educ_{append}"] * education
            + f"job_finding_logit_good_health_{append}" * good_health
            + f"job_finding_logit_above_50_{append}" * (age >= 50)
            + f"job_finding_logit_above_55_{append}" * (age >= 55)
            + f"job_finding_logit_above_60_{append}" * (age >= 60)
        )
        offer_prob = exp_value / (1 + exp_value)
        full_probs = np.array([1 - offer_prob, offer_prob])

    probs = job_offer_process_transition(
        params=params,
        model_specs=model_specs,
        education=education,
        health=health,
        sex=sex,
        period=period,
        choice=choice,
    )
    np.testing.assert_almost_equal(probs, full_probs)
