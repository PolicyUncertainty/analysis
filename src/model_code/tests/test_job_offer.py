from itertools import product

import numpy as np
import pytest
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

EDU_GRID = [0, 1]
PERIOD_GRID = np.arange(0, 20, 2, dtype=int)
LOGIT_PARAM_GRID = np.arange(0.1, 0.9, 0.2)
WORK_CHOICE_GRID = [2, 3]


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "education, period, logit_param, work_choice",
    list(product(EDU_GRID, PERIOD_GRID, LOGIT_PARAM_GRID, WORK_CHOICE_GRID)),
)
def test_job_destruction(education, period, logit_param, work_choice, paths_and_specs):
    # Test job destruction probs.
    path_dict, options = paths_and_specs

    # These are irrelevant!
    params = {
        "job_finding_logit_const": logit_param,
        "job_finding_logit_age": logit_param,
        "job_finding_logit_age_squ": logit_param,
        "job_finding_logit_high_educ": logit_param,
    }
    job_dest_prob = options["job_sep_probs"][education, period]
    full_probs_expec = np.array([job_dest_prob, 1 - job_dest_prob])

    probs = job_offer_process_transition(
        params, options, education, period, work_choice
    )
    np.testing.assert_almost_equal(probs, full_probs_expec)
