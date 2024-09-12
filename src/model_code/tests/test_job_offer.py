from itertools import product

import numpy as np
import pytest
from model_code.derive_specs import generate_derived_and_data_derived_specs
from model_code.job_offers import job_offer_process_transition
from set_paths import create_path_dict

EDU_GRID = [0, 1]
PERIOD_GRID = np.arange(0, 20, 2, dtype=int)
LOGIT_PARAM_GRID = np.arange(0.1, 0.9, 0.2)


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "education, period, logit_param",
    list(product(EDU_GRID, PERIOD_GRID, LOGIT_PARAM_GRID)),
)
def test_job_destruction(education, period, logit_param, paths_and_specs):
    # Test job destruction probs. Individual works
    choice = 1
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

    probs = job_offer_process_transition(params, options, education, period, choice)
    np.testing.assert_almost_equal(probs, full_probs_expec)
