from itertools import product

import numpy as np
import pytest
from model_code.stochastic_processes.partner_transitions import partner_transition
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

EDU_GRID = np.arange(2, dtype=int)
PERIOD_GRID = np.linspace(0, 45, 1, dtype=int)
PARTNER_STATE_GRID = np.arange(3, dtype=int)


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


@pytest.mark.parametrize(
    "education, period, partner_state",
    list(product(EDU_GRID, PERIOD_GRID, PARTNER_STATE_GRID)),
)
def test_vec_shape(education, period, partner_state, paths_and_specs):
    path_dict, specs = paths_and_specs
    res = partner_transition(period, education, partner_state, specs)
    assert res.shape == (specs["n_partner_states"],)
