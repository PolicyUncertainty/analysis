import jax.numpy as jnp
import pytest

from model_code.policy_processes.step_function import (
    realized_policy_step_function,
    update_specs_step_function_with_slope,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs


@pytest.fixture(scope="module")
def paths_and_specs():
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
    return path_dict, specs


def test_step_function(paths_and_specs):
    path_dict, specs = paths_and_specs

    specs = update_specs_step_function_with_slope(
        specs=specs,
        slope=1.0,
    )

    realized_policy_step_function(
        policy_state=jnp.array(29, dtype=jnp.uint8),
        period=jnp.array(3, dtype=jnp.uint8),
        lagged_choice=jnp.array(3, dtype=jnp.uint8),
        choice=jnp.array(3, dtype=jnp.uint8),
        model_specs=specs,
    )
