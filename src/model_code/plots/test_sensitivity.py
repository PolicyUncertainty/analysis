import numpy as np


def test_solution_changes(model_solved, specs, choice=3):
    """Test if changing one state variable affects the model solution."""

    # Base state
    n_obs = 1
    prototype_array = np.arange(n_obs)
    base_states = {
        "period": np.ones_like(prototype_array) * 69,
        "lagged_choice": np.ones_like(prototype_array) * 3,
        "education": np.zeros_like(prototype_array),
        "sex": np.zeros_like(prototype_array),
        "informed": np.zeros_like(prototype_array),
        "policy_state": np.zeros_like(prototype_array),
        "job_offer": np.zeros_like(prototype_array),
        "partner_state": np.zeros_like(prototype_array),
        "health": np.zeros_like(prototype_array),
    }

    # Get baseline solution
    baseline_endog, baseline_value, baseline_policy = (
        model_solved.get_solution_for_discrete_state_choice(
            states=base_states, choices=np.ones_like(prototype_array) * choice
        )
    )

    # Test ranges for each variable
    test_ranges = {
        "education": [1],
        "sex": [1],
        "health": [1, 2],
        "partner_state": [1, 2],
        "policy_state": [8, 12],
        "job_offer": [1],
        "informed": [1],
    }

    print(f"Testing solution sensitivity for choice {choice}:")
    print("-" * 50)

    for state_var, test_values in test_ranges.items():
        for test_value in test_values:
            # Modify one state variable
            modified_states = base_states.copy()
            modified_states[state_var] = np.ones_like(prototype_array) * test_value

            # Get modified solution
            mod_endog, mod_value, mod_policy = (
                model_solved.get_solution_for_discrete_state_choice(
                    states=modified_states,
                    choices=np.ones_like(prototype_array) * choice,
                )
            )

            # Check if anything changed
            changed = (
                not np.allclose(baseline_value, mod_value, atol=1e-2)
                or not np.allclose(baseline_endog, mod_endog, atol=1e-2)
                or not np.allclose(baseline_policy, mod_policy, atol=1e-2)
            )

            status = "CHANGED" if changed else "NO CHANGE"
            print(
                f"{state_var}: {base_states[state_var][0]} -> {test_value}: {status}",
                flush=True,
            )
