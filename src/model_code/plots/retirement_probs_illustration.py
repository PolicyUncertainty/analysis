import matplotlib.pyplot as plt
import numpy as np

from model_code.state_space.experience import scale_experience_years


def plot_ret_probs_for_asset_level(model_solved, specs, assets):
    periods = np.arange(30, 40, dtype=int)
    states = {
        "period": periods,
        "lagged_choice": np.ones_like(periods) * 3,
        "education": np.zeros_like(periods),
        "sex": np.zeros_like(periods),
        "informed": np.ones_like(periods),
        "policy_state": np.ones_like(periods) * 8,
        "job_offer": np.ones_like(periods),
        "partner_state": np.zeros_like(periods),
        "health": np.zeros_like(periods),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for exp_years in np.arange(10, 50, 10):
        states["assets_begin_of_period"] = np.ones_like(periods, dtype=float) * assets
        exp = scale_experience_years(
            exp_years, periods, specs["max_exp_diffs_per_period"]
        )
        states["experience"] = exp

        choice_probs = model_solved.choice_probabilities_for_states(states=states)
        ax.plot(
            periods + 30,
            np.nan_to_num(choice_probs[:, 0], nan=0.0),
            label=f"Asset: {assets}, Exp {exp_years}",
        )
        ax.legend()

    plt.show()
