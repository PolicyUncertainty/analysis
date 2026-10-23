# Calibrate the common shift of the disutility of full-time work and unemployment of
# men in the true params of the Monte Carlo (DISUTIL_SHIFTS in run_fake_data_est.py).
# The shift moves the value of all non-retirement choices relative to retirement,
# but keeps the difference between work and unemployment. It is chosen such that the
# share of fresh retirements at 63 in the easy model (everybody informed) is the one
# of low educated men in the data. See monte_carlo.md.
#
# Run from src/:
#   python estimation/monte_carlo/calibrate_disutil_shift.py
import argparse

import pandas as pd

from estimation.monte_carlo.run_fake_data_est import (
    TRUE_PARAMS_NAME,
    create_true_params,
    mask_fake_data,
    paths_dict,
    simulate_fake_data,
)
from estimation.monte_carlo.specify_easy_model import specify_easy_model
from specs.derive_specs import generate_derived_and_data_derived_specs

AGE_TARGET = 63


def retirement_moments(df):
    """Share of fresh retirements at AGE_TARGET and choice shares of individuals
    not retired last period."""
    working_life = df[df["lagged_choice"] != 0]
    fresh = working_life[working_life["choice"] == 0]
    choice_shares = working_life["choice"].value_counts(normalize=True)
    working_life_below_target = working_life[working_life["age"] < AGE_TARGET]
    choice_shares_below = working_life_below_target["choice"].value_counts(
        normalize=True
    )
    return {
        "share_fresh_retired_at_63": (fresh["age"] == AGE_TARGET).mean(),
        "mean_retirement_age": fresh["age"].mean(),
        "ratio_fresh_retired": len(fresh) / len(working_life),
        "unemployed_share": choice_shares.get(1, 0.0),
        "ft_share": choice_shares.get(3, 0.0),
        "unemployed_share_below_63": choice_shares_below.get(1, 0.0),
        "ft_share_below_63": choice_shares_below.get(3, 0.0),
    }


def data_moments(model_specs):
    data = pd.read_csv(paths_dict["struct_est_sample"])
    data = data[(data["sex"] == 0) & (data["education"] == 0)].copy()
    data["age"] = data["period"] + int(model_specs["start_age"])
    return retirement_moments(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-params", default=TRUE_PARAMS_NAME)
    parser.add_argument("--n-agents", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-bisection-steps", type=int, default=7)
    parser.add_argument("--disabled-must-retire", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    specs = generate_derived_and_data_derived_specs(paths_dict)
    model = specify_easy_model(
        path_dict=paths_dict,
        specs=specs,
        load_model=args.load_model,
        disabled_must_retire=args.disabled_must_retire,
    )
    target = data_moments(model.model_specs)
    print(f"Data moments: {target}", flush=True)

    results = {}

    def simulated_moments(shift):
        params = create_true_params(args.true_params, shift)
        model_solved = model.solve(params)
        data_sim = simulate_fake_data(
            model_solved=model_solved,
            model=model,
            params=params,
            n_agents=args.n_agents,
            seed=args.seed,
            only_informed=True,
        )
        moments = retirement_moments(
            mask_fake_data(
                data_sim,
                model_specs=model.model_specs,
                informed_coding="all",
                observe_all_states=True,
            )
        )
        results[shift] = moments
        print(f"Shift {shift:.4f}: {moments}", flush=True)
        return moments["share_fresh_retired_at_63"]

    # A lower disutility of the non-retirement choices postpones retirement and
    # lowers the share retiring at 63.
    shift_low, shift_high = -1.0, 0.0
    if simulated_moments(shift_high) < target["share_fresh_retired_at_63"]:
        raise ValueError("Share at 63 already below the target without shift.")
    if simulated_moments(shift_low) > target["share_fresh_retired_at_63"]:
        raise ValueError("Share at 63 above the target at the lowest shift.")
    for _ in range(args.n_bisection_steps):
        shift = (shift_low + shift_high) / 2
        if simulated_moments(shift) > target["share_fresh_retired_at_63"]:
            shift_high = shift
        else:
            shift_low = shift

    table = pd.DataFrame(results).T.sort_index()
    table.loc["data"] = pd.Series(target)
    print(table.to_string(), flush=True)
    best = min(
        results,
        key=lambda s: abs(
            results[s]["share_fresh_retired_at_63"]
            - target["share_fresh_retired_at_63"]
        ),
    )
    print(f"Calibrated shift: {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
