"""Functions for pre and post estimation setup."""

import pickle
from functools import partial
from typing import Any, Callable, Dict, Optional

import jax
import numpy as np
import optimagic as om
import pandas as pd
import yaml
from dcegm.asset_correction import adjust_observed_assets

from estimation.msm.scripts.calc_moments import calc_all_moments
from estimation.struct_estimation.scripts.estimate_setup import generate_print_func
from model_code.specify_model import specify_model
from process_data.structural_sample_scripts.create_structural_est_sample import (
    CORE_TYPE_DICT,
)
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import generate_derived_and_data_derived_specs

jax.config.update("jax_enable_x64", True)


def estimate_model(
    path_dict,
    params_to_estimate_names,
    start_params_all: Dict[str, Any],
    file_append,
    load_model: bool = False,
    weighting_method: str = "identity",
    last_estimate: Optional[Dict[str, Any]] = None,
):
    """Estimate the model based on empirical data and starting parameters."""
    print_function = generate_print_func(params_to_estimate_names)

    # # Assign start params from before
    if last_estimate is not None:
        print_function(last_estimate)

        for key in start_params_all:
            try:
                print(
                    f"Start params value of {key} was {start_params_all[key]} and is "
                    f"replaced by {last_estimate[key]}",
                    flush=True,
                )
            except:
                raise ValueError(f"Key {key} not found in last_estimate.")
            start_params_all[key] = last_estimate[key]

    start_params = {name: start_params_all[name] for name in params_to_estimate_names}
    print_function(start_params)

    lower_bounds_all = yaml.safe_load(
        open(path_dict["start_params_and_bounds"] + "lower_bounds.yaml", "rb")
    )
    lower_bounds = {name: lower_bounds_all[name] for name in params_to_estimate_names}

    upper_bounds_all = yaml.safe_load(
        open(path_dict["start_params_and_bounds"] + "upper_bounds.yaml", "rb")
    )
    upper_bounds = {name: upper_bounds_all[name] for name in params_to_estimate_names}

    bounds = om.Bounds(lower=lower_bounds, upper=upper_bounds)

    sim_specs = {
        "announcement_age": None,
        "SRA_at_start": 67,
        "SRA_at_retirement": 67,
    }

    model = specify_model(
        path_dict=path_dict,
        subj_unc=True,
        custom_resolution_age=None,
        load_model=load_model,
        sim_specs=sim_specs,
    )

    data_decision = load_and_prep_data(
        path_dict=path_dict, start_params=start_params, model_class=model
    )

    # Load empirical data
    empirical_moments = calc_all_moments(data_decision)

    if weighting_method == "identity":
        weights = np.identity(empirical_moments.shape[0])
    # elif weighting_method == "diagonal":
    #     empirical_variances_reg = np.maximum(empirical_variances, 1e-4)
    #     weights = np.diag(1 / empirical_variances_reg)
    else:
        raise ValueError(f"Unknown weighting method: {weighting_method}")

    initial_states = generate_start_states_from_obs(
        path_dict=path_dict,
        params=start_params_all,
        model_class=model,
        inital_SRA=67,
        only_informed=False,
    )

    sim_func = model.get_solve_and_simulate_func(
        states_initial=initial_states, seed=model.model_specs["seed"]
    )

    def simulate_moments_for_params(params_int):
        df = sim_func(params_int)
        df = df.reset_index()
        moments = calc_all_moments(df)

        return moments

    criterion_func = get_msm_optimization_function(
        simulate_moments=simulate_moments_for_params,
        print_function=print_function,
        start_params_all=start_params_all,
        empirical_moments=empirical_moments,
        weights=weights,
    )

    # algo_options = {
    #     "convergence.relative_criterion_change": 1e-14,
    #     # "stopping.max_iterations": 2,
    #     "noisy": False,
    #     "n_cores": 1,
    #     "batch_size": 4,
    #     # "logging": "my_log.db",
    #     # "log_options": {"fast_logging": True},
    # }

    minimize_kwargs = {
        "fun": criterion_func,
        "params": start_params,
        "algorithm": "tranquilo_ls",
        # "algo_options": algo_options,
        "bounds": bounds,
        "error_handling": "continue",
    }

    result = om.minimize(**minimize_kwargs)

    pickle.dump(
        result, open(path_dict["struct_results"] + f"em_result_{file_append}.pkl", "wb")
    )
    start_params_all.update(result.params)

    pickle.dump(
        start_params_all,
        open(path_dict["struct_results"] + f"est_params_{file_append}.pkl", "wb"),
    )
    return result


# =====================================================================================
# Criterion function
# =====================================================================================
def get_msm_optimization_function(
    simulate_moments: callable,
    start_params_all: Dict[str, Any],
    print_function: Callable,
    empirical_moments: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:

    chol_weights = np.linalg.cholesky(weights)

    criterion = om.mark.least_squares(
        partial(
            msm_criterion,
            simulate_moments=simulate_moments,
            print_function=print_function,
            start_params_all=start_params_all,
            flat_empirical_moments=empirical_moments,
            chol_weights=chol_weights,
        )
    )

    return criterion


def msm_criterion(
    params: np.ndarray,
    start_params_all: Dict[str, Any],
    print_function: Callable,
    simulate_moments: callable,
    flat_empirical_moments: np.ndarray,
    chol_weights: np.ndarray,
) -> np.ndarray:
    """Calculate the raw criterion based on simulated and empirical moments."""

    params_int = start_params_all.copy()
    params_int.update(params)

    simulated_flat = simulate_moments(params_int)

    difference = simulated_flat - flat_empirical_moments
    deviations = difference / flat_empirical_moments
    mask = ~np.isfinite(deviations)
    deviations[mask] = difference[mask]
    residuals = deviations @ chol_weights
    # Print squared sum of residuals
    print(f"Sum of squared residuals: {np.sum(residuals**2):.4f} ")
    print_function(params)
    return residuals


# =====================================================================================
# Preparation for estimation
# =====================================================================================


def load_and_prep_data(path_dict, start_params, model_class):
    specs = generate_derived_and_data_derived_specs(path_dict)
    # Load data
    data_decision = pd.read_csv(path_dict["struct_est_sample"])
    data_decision = data_decision.astype(CORE_TYPE_DICT)

    model_specs = model_class.model_specs
    #
    # data_decision["age"] = data_decision["period"] + model_specs["start_age"]
    # data_decision["age_bin"] = np.floor(data_decision["age"] / 10)
    # data_decision.loc[data_decision["age_bin"] > 6, "age_bin"] = 6
    # age_bin_av_size = data_decision.shape[0] / data_decision["age_bin"].nunique()
    # data_decision.loc[:, "age_weights"] = 1.0
    # data_decision.loc[:, "age_weights"] = age_bin_av_size / data_decision.groupby(
    #     "age_bin"
    # )["age_weights"].transform("sum")
    #
    # # Transform experience
    max_init_exp = model_specs["max_exp_diffs_per_period"][
        data_decision["period"].values
    ]
    exp_denominator = data_decision["period"].values + max_init_exp
    data_decision["experience"] = data_decision["experience"] / exp_denominator

    # We can adjust wealth outside, as it does not depend on estimated parameters
    # (only on interest rate)
    # Now transform for dcegm
    states_dict = {
        name: data_decision[name].values
        for name in model_class.model_structure["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["assets_begin_of_period"] = (
        data_decision["wealth"].values / specs["wealth_unit"]
    )

    assets_begin_of_period = adjust_observed_assets(
        observed_states_dict=states_dict,
        params=start_params,
        model_class=model_class,
    )
    data_decision["assets_begin_of_period"] = assets_begin_of_period

    return data_decision
