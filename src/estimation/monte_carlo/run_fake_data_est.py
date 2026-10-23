# Monte Carlo: simulate data from the easy model with known params and check if the
# estimation recovers them. See monte_carlo.md in this folder.
#
# Run from src/, e.g.
#   python estimation/monte_carlo/run_fake_data_est.py --informed all --sample full
# --informed all: everybody informed and observed as informed.
# The other options simulate uninformed individuals as in the full model:
# --informed data_coding: informed coded as in the real data (retirees from 63 on are
#   observed as informed, everybody else is unobserved).
# --informed unobserved: informed unobserved for everybody.
# --informed true: informed observed for everybody (benchmark).
# --informed informed_from_63: everybody is informed from 63 on in the simulation,
#   so the data coding is correct; coded as data_coding.
# --job-offer working_only: job offer observed (1) only for individuals working.
# --job-offer data_coding: additionally observed for individuals who worked last
#   period and stop working, as in the real data (0 if fired, 1 otherwise).
# --sample full: the full simulated panel.
# --sample data_ratio: panels censored such that the ratio of fresh retirees to
#   working life observations is the one of the real data.
import argparse
import os
import pickle as pkl
import time

import jax.numpy as jnp
import numpy as np
import optimagic as om
import pandas as pd
import yaml

from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=False)

from estimation.monte_carlo.specify_easy_model import SRA_67_INDEX, specify_easy_model
from estimation.struct_estimation.scripts.estimate_setup import (
    compare_probs_with_reference,
)
from estimation.struct_estimation.scripts.std_errors import calc_scores
from estimation.struct_estimation.start_params_and_bounds.param_lists import (
    men_disability_params,
    men_disutil_params,
    men_job_offer_params,
)
from model_code.transform_data_from_model import (
    create_informed_probability,
    create_states_dict,
)
from model_code.unobserved_state_weighting import create_unobserved_state_specs
from simulation.sim_tools.start_obs_for_sim import generate_start_states_from_obs
from specs.derive_specs import generate_derived_and_data_derived_specs

# Estimated params the true params are based on. They must be a complete params
# vector of the current model, as all params which are not estimated are kept at
# these values.
TRUE_PARAMS_NAME = "alg1_sparse_inf"
# Common shift of the disutility of full-time work and unemployment (good and bad
# health) of men in the true params, by true params name. The differences between
# the disutilities, and with it the labor shares of working individuals, stay the
# same. The level is calibrated with calibrate_disutil_shift.py to the share of
# fresh retirements at 63 in the data.
DISUTIL_SHIFTS = {
    "alg1_sparse_inf": -0.9375,
}
SHIFTED_DISUTIL_PARAMS = [
    "disutil_ft_work_good_men",
    "disutil_ft_work_bad_men",
    "disutil_unemployed_good_men",
    "disutil_unemployed_bad_men",
]

# The high education shifter of job finding is not identified with low educated
# men only.
PARAMS_TO_ESTIMATE_NAMES = [
    name
    for name in men_disutil_params + men_disability_params + men_job_offer_params
    if name != "job_finding_logit_high_educ_men"
]

# Specification of the disability logit of men. "steps" has a constant and steps at
# 50, 55 and 60. "above_55" has a constant and a single step at 55, the steps at 50
# and 60 are zero in the true params and not estimated.
DISABILITY_SPECS = ["steps", "above_55"]
DISABILITY_PARAMS_FIXED_AT_ZERO = {
    "steps": [],
    "above_55": ["disability_logit_above_50_men", "disability_logit_above_60_men"],
}

# Relative scale of the perturbation of the start values around the true params.
# Params with a true value close to zero are perturbed on an absolute scale of
# START_PERTURBATION * 0.1.
START_PERTURBATION = 0.25
# If True, job offer, health and informed are fully observed. Otherwise they are
# masked as described in monte_carlo.md.
OBSERVE_ALL_STATES = False


def create_true_params(true_params_name, disutil_shift, disability_spec="steps"):
    params = pkl.load(
        open(paths_dict["struct_results"] + f"est_params_{true_params_name}.pkl", "rb")
    )
    params = params.copy()
    for name in SHIFTED_DISUTIL_PARAMS:
        params[name] = float(params[name]) + disutil_shift
    if disability_spec == "above_55":
        params = _disability_above_55_params(params)
    return params


def _disability_above_55_params(params):
    """Disability logit with a constant and a single step at 55.

    The constant is unchanged. The step is chosen such that the disability
    probability from 55 on is the mean of the probabilities of the steps
    specification at ages 55 to 62.

    """
    params = params.copy()
    const = float(params["disability_logit_const_men"])
    ages = np.arange(55, 63)
    logits = (
        const
        + float(params["disability_logit_above_50_men"])
        + float(params["disability_logit_above_55_men"])
        + float(params["disability_logit_above_60_men"]) * (ages >= 60)
    )
    mean_prob = (1 / (1 + np.exp(-logits))).mean()
    params["disability_logit_above_55_men"] = float(
        np.log(mean_prob / (1 - mean_prob)) - const
    )
    for name in DISABILITY_PARAMS_FIXED_AT_ZERO["above_55"]:
        params[name] = 0.0
    return params


def true_params_label(true_params_name, disutil_shift, disability_spec="steps"):
    label = f"{true_params_name}_shift_{disutil_shift:.3f}"
    if disability_spec != "steps":
        label += f"_dis_{disability_spec}"
    return label


INFORMED_CODINGS = ["all", "data_coding", "unobserved", "true", "informed_from_63"]
JOB_OFFER_CODINGS = ["working_only", "data_coding"]
# Age from which retirees are coded as informed in the real data
INFORMED_RETIREMENT_AGE = 63


def simulate_fake_data(model_solved, model, params, n_agents, seed, only_informed):
    """Simulate low educated men starting with SRA 67.

    Initial states are drawn from the observed start period. With only_informed,
    everybody starts informed, otherwise the informed state is drawn from the
    initial informed shares.

    """
    initial_states = generate_start_states_from_obs(
        path_dict=paths_dict,
        params=params,
        inital_SRA=67,
        seed=seed,
        model_class=model,
        only_informed=only_informed,
    )
    low_educ_men = np.where(
        (np.asarray(initial_states["sex"]) == 0)
        & (np.asarray(initial_states["education"]) == 0)
    )[0]
    keep = np.random.default_rng(seed).choice(
        low_educ_men, size=min(n_agents, len(low_educ_men)), replace=False
    )
    initial_states = {
        name: np.asarray(arr)[keep] for name, arr in initial_states.items()
    }
    # The easy model holds the index of the policy state in its own grid
    initial_states["policy_state"] = np.full_like(
        initial_states["policy_state"], SRA_67_INDEX
    )
    print(f"Simulating {len(keep)} low educated men.", flush=True)

    model_specs = model.model_specs
    data_sim = model_solved.simulate(states_initial=initial_states, seed=seed)
    data_sim = data_sim.reset_index()
    data_sim = data_sim[
        data_sim["health"] != int(model_specs["death_health_var"])
    ].copy()
    data_sim.sort_values(["agent", "period"], inplace=True)
    data_sim.reset_index(drop=True, inplace=True)
    data_sim["age"] = data_sim["period"] + int(model_specs["start_age"])
    return data_sim


def mask_fake_data(
    data_sim,
    model_specs,
    informed_coding,
    observe_all_states,
    job_offer_coding="working_only",
):
    """Code the simulated states as in the observed data.

    informed_coding is one of INFORMED_CODINGS and job_offer_coding one of
    JOB_OFFER_CODINGS, see the top of this file.

    """
    bad_health_var = int(model_specs["bad_health_var"])
    disabled_health_var = int(model_specs["disabled_health_var"])
    working_choices = [2, 3]

    data_fake = data_sim.copy()

    # Lagged health as in the observed data, where the survey distinguishes only
    # good and bad health. The first period has no lag in the simulation and is
    # dropped. Its states are the initial states, which are drawn from the data and
    # not from the model's transitions (e.g. the share of disabled), so the
    # likelihood must not treat them as transitions.
    surveyed_health = data_fake["health"].replace(disabled_health_var, bad_health_var)
    data_fake["lagged_health"] = surveyed_health.groupby(data_fake["agent"]).shift(1)
    data_fake = data_fake[data_fake["lagged_health"].notna()].copy()
    data_fake["lagged_health"] = data_fake["lagged_health"].astype(int)

    data_fake["health"] = data_fake["health"].astype(int)
    data_fake["informed"] = data_fake["informed"].astype(int)

    if informed_coding == "all":
        # Everybody is informed and this is observed.
        data_fake["informed"] = 1
        data_fake["probability_informed"] = 1.0
    else:
        # Unobserved informed states are weighted with the predicted share of
        # informed by age, as in the estimation on the real data.
        data_fake = create_informed_probability(data_fake, model_specs)
        retired_past_63 = (data_fake["choice"] == 0) & (
            data_fake["age"] >= INFORMED_RETIREMENT_AGE
        )
        fresh_retired_past_63 = retired_past_63 & (data_fake["lagged_choice"] != 0)
        n_uninformed = (fresh_retired_past_63 & (data_fake["informed"] == 0)).sum()
        print(
            f"{n_uninformed} of {fresh_retired_past_63.sum()} fresh retirees from "
            f"{INFORMED_RETIREMENT_AGE} on are uninformed in the simulation.",
            flush=True,
        )
        if informed_coding == "informed_from_63":
            # Everybody is informed from 63 on in the simulation.
            data_fake.loc[
                data_fake["age"] >= INFORMED_RETIREMENT_AGE, "probability_informed"
            ] = 1.0
        if observe_all_states or informed_coding == "true":
            pass
        elif informed_coding == "unobserved":
            data_fake["informed"] = -99
        elif informed_coding in ["data_coding", "informed_from_63"]:
            # As in process_data/structural_sample_scripts/informed_state.py:
            # individuals choosing retirement from 63 on are coded as informed,
            # everybody else (including disability pensioners) is unobserved.
            data_fake["informed"] = -99
            data_fake.loc[retired_past_63, "informed"] = 1
        else:
            raise ValueError(f"Unknown informed coding {informed_coding}.")

    if not observe_all_states:
        # Health: bad and disabled are not distinguished, except for fresh retirees
        # below the long insured retirement age, who must be disabled.
        fresh_early_retired = (
            (data_fake["choice"] == 0)
            & (data_fake["lagged_choice"] != 0)
            & (data_fake["age"] < int(model_specs["min_long_insured_age"]))
        )
        n_not_disabled = (
            fresh_early_retired & (data_fake["health"] != disabled_health_var)
        ).sum()
        if n_not_disabled > 0:
            raise ValueError(
                f"{n_not_disabled} fresh retirees below the long insured retirement "
                f"age are not disabled in the simulation."
            )
        data_fake.loc[
            data_fake["health"].isin([bad_health_var, disabled_health_var]), "health"
        ] = -99
        data_fake.loc[fresh_early_retired, "health"] = disabled_health_var

        # Job offer: observed for everybody working.
        working = data_fake["choice"].isin(working_choices)
        stopped_working = (~working) & data_fake["lagged_choice"].isin(working_choices)
        true_job_offer = data_fake["job_offer"].copy()
        data_fake["job_offer"] = -99
        data_fake.loc[working, "job_offer"] = 1
        if job_offer_coding == "data_coding":
            # As in process_data/soep_vars/job_hire_and_fire.py: individuals who
            # worked last period and stop working are observed, with no job offer
            # if they were fired and a job offer otherwise. This is the true state.
            data_fake.loc[stopped_working, "job_offer"] = true_job_offer[
                stopped_working
            ]
            print(
                f"Job offer of {stopped_working.sum()} individuals who stopped "
                f"working observed: "
                f"{(true_job_offer[stopped_working] == 0).sum()} fired.",
                flush=True,
            )
        elif job_offer_coding != "working_only":
            raise ValueError(f"Unknown job offer coding {job_offer_coding}.")

    # Already retired individuals hold no identification
    return data_fake[data_fake["lagged_choice"] != 0].copy()


def data_fresh_retirement_ratio():
    """Fresh retirements per working life observation of low educated men in the
    estimation sample."""
    data = pd.read_csv(paths_dict["struct_est_sample"])
    data = data[
        (data["sex"] == 0) & (data["education"] == 0) & (data["lagged_choice"] != 0)
    ]
    return (data["choice"] == 0).mean()


def censor_to_retirement_ratio(data_est, target_ratio, seed):
    """Censor panels such that the share of fresh retirements is target_ratio.

    In the real data, many individuals are observed during their working life but
    not when they retire (e.g. young cohorts). We mimic this with random right
    censoring of the panels, which is independent of the choices: a share q of
    individuals is observed only up to a random period, drawn uniformly from their
    observed periods. q is chosen such that the share of fresh retirements among
    the remaining observations is target_ratio.

    """
    rng = np.random.default_rng(seed + 1)
    agents = data_est["agent"].unique()
    u_censored = pd.Series(rng.uniform(size=len(agents)), index=agents)
    u_period = pd.Series(rng.uniform(size=len(agents)), index=agents)

    first = data_est.groupby("agent")["period"].min()
    last = data_est.groupby("agent")["period"].max()
    # Last period observed if censored
    censor_period = first + np.floor(u_period * (last - first + 1)).astype(int) - 1
    censor_period_obs = data_est["agent"].map(censor_period)
    u_censored_obs = data_est["agent"].map(u_censored)

    def censor(q):
        censored = u_censored_obs < q
        return data_est[~censored | (data_est["period"] <= censor_period_obs)]

    def ratio(df):
        return (df["choice"] == 0).mean()

    if ratio(data_est) <= target_ratio:
        raise ValueError(
            f"The simulated ratio of fresh retirements {ratio(data_est):.4f} is "
            f"already below the target {target_ratio:.4f}."
        )
    q_low, q_high = 0.0, 1.0
    if ratio(censor(q_high)) > target_ratio:
        raise ValueError("Censoring all panels does not reach the target ratio.")
    for _ in range(30):
        q = (q_low + q_high) / 2
        if ratio(censor(q)) > target_ratio:
            q_low = q
        else:
            q_high = q
    data_censored = censor(q_high).copy()
    print(
        f"Censored panels of {q_high:.1%} of individuals: "
        f"{len(data_censored)} of {len(data_est)} observations and "
        f"{(data_censored['choice'] == 0).sum()} of {(data_est['choice'] == 0).sum()} "
        f"fresh retirements kept. Ratio {ratio(data_censored):.4f} "
        f"(data {target_ratio:.4f}).",
        flush=True,
    )
    return data_censored


def draw_start_values(true_values, lower_bounds, upper_bounds, seed):
    """Perturb the true values randomly within the bounds."""
    rng = np.random.default_rng(seed)
    start_values = {}
    for name, true_value in true_values.items():
        scale = START_PERTURBATION * max(abs(true_value), 0.1)
        start_values[name] = float(
            np.clip(
                true_value + scale * rng.standard_normal(),
                lower_bounds[name],
                upper_bounds[name],
            )
        )
    return start_values


def estimate_on_fake_data(
    model, params, data_est, seed, mc_folder, results_append, disability_spec="steps"
):
    params_to_estimate_names = [
        name
        for name in PARAMS_TO_ESTIMATE_NAMES
        if name not in DISABILITY_PARAMS_FIXED_AT_ZERO[disability_spec]
    ]
    lower_bounds_all = yaml.safe_load(
        open(paths_dict["start_params_and_bounds"] + "lower_bounds.yaml", "rb")
    )
    upper_bounds_all = yaml.safe_load(
        open(paths_dict["start_params_and_bounds"] + "upper_bounds.yaml", "rb")
    )
    lower_bounds = {name: lower_bounds_all[name] for name in params_to_estimate_names}
    upper_bounds = {name: upper_bounds_all[name] for name in params_to_estimate_names}

    # All params which are not estimated are kept at their true values.
    true_values = {name: float(params[name]) for name in params_to_estimate_names}
    start_values = draw_start_values(true_values, lower_bounds, upper_bounds, seed)

    ll_func = model.create_experimental_ll_func(
        params_all=params,
        observed_states=create_states_dict(data_est, model_class=model),
        observed_choices=data_est["choice"].values,
        unobserved_state_specs=create_unobserved_state_specs(data_decision=data_est),
    )

    def crit_func(params_est):
        start = time.time()
        ll_value = jnp.mean(ll_func(params_est))
        print("Params: ", params_est, flush=True)
        print("Negative log-likelihood: ", ll_value, flush=True)
        print("Likelihood evaluation took ", time.time() - start, flush=True)
        return ll_value

    # The true params should fit at least as well as the start values.
    crit_true = float(crit_func(true_values))
    crit_start = float(crit_func(start_values))
    print(
        f"Criterion at true params: {crit_true}, at start values: {crit_start}",
        flush=True,
    )

    result = om.minimize(
        fun=crit_func,
        params=start_values,
        bounds=om.Bounds(lower=lower_bounds, upper=upper_bounds),
        algorithm="scipy_lbfgsb",
        # Close to the optimum, noise in the numerical gradient keeps L-BFGS-B
        # evaluating without progress, so we stop on a small relative change.
        algo_options={"convergence_ftol_rel": 1e-8, "stopping_maxfun": 100},
        error_handling="continue",
        scaling=om.ScalingOptions(method="bounds"),
    )
    pkl.dump(result, open(mc_folder + f"result_{results_append}.pkl", "wb"))
    est_values = {name: float(result.params[name]) for name in params_to_estimate_names}

    # Standard errors from the outer product of the scores. Scores are summed by
    # individual, as observations of the same individual are not independent.
    scores = calc_scores(
        ll_func=ll_func,
        est_params=est_values,
        params_to_estimate_names=params_to_estimate_names,
    )
    scores_agent = pd.DataFrame(scores).groupby(data_est["agent"].values).sum().values
    n_obs = scores.shape[0]
    opg = scores_agent.T @ scores_agent / n_obs
    cov = np.linalg.inv(opg) / n_obs
    std_errors = np.sqrt(np.diag(cov))

    recovery = pd.DataFrame(
        {
            "true": true_values,
            "start": start_values,
            "estimate": est_values,
            "std_error": dict(zip(params_to_estimate_names, std_errors)),
        }
    )
    recovery["t_vs_true"] = (recovery["estimate"] - recovery["true"]) / recovery[
        "std_error"
    ]
    recovery.to_csv(mc_folder + f"recovery_{results_append}.csv")
    pd.DataFrame(
        cov, index=params_to_estimate_names, columns=params_to_estimate_names
    ).to_csv(mc_folder + f"cov_{results_append}.csv")
    print(recovery.to_string(), flush=True)
    print(
        f"Criterion at true params: {crit_true}, at start values: {crit_start}, "
        f"at estimate: {result.fun}",
        flush=True,
    )

    # Implied job finding and disability probabilities against the true ones
    est_params_all = params.copy()
    est_params_all.update(est_values)
    compare_probs_with_reference(
        reference_params=params,
        est_params=est_params_all,
        specs=model.model_specs,
        path=mc_folder + f"prob_check_{results_append}.csv",
    )
    return recovery


def solve_easy_model(model, params, mc_folder, label, load_solution):
    """load_solution: None solves and doesn't save, False solves and saves, True
    loads the saved solution."""
    solution_file = mc_folder + f"sol_{label}.pkl"
    if load_solution is None:
        return model.solve(params)
    elif load_solution:
        return model.solve(params, load_sol_path=solution_file)
    else:
        return model.solve(params, save_sol_path=solution_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--informed", choices=INFORMED_CODINGS, default="all")
    parser.add_argument("--sample", choices=["full", "data_ratio"], default="full")
    parser.add_argument(
        "--job-offer", choices=JOB_OFFER_CODINGS, default="working_only"
    )
    parser.add_argument("--seed", type=int, default=123)
    # Number of simulated individuals, drawn from the initial states of low
    # educated men. The observed estimation sample has about 21,000 observations
    # of low educated men.
    parser.add_argument("--n-agents", type=int, default=1_000)
    parser.add_argument("--true-params", default=TRUE_PARAMS_NAME)
    # Defaults to the calibrated shift of the true params
    parser.add_argument("--disutil-shift", type=float, default=None)
    # Disabled individuals must take up the disability pension
    parser.add_argument("--disabled-must-retire", action="store_true")
    parser.add_argument("--disability-spec", choices=DISABILITY_SPECS, default="steps")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--load-solution", action="store_true")
    args = parser.parse_args()

    if args.disutil_shift is None:
        if args.true_params not in DISUTIL_SHIFTS:
            raise ValueError(
                f"No calibrated disutility shift for {args.true_params}. Run "
                f"calibrate_disutil_shift.py and add it to DISUTIL_SHIFTS."
            )
        args.disutil_shift = DISUTIL_SHIFTS[args.true_params]

    only_informed = args.informed == "all"
    mc_name = f"easy_inf_{args.informed}_sample_{args.sample}"
    if args.job_offer != "working_only":
        mc_name += f"_jo_{args.job_offer}"
    if args.disabled_must_retire:
        mc_name += "_dis_ret"
    params_label = true_params_label(
        args.true_params, args.disutil_shift, args.disability_spec
    )
    results_append = f"mc_{mc_name}_{params_label}_seed_{args.seed}"

    mc_folder = paths_dict["intermediate_data"] + "monte_carlo/"
    os.makedirs(mc_folder, exist_ok=True)
    print(f"Running Monte Carlo {results_append}", flush=True)

    params = create_true_params(
        args.true_params, args.disutil_shift, args.disability_spec
    )
    specs = generate_derived_and_data_derived_specs(paths_dict)
    model = specify_easy_model(
        path_dict=paths_dict,
        specs=specs,
        load_model=args.load_model,
        only_informed=only_informed,
        informed_from_age=(
            INFORMED_RETIREMENT_AGE if args.informed == "informed_from_63" else None
        ),
        disabled_must_retire=args.disabled_must_retire,
    )
    model_specs = model.model_specs

    informed_label = "inf" if only_informed else "uninf"
    if args.disabled_must_retire:
        informed_label += "_dis_ret"
    model_solved = solve_easy_model(
        model,
        params,
        mc_folder,
        label=f"easy_{informed_label}_{params_label}",
        load_solution=args.load_solution,
    )
    data_sim = simulate_fake_data(
        model_solved=model_solved,
        model=model,
        params=params,
        n_agents=args.n_agents,
        seed=args.seed,
        only_informed=only_informed,
    )
    data_sim.to_csv(mc_folder + f"data_sim_{results_append}.csv")

    data_est = mask_fake_data(
        data_sim,
        model_specs=model_specs,
        informed_coding=args.informed,
        observe_all_states=OBSERVE_ALL_STATES,
        job_offer_coding=args.job_offer,
    )
    if args.sample == "data_ratio":
        data_est = censor_to_retirement_ratio(
            data_est, target_ratio=data_fresh_retirement_ratio(), seed=args.seed
        )
    print(
        f"Estimation sample has {len(data_est)} observations and "
        f"{(data_est['choice'] == 0).sum()} fresh retirements.",
        flush=True,
    )

    estimate_on_fake_data(
        model=model,
        params=params,
        data_est=data_est,
        seed=args.seed,
        mc_folder=mc_folder,
        results_append=results_append,
        disability_spec=args.disability_spec,
    )


if __name__ == "__main__":
    main()
