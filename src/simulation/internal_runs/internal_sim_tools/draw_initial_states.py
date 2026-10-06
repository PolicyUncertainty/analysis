import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.neighbors import KernelDensity

from model_code.state_space.experience import scale_experience_years
from model_code.stochastic_processes.health_transition import (
    calc_disability_probability,
)
from model_code.stochastic_processes.job_offers import job_offer_process_transition
from model_code.transform_data_from_model import load_scale_and_correct_data


def draw_initial_states(
    path_dict, params, model_class, inital_SRA, seed, only_informed=False
):
    """Draw a synthetic start population of ``n_agents`` agents.

    Sibling of ``generate_start_states_from_obs``, which instead replicates the
    observed start-period individuals themselves. This one draws every state from
    its own type-specific (sex x education) marginal and smooths the asset
    distribution with a KDE, so the population size is set by
    ``model_specs["n_agents"]`` rather than by the sample.

    """
    model_specs = model_class.model_specs
    model_structure = model_class.model_structure
    n_agents = model_specs["n_agents"]
    n_alg_1_claim_states = len(
        model_class.model_config["deterministic_states"]["alg_1_claim"]
    )

    # Generate start policy state from initial SRA
    initial_policy_state = np.floor(
        (inital_SRA - model_specs["min_SRA"]) / model_specs["SRA_grid_size"]
    )

    # Shares the observed-data pipeline with the sibling builder, so experience is
    # already scaled to the model's state and assets are already in
    # budget_constraint's individual-bookkeeping units.
    observed_data = load_scale_and_correct_data(
        path_dict=path_dict, model_class=model_class
    )

    np.random.seed(seed)
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"] == min_period].copy()

    # Generate container
    sex_agents = np.array([], np.uint8)
    education_agents = np.array([], np.uint8)
    for sex_var in range(model_specs["n_sexes"]):
        if model_specs["n_sexes"] > 1:
            if sex_var == 0:
                n_agents_sex = n_agents - n_agents // 2
            else:
                n_agents_sex = n_agents // 2
        else:
            n_agents_sex = n_agents

        sex_vars = np.ones(n_agents_sex, np.uint8) * sex_var
        sex_agents = np.append(sex_agents, sex_vars)

        # Restrict start data
        start_data_sex = start_period_data[start_period_data["sex"] == sex_var]

        # Generate education level
        edu_shares = start_data_sex["education"].value_counts(normalize=True)
        n_agents_edu_types = np.round(edu_shares.sort_index() * n_agents_sex).astype(
            int
        )

        # Generate education array
        edu_agents_per_sex = np.repeat(edu_shares.index, n_agents_edu_types)
        education_agents = np.append(education_agents, edu_agents_per_sex)

    # Generate containers
    assets_agents = np.empty(n_agents, np.float64)
    informed_agents = np.zeros(n_agents, np.uint8)
    exp_years_agents = np.empty(n_agents, np.float64)
    lagged_choice = np.empty(n_agents, np.uint8)
    partner_states = np.empty(n_agents, np.uint8)
    health_agents = np.empty(n_agents, np.uint8)
    job_offer_agents = np.empty(n_agents, np.uint8)
    alg_1_claims = np.empty(n_agents, np.uint8)

    for sex_var in range(model_specs["n_sexes"]):
        for edu in range(model_specs["n_education_types"]):
            type_mask = (sex_agents == sex_var) & (education_agents == edu)
            start_period_data_edu = start_period_data[
                (start_period_data["sex"] == sex_var)
                & (start_period_data["education"] == edu)
            ]

            n_agents_edu = np.sum(type_mask)

            # What agents carry *into* the first period; dcegm's simulation runs the
            # budget equation on it like in every other period.
            assets_agents[type_mask] = draw_start_asset_dist(
                start_period_data_edu["assets_end_of_previous_period"],
                n_agents_edu,
                method="kde",
            )

            # Generate edu specific informed shares
            informed_share_edu = model_specs["informed_shares_in_ages"][
                model_specs["start_age"], edu
            ]
            # Draw informed states according to inital distribution
            dist = np.array([1 - informed_share_edu, informed_share_edu])
            informed_draws_edu = np.random.choice(2, n_agents_edu, p=dist)
            informed_agents[type_mask] = informed_draws_edu

            # Generate type specific initial experience distribution
            exp_years_agents[type_mask] = _draw_from_empirical_marginal(
                start_period_data_edu["experience_years"],
                n_support=int(start_period_data_edu["experience_years"].max()) + 1,
                size=n_agents_edu,
            )

            # Generate type specific initial lagged choice distribution
            lagged_choice_edu = _draw_from_empirical_marginal(
                start_period_data_edu["lagged_choice"],
                n_support=model_specs["n_choices"],
                size=n_agents_edu,
            )
            lagged_choice[type_mask] = lagged_choice_edu

            alg_1_claims[type_mask] = _draw_from_empirical_marginal(
                start_period_data_edu["alg_1_claim"],
                n_support=n_alg_1_claim_states,
                size=n_agents_edu,
            )

            # Bad health is never observed in the start sample -- the data codes
            # anything but good health as missing -- so only the good-health share
            # is empirical; the model's own disability probability splits the rest
            # into bad health and disability.
            share_good = (
                start_period_data_edu["health"] == model_specs["good_health_var"]
            ).mean()
            disability_prob = calc_disability_probability(
                params=params,
                sex=sex_var,
                period=jnp.array(-1, dtype=int),
                education=edu,
                model_specs=model_specs,
            )
            health_edu = np.random.choice(
                [
                    model_specs["good_health_var"],
                    model_specs["bad_health_var"],
                    model_specs["disabled_health_var"],
                ],
                size=n_agents_edu,
                p=[
                    share_good,
                    (1 - share_good) * (1 - disability_prob),
                    (1 - share_good) * disability_prob,
                ],
            )
            health_agents[type_mask] = health_edu

            # Job offers are the period-0 realisation of the transition, so it is
            # evaluated from period -1 with the state agents enter period 0 with.
            job_offer_probs = job_offer_process_transition(
                params=params,
                sex=np.full(n_agents_edu, sex_var),
                health=health_edu,
                model_specs=model_specs,
                education=np.full(n_agents_edu, edu),
                policy_state=np.full(n_agents_edu, initial_policy_state),
                period=np.full(n_agents_edu, -1),
                choice=lagged_choice_edu,
            ).T
            # Job offer probs is n_agents x 2. Choose for each row the job offer state
            # with np random choice
            job_offer_edu = np.array(
                [np.random.choice(a=len(p), p=p) for p in job_offer_probs]
            )
            job_offer_agents[type_mask] = job_offer_edu

            # Get type specific partner states
            partner_states[type_mask] = _draw_from_empirical_marginal(
                start_period_data_edu["partner_state"],
                n_support=model_specs["n_partner_states"],
                size=n_agents_edu,
            )

    # Transform experience years into the model's stored state
    exp_agents = np.asarray(
        scale_experience_years(
            experience_years=exp_years_agents,
            period=jnp.zeros_like(exp_years_agents, dtype=int),
            sex=jnp.asarray(sex_agents, dtype=int),
            is_retired=lagged_choice == 0,
            model_specs=model_specs,
        )
    )

    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    policy_state_agents = (jnp.ones_like(exp_agents) * initial_policy_state).astype(
        jnp.uint8
    )

    if only_informed:
        informed_agents = np.ones_like(informed_agents)

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "assets_end_of_previous_period": jnp.array(assets_agents, dtype=jnp.float64),
        "sex": jnp.array(sex_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "informed": jnp.array(informed_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "policy_state": policy_state_agents,
        "job_offer": jnp.array(job_offer_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
        "alg_1_claim": jnp.array(alg_1_claims, dtype=jnp.uint8),
    }

    missing = set(model_structure["discrete_states_names"]) - set(states)
    if missing:
        raise ValueError(
            f"No start distribution is drawn for the discrete state(s) "
            f"{sorted(missing)}. Add one above; the model cannot be simulated "
            "without them."
        )

    return states


def _draw_from_empirical_marginal(observed_values, n_support, size):
    """Draw from the empirical distribution of an integer state over ``range(n_support)``.

    Values the type cell never shows get probability zero rather than being dropped
    from the support, so the draws stay on the model's own state values.

    """
    probs = pd.Series(index=np.arange(n_support), data=0.0, dtype=float)
    probs.update(observed_values.value_counts(normalize=True))
    return np.random.choice(n_support, size=size, p=probs.values)


def draw_start_asset_dist(asset_data, n_agents_edu, method="uniform"):
    """
    Draws samples from the starting asset distribution using different methods.

    Methods:
    - "uniform": Uniform sampling between the 30th and 70th percentiles.
    - "lognormal": Fit a shifted lognormal distribution and sample from it.
    - "kde": Kernel Density Estimation (KDE) based sampling.
    - "pareto": Fit a shifted Pareto distribution and sample from it.

    Parameters:
        asset_data (pd.Series): Observed assets carried into the first period.
        n_agents_edu (int): Number of samples to draw.
        method (str): Sampling method ("uniform", "lognormal", "kde", "pareto").

    Returns:
        np.ndarray: Sampled asset values.
    """

    min_assets = asset_data.values.min()

    if method == "uniform":
        # Existing uniform sampling between 30th and 70th quantiles
        assets_start_edu = np.random.uniform(
            asset_data.quantile(0.3), asset_data.quantile(0.7), n_agents_edu
        )

    elif method == "lognormal":
        # Fit a shifted lognormal distribution
        shifted_data = asset_data - min_assets + 1e-6  # Avoid log(0)
        shape, loc, scale = stats.lognorm.fit(
            shifted_data, floc=0
        )  # Fix location at zero
        samples = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=n_agents_edu)
        assets_start_edu = samples + min_assets - 1e-6  # Shift back

    elif method == "kde":
        # Kernel Density Estimation (KDE) sampling
        kde = KernelDensity(kernel="gaussian", bandwidth=0.1 * asset_data.std()).fit(
            asset_data.values.reshape(-1, 1)
        )
        assets_start_edu = kde.sample(n_agents_edu).flatten()

    elif method == "pareto":
        # Fit a Pareto-like distribution (Shifted Pareto)
        shifted_data = asset_data - min_assets + 1e-6  # Shift data to avoid 0
        shape, loc, scale = stats.pareto.fit(
            shifted_data, floc=0
        )  # Fix location at zero
        samples = stats.pareto.rvs(shape, loc=loc, scale=scale, size=n_agents_edu)
        assets_start_edu = samples + min_assets - 1e-6  # Shift back

    else:
        raise ValueError(
            "Invalid method. Choose 'uniform', 'lognormal', 'kde', or 'pareto'."
        )

    assets_start_edu[assets_start_edu < min_assets] = min_assets

    return assets_start_edu
