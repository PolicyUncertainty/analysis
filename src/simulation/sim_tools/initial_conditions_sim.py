import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from dcegm.wealth_correction import adjust_observed_wealth
from sklearn.neighbors import KernelDensity

from model_code.stochastic_processes.job_offers import job_offer_process_transition


def generate_start_states(
    path_dict, params, model, inital_SRA, n_agents, seed, only_informed=False
):
    specs = model["options"]["model_params"]

    observed_data = pd.read_csv(path_dict["struct_est_sample"])

    np.random.seed(seed)
    # Define start data and adjust wealth
    min_period = observed_data["period"].min()
    start_period_data = observed_data[observed_data["period"].isin([min_period])].copy()

    states_dict = {
        name: start_period_data[name].values
        for name in model["model_structure"]["discrete_states_names"]
    }
    states_dict["wealth"] = start_period_data["wealth"].values / specs["wealth_unit"]
    states_dict["experience"] = start_period_data["experience"].values

    start_period_data.loc[:, "adjusted_wealth"] = adjust_observed_wealth(
        observed_states_dict=states_dict,
        params=params,
        model=model,
    )

    # # Define unique values
    # education_levels = start_period_data["education"].unique()
    # sexes = start_period_data["sex"].unique()
    # periods = sorted(start_period_data["period"].unique())

    # # Compute 99th percentile threshold for adjusted_wealth
    # wealth_99 = np.percentile(start_period_data["adjusted_wealth"], 99)

    # # Filter out top 1%
    # filtered_data = start_period_data[start_period_data["adjusted_wealth"] <= wealth_99]

    # # Create a figure with 4 subplots (one for each period)
    # fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)

    # # Plot distributions separately for each period
    # for period, ax in zip(periods, axes.flatten()):
    #     subset = filtered_data[filtered_data["period"] == period]

    #     for edu in education_levels:
    #         for sex in sexes:
    #             sub_subset = subset[(subset["education"] == edu) & (subset["sex"] == sex)]

    #             if not sub_subset.empty:
    #                 sns.histplot(
    #                     data=sub_subset,
    #                     x="adjusted_wealth",
    #                     bins=30,
    #                     ax=ax,
    #                     alpha=0.3,  # Translucent bars
    #                     kde=True,  # Add KDE
    #                     label=f"Edu: {edu}, Sex: {sex}"
    #                 )

    #     ax.set_title(f"Period {period}")
    #     ax.set_xlabel("Adjusted Wealth")
    #     ax.set_ylabel("Density / Count")
    #     ax.legend(title="Groups")

    # # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()

    # breakpoint()

    # Generate container
    sex_agents = np.array([], np.uint8)
    education_agents = np.array([], np.uint8)
    for sex_var in range(specs["n_sexes"]):
        if specs["n_sexes"] > 1:
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
    wealth_agents = np.empty(n_agents, np.float64)
    informed_agents = np.zeros(n_agents, np.uint8)
    exp_agents = np.empty(n_agents, np.float64)
    lagged_choice = np.empty(n_agents, np.uint8)
    partner_states = np.empty(n_agents, np.uint8)
    health_agents = np.empty(n_agents, np.uint8)
    job_offer_agents = np.empty(n_agents, np.uint8)

    for sex_var in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            type_mask = (sex_agents == sex_var) & (education_agents == edu)
            start_period_data_edu = start_period_data[
                (start_period_data["sex"] == sex_var)
                & (start_period_data["education"] == edu)
            ]

            n_agents_edu = np.sum(type_mask)

            # Restrict dataset on education level

            wealth_start_edu = draw_start_wealth_dist(
                start_period_data_edu, n_agents_edu
            )
            wealth_agents[type_mask] = wealth_start_edu

            # Generate edu specific informed shares
            informed_share_edu = specs["initial_informed_shares"][edu]
            # Draw informed states according to inital distribution
            dist = np.array([1 - informed_share_edu, informed_share_edu])
            informed_draws_edu = np.random.choice(2, n_agents_edu, p=dist)
            informed_agents[type_mask] = informed_draws_edu

            # Generate type specific initial experience distribution
            exp_max_edu = start_period_data_edu["experience"].max()
            empirical_exp_probs = start_period_data_edu["experience"].value_counts(
                normalize=True
            )
            exp_probs = pd.Series(
                index=np.arange(0, exp_max_edu + 1), data=0, dtype=float
            )
            exp_probs.update(empirical_exp_probs)
            exp_agents[type_mask] = np.random.choice(
                exp_max_edu + 1, size=n_agents_edu, p=exp_probs.values
            )

            # Generate type specific initial lagged choice distribution
            empirical_lagged_choice_probs = start_period_data_edu[
                "lagged_choice"
            ].value_counts(normalize=True)
            lagged_choice_probs = pd.Series(
                index=np.arange(0, specs["n_choices"]), data=0, dtype=float
            )
            lagged_choice_probs.update(empirical_lagged_choice_probs)
            lagged_choice_edu = np.random.choice(
                specs["n_choices"], size=n_agents_edu, p=lagged_choice_probs.values
            )
            lagged_choice[type_mask] = lagged_choice_edu

            # Generate job offer probabilities
            job_offer_probs = job_offer_process_transition(
                params=params,
                sex=jnp.ones_like(lagged_choice_edu) * sex_var,
                options=specs,
                education=jnp.ones_like(lagged_choice_edu) * edu,
                period=jnp.zeros_like(lagged_choice_edu),
                choice=lagged_choice_edu,
            ).T
            # Job offer probs is n_agents x 2. Choose for each row the job offer state
            # with np random choice
            job_offer_edu = np.array(
                [np.random.choice(a=len(p), p=p) for p in job_offer_probs]
            )
            job_offer_agents[type_mask] = job_offer_edu

            # Get type specific partner states
            empirical_partner_probs = start_period_data_edu[
                "partner_state"
            ].value_counts(normalize=True)
            partner_probs = pd.Series(
                index=np.arange(specs["n_partner_states"]), data=0, dtype=float
            )
            partner_probs.update(empirical_partner_probs)
            partner_states_edu = np.random.choice(
                specs["n_partner_states"], size=n_agents_edu, p=partner_probs.values
            )
            partner_states[type_mask] = partner_states_edu

            # Generate health states
            empirical_health_probs = start_period_data_edu[
                "surveyed_health"
            ].value_counts(normalize=True)
            # We let people start only in good and bad health
            health_probs = pd.Series(index=np.arange(2), data=0, dtype=float)
            health_probs.update(empirical_health_probs)
            health_states_edu = np.random.choice(
                2, size=n_agents_edu, p=health_probs.values
            )
            health_agents[type_mask] = health_states_edu

    # Transform it to be between 0 and 1
    exp_agents /= specs["max_exp_diffs_per_period"][0]

    # Set lagged choice to 1(unemployment) if experience is 0
    exp_zero_mask = exp_agents == 0
    lagged_choice[exp_zero_mask] = 1

    # Generate start policy state from initial SRA
    initial_policy_state = np.floor(
        (inital_SRA - specs["min_SRA"]) / specs["SRA_grid_size"]
    )

    policy_state_agents = (jnp.ones_like(exp_agents) * initial_policy_state).astype(
        jnp.uint8
    )

    if only_informed:
        informed_agents = jnp.ones_like(informed_agents)

    states = {
        "period": jnp.zeros_like(exp_agents, dtype=jnp.uint8),
        "experience": jnp.array(exp_agents, dtype=jnp.float64),
        "sex": jnp.array(sex_agents, dtype=jnp.uint8),
        "health": jnp.array(health_agents, dtype=jnp.uint8),
        "education": jnp.array(education_agents, dtype=jnp.uint8),
        "informed": jnp.array(informed_agents, dtype=jnp.uint8),
        "lagged_choice": jnp.array(lagged_choice, dtype=jnp.uint8),
        "policy_state": policy_state_agents,
        "job_offer": jnp.array(job_offer_agents, dtype=jnp.uint8),
        "partner_state": jnp.array(partner_states, dtype=jnp.uint8),
    }
    return states, wealth_agents


def draw_start_wealth_dist(start_period_data_edu, n_agents_edu, method="uniform"):
    """
    Draws samples from the starting wealth distribution using different methods.

    Methods:
    - "uniform": Uniform sampling between the 30th and 70th percentiles.
    - "lognormal": Fit a shifted lognormal distribution and sample from it.
    - "kde": Kernel Density Estimation (KDE) based sampling.
    - "pareto": Fit a shifted Pareto distribution and sample from it.

    Parameters:
        start_period_data_edu (pd.DataFrame): Data containing "adjusted_wealth".
        n_agents_edu (int): Number of samples to draw.
        method (str): Sampling method ("uniform", "lognormal", "kde", "pareto").

    Returns:
        np.ndarray: Sampled wealth values.
    """

    wealth_data = start_period_data_edu["adjusted_wealth"]

    if method == "uniform":
        # Existing uniform sampling between 30th and 70th quantiles
        wealth_start_edu = np.random.uniform(
            wealth_data.quantile(0.3), wealth_data.quantile(0.7), n_agents_edu
        )

    elif method == "lognormal":
        # Fit a shifted lognormal distribution
        min_val = wealth_data.min()
        shifted_data = wealth_data - min_val + 1e-6  # Avoid log(0)
        shape, loc, scale = stats.lognorm.fit(
            shifted_data, floc=0
        )  # Fix location at zero
        samples = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=n_agents_edu)
        wealth_start_edu = samples + min_val - 1e-6  # Shift back

    elif method == "kde":
        # Kernel Density Estimation (KDE) sampling
        kde = KernelDensity(kernel="gaussian", bandwidth=0.1 * wealth_data.std()).fit(
            wealth_data.values.reshape(-1, 1)
        )
        wealth_start_edu = kde.sample(n_agents_edu).flatten()

    elif method == "pareto":
        # Fit a Pareto-like distribution (Shifted Pareto)
        min_val = wealth_data.min()
        shifted_data = wealth_data - min_val + 1e-6  # Shift data to avoid 0
        shape, loc, scale = stats.pareto.fit(
            shifted_data, floc=0
        )  # Fix location at zero
        samples = stats.pareto.rvs(shape, loc=loc, scale=scale, size=n_agents_edu)
        wealth_start_edu = samples + min_val - 1e-6  # Shift back

    else:
        raise ValueError(
            "Invalid method. Choose 'uniform', 'lognormal', 'kde', or 'pareto'."
        )

    return wealth_start_edu
