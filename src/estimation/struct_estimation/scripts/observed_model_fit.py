import matplotlib.pyplot as plt
import numpy as np
from dcegm.likelihood import (
    create_choice_prob_func_unobserved_states,
    create_partial_choice_prob_calculation,
)

from estimation.struct_estimation.scripts.estimate_setup import load_and_prep_data
from export_results.figures.color_map import JET_COLOR_MAP
from model_code.specify_model import specify_and_solve_model
from model_code.unobserved_state_weighting import create_unobserved_state_specs


def observed_model_fit(
    paths_dict, specs, params, model_name, load_sol_model, load_solution
):
    model_solved = specify_and_solve_model(
        path_dict=paths_dict,
        params=params,
        subj_unc=True,
        custom_resolution_age=None,
        file_append=model_name,
        load_model=load_sol_model,
        load_solution=load_solution,
        sim_specs=None,
        debug_info="all",
    )

    data_decision, states_dict = load_and_prep_data_for_model_fit(
        paths_dict=paths_dict,
        specs=specs,
        params=params,
        model_class=model_solved,
        drop_retirees=False,
    )

    unobserved_state_specs = create_unobserved_state_specs(data_decision)

    plot_observed_model_fit_choice_probs(
        specs=specs,
        data_decision=data_decision,
        states_dict=states_dict,
        model_solved=model_solved,
        unobserved_state_specs=unobserved_state_specs,
        params=params,
        save_folder=paths_dict["plots"],
    )


def plot_observed_model_fit_choice_probs(
    specs,
    data_decision,
    states_dict,
    model_solved,
    unobserved_state_specs,
    params,
    save_folder,
):
    for choice in range(specs["n_choices"]):
        choice_vals = np.ones_like(data_decision["choice"].values) * choice

        choice_probs_observations = choice_probs_for_choice_vals(
            choice_vals=choice_vals,
            states_dict=states_dict,
            model_solved=model_solved,
            unobserved_state_specs=unobserved_state_specs,
            params=params,
            use_probability_of_observed_states=False,
        )

        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        data_decision[f"choice_{choice}"] = choice_probs_observations

    choice_share_labels = ["Choice Share Men", "Choice Share Women"]
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        if sex_var == 0:
            n_choices = 3
        else:
            n_choices = 4
        fig, axes = plt.subplots(ncols=n_choices, figsize=(14, 8))
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            data_subset = data_decision[
                (data_decision["education"] == edu_var)
                & (data_decision["sex"] == sex_var)
            ]
            all_choice_shares_obs = (
                data_subset.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .unstack()
            )

            labels = specs["choice_labels"]
            for choice in range(specs["n_choices"]):

                choice_shares_predicted = data_subset.groupby(["age"])[
                    f"choice_{choice}"
                ].mean()

                # Only plot if we are not in men and part-time
                men_and_part_time = (sex_var == 0) and (choice == 2)
                if not men_and_part_time:
                    men_and_full_time = (sex_var == 0) and (choice == 3)
                    if men_and_full_time:
                        ax = axes[choice - 1]
                    else:
                        ax = axes[choice]

                    ax.plot(
                        choice_shares_predicted,
                        label=f"Pred. {edu_label}",
                        color=JET_COLOR_MAP[edu_var],
                    )
                    ax.plot(
                        all_choice_shares_obs[choice],
                        label=f"Obs. {edu_label}",
                        color=JET_COLOR_MAP[edu_var],
                        linestyle="--",
                    )

                    ax.set_ylim([-0.05, 1.05])
                    ax.set_title(f"{labels[choice]}")
                    ax.set_xlabel("Age")

        axes[1].legend(loc="upper left")
        axes[0].set_ylabel(choice_share_labels[sex_var])
        # Fig title
        fig.tight_layout()

        fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
            save_folder + f"observed_model_fit_{sex_label}.png",
            transparent=True,
            dpi=300,
        )

    # Create a second figure just for the retirement phase of 60 to 75 years
    ages = np.arange(60, 72)
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(14, 8))
    for sex_var, sex_label in enumerate(specs["sex_labels"]):
        for edu_var, edu_label in enumerate(specs["education_labels"]):
            data_subset = data_decision[
                (data_decision["education"] == edu_var)
                & (data_decision["sex"] == sex_var)
                & (data_decision["lagged_choice"] != 0)
            ]
            ret_choice_shares_obs = (
                data_subset.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .loc[(ages, 0)]
            )
            ret_choice_shares_est = (
                data_subset.groupby(["age"])[f"choice_0"].mean().loc[ages]
            )

            # Plotting the retirement choice shares as bars
            ax = axs[sex_var, edu_var]
            ax.bar(
                ages - 0.4,
                ret_choice_shares_obs,
                color=JET_COLOR_MAP[0],
                label="Observed",
                width=0.4,
            )
            ax.bar(
                ages,
                ret_choice_shares_est,
                color=JET_COLOR_MAP[1],
                label="Predicted",
                width=0.4,
            )

    axs[0, 1].legend(loc="upper left")

    # Make x ticks invisible in the first row
    axs[0, 0].set_xticks([])
    axs[0, 1].set_xticks([])
    axs[1, 0].set_xlabel("Age")
    axs[1, 1].set_xlabel("Age")

    axs[0, 0].set_title("Low Education")
    axs[0, 1].set_title("High Education")
    axs[0, 0].set_ylabel("Choice Share Men")
    axs[1, 0].set_ylabel("Choice Share Women")
    fig.tight_layout()
    fig.savefig(  # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
        save_folder + f"retirement_fit.png",
        transparent=True,
        dpi=300,
    )


def load_and_prep_data_for_model_fit(
    paths_dict, specs, params, model_class, drop_retirees=False
):
    data_decision, _ = load_and_prep_data(
        path_dict=paths_dict,
        start_params=params,
        model_class=model_class,
        drop_retirees=drop_retirees,
    )
    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_decision = data_decision[data_decision["age"] < 75]
    states_dict = {
        name: data_decision[name].values.copy()
        for name in model_class.model_structure["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["assets_begin_of_period"] = data_decision[
        "assets_begin_of_period"
    ].values
    return data_decision, states_dict


def choice_probs_for_choice_vals(
    choice_vals,
    states_dict,
    model_solved,
    params,
    unobserved_state_specs=None,
    use_probability_of_observed_states=False,
):
    if unobserved_state_specs is None:
        choice_prob_func = create_partial_choice_prob_calculation(
            observed_states=states_dict,
            observed_choices=choice_vals,
            model_structure=model_solved.model_structure,
            model_config=model_solved.model_config,
            model_funcs=model_solved.model_funcs,
        )
    else:
        choice_prob_func = create_choice_prob_func_unobserved_states(
            model_structure=model_solved.model_structure,
            model_specs=model_solved.model_specs,
            model_funcs=model_solved.model_funcs,
            model_config=model_solved.model_config,
            observed_states=states_dict,
            observed_choices=choice_vals,
            unobserved_state_specs=unobserved_state_specs,
            use_probability_of_observed_states=use_probability_of_observed_states,
        )

    choice_probs_observations = choice_prob_func(
        value_in=model_solved.value,
        endog_grid_in=model_solved.endog_grid,
        params_in=params,
    )
    return choice_probs_observations
