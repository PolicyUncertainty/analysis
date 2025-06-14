from matplotlib import pyplot as plt

from estimation.msm.scripts.calc_moments import (
    calc_labor_supply_choice,
    calc_labor_transitions_by_age_bins,
    calc_median_wealth_by_age,
)
from export_results.figures.color_map import JET_COLOR_MAP, LINE_STYLES


def plot_moments_all_moments_for_dfs(data_decision, data_sim, specs):

    empirical_choice_moments = calc_labor_supply_choice(data_decision)
    simulated_choice_moments = calc_labor_supply_choice(data_sim)
    choice_moments = [empirical_choice_moments, simulated_choice_moments]

    moment_labels = ["Empirical", "Simulated"]
    plot_choice_moments(choice_moments, moment_labels, specs)

    empirical_transition_moments = calc_labor_transitions_by_age_bins(data_decision)
    simulated_transition_moments = calc_labor_transitions_by_age_bins(data_sim)
    transition_moments = [empirical_transition_moments, simulated_transition_moments]
    plot_transition_moments(transition_moments, moment_labels, specs)

    empirical_wealth_moments = calc_median_wealth_by_age(data_decision)
    simulated_wealth_moments = calc_median_wealth_by_age(data_sim)
    wealth_moments = [empirical_wealth_moments, simulated_wealth_moments]
    plot_wealth_moments(wealth_moments, specs)


def plot_choice_moments(moments_list, moment_labels, specs):

    # Choice moments
    fig, axs = plt.subplots(nrows=4, ncols=2)
    axs[0, 0].set_title("Men")
    axs[0, 1].set_title("Women")

    sex_vars = moments_list[0].index.get_level_values("sex").unique()
    edu_vars = moments_list[0].index.get_level_values("education")

    for sex_var in sex_vars:
        for choice_var, choice_label in enumerate(specs["choice_labels"]):
            for edu_var in edu_vars:
                for id_moment, moment in enumerate(moments_list):
                    type_choice_moments = moment.loc[
                        (sex_var, edu_var, slice(None), choice_var)
                    ]

                    if (sex_var == 0) & (choice_var == 2):
                        continue
                    ax = axs[choice_var, sex_var]
                    ax.plot(
                        type_choice_moments,
                        color=JET_COLOR_MAP[edu_var],
                        ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                    )

                    if sex_var == 0:
                        # Set y label
                        ax.set_ylabel(f"Share {choice_label}")

        ax = axs[0, sex_var]
        # Addlabels for each linestyle and moment name
        for id_moment, moment in enumerate(moments_list):
            ax.plot(
                [],
                [],
                color=JET_COLOR_MAP[edu_vars[0]],
                ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                label=moment_labels[id_moment],
            )
        ax.legend()


def plot_transition_moments(moments_list, moment_labels, specs):
    # Labor transitions moments
    fig, axs = plt.subplots(nrows=4, ncols=2)
    axs[0, 0].set_title("Men")
    axs[0, 1].set_title("Women")

    sex_vars = moments_list[0].index.get_level_values("sex").unique()
    edu_vars = moments_list[0].index.get_level_values("education")

    state_label = ["unemployed", "work"]

    for sex_var in sex_vars:
        count = 0
        for current_state, current_label in enumerate(state_label):
            for next_state, next_label in enumerate(state_label):
                for edu_var in edu_vars:
                    for id_moment, moment in enumerate(moments_list):
                        type_transition_moments = moment.loc[
                            (sex_var, edu_var, slice(None), current_state, next_state)
                        ]

                        ax = axs[count, sex_var]
                        ax.plot(
                            type_transition_moments,
                            color=JET_COLOR_MAP[edu_var],
                            ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                        )

                        if sex_var == 0:
                            # Set y label
                            ax.set_ylabel(f"{current_label} to {next_label}")

                count += 1
        ax = axs[0, sex_var]
        # Addlabels for each linestyle and moment name
        for id_moment, moment in enumerate(moments_list):
            ax.plot(
                [],
                [],
                color=JET_COLOR_MAP[edu_vars[0]],
                ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                label=moment_labels[id_moment],
            )
        ax.legend()


def plot_wealth_moments(moments_list, moment_labels, specs):

    # Labor transitions moments
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].set_title("Men")
    axs[1].set_title("Women")

    sex_vars = moments_list[0].index.get_level_values("sex").unique()
    edu_vars = moments_list[0].index.get_level_values("education")

    for sex_var in sex_vars:
        for edu_var in edu_vars:
            for id_moment, moment in enumerate(moments_list):
                type_wealth_moments = moment.loc[(sex_var, edu_var, slice(None))]

                ax = axs[sex_var]
                ax.plot(
                    type_wealth_moments,
                    color=JET_COLOR_MAP[edu_var],
                    ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                )

                if sex_var == 0:
                    # Set y label
                    ax.set_ylabel(f"Median Wealth")

        ax = axs[sex_var]
        # Addlabels for each linestyle and moment name
        for id_moment, moment in enumerate(moments_list):
            ax.plot(
                [],
                [],
                color=JET_COLOR_MAP[edu_vars[0]],
                ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                label=moment_labels[id_moment],
            )
        ax.legend()
