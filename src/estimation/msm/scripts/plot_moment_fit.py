from matplotlib import pyplot as plt

from estimation.msm.scripts.labor_supply_moments import calc_labor_supply_choice
from estimation.msm.scripts.labor_transition_moments import calc_transition_to_work
from estimation.msm.scripts.wealth_moments import calc_wealth_moment
from export_results.figures.color_map import JET_COLOR_MAP, LINE_STYLES


def plot_moments_all_moments_for_dfs(df_list, moment_labels, specs):

    labor_supply_moment_list = []
    labor_transitions_moment_list = []
    wealth_moment_list = []
    for i, df in enumerate(df_list):
        labor_supply_moment_list += [calc_labor_supply_choice(df)]
        labor_transitions_moment_list += [calc_transition_to_work(df)]
        empirical = True if moment_labels[i] == "empirical" else False
        wealth_moment_list += [calc_wealth_moment(df, empirical=empirical)]

    plot_choice_moments(labor_supply_moment_list, moment_labels, specs)
    plot_transition_moments(labor_transitions_moment_list, moment_labels, specs)
    plot_wealth_moments(wealth_moment_list, moment_labels, specs)


def plot_choice_moments(moments_list, moment_labels, specs):

    max_val = 0.0
    for id_moment, moment in enumerate(moments_list):
        max_val = max(max_val, moment.max().max())

    max_val *= 1.1
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
                    ax.set_ylim([0, max_val])
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
    max_val = 0.0
    for id_moment, moment in enumerate(moments_list):
        max_val = max(max_val, moment.max().max())

    max_val *= 1.1

    # Labor transitions moments
    sex_vars = moments_list[0].index.get_level_values("sex").unique()
    edu_vars = moments_list[0].index.get_level_values("education").unique()

    n_types = len(sex_vars) + len(edu_vars)
    fig, axs = plt.subplots(nrows=2, ncols=n_types)

    state_label = ["unemployed", "work"]
    sex_labels = specs["sex_labels"]
    edu_labels = ["Low", "High"]

    for current_state, current_label in enumerate(state_label):
        count = 0
        for sex_var in sex_vars:
            for edu_var in edu_vars:
                ax = axs[current_state, count]
                ax.set_ylim([0, max_val])
                for id_moment, moment in enumerate(moments_list):
                    type_transition_moments = moment.loc[
                        (sex_var, edu_var, slice(None), current_state)
                    ]

                    ax.plot(
                        type_transition_moments,
                        color=JET_COLOR_MAP[edu_var],
                        ls=LINE_STYLES[id_moment % len(LINE_STYLES)],
                    )

                if current_state == 0:
                    ax.set_title(f"{sex_labels[sex_var]} - {edu_labels[edu_var]}")
                count += 1

        axs[current_state, 0].set_ylabel(f"{current_label} to work")

    ax = axs[0, 0]
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
    max_val = 0.0
    for id_moment, moment in enumerate(moments_list):
        max_val = max(max_val, moment.max().max())

    max_val *= 1.1
    # Labor transitions moments
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].set_title("Low Educated")
    axs[0, 1].set_title("High Educated")

    sex_vars = moments_list[0].index.get_level_values("sex").unique()
    edu_vars = moments_list[0].index.get_level_values("education")

    partner_labels = ["Single", "Partnered"]
    for id_partner, partner_label in enumerate(partner_labels):
        for sex_var in sex_vars:
            for edu_var in edu_vars:
                for id_moment, moment in enumerate(moments_list):
                    type_wealth_moments = moment.loc[
                        (sex_var, edu_var, id_partner, slice(None))
                    ]

                    ax = axs[id_partner, edu_var]
                    ax.set_ylim([0, max_val])
                    ax.plot(
                        type_wealth_moments,
                        color=JET_COLOR_MAP[sex_var],
                        ls=LINE_STYLES[id_moment],
                        # label=moment_labels[id_moment],
                    )

    ax = axs[0, 0]
    # Addlabels for each linestyle and moment name
    for id_moment, moment in enumerate(moments_list):
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            ax.plot(
                [],
                [],
                color=JET_COLOR_MAP[sex_var],
                ls=LINE_STYLES[id_moment],
                label=f"{moment_labels[id_moment]} - {sex_label}",
            )
    axs[0, 0].legend()
    axs[0, 0].set_ylabel(f"Mean Wealth - {partner_labels[0]}")
    axs[1, 0].set_ylabel(f"Mean Wealth - {partner_labels[1]}")
