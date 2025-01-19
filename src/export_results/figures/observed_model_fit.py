import matplotlib.pyplot as plt
import numpy as np
from dcegm.likelihood import create_choice_prob_func_unobserved_states
from estimation.struct_estimation.estimate_setup import load_and_prep_data
from model_code.specify_model import specify_and_solve_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from model_code.unobserved_state_weighting import create_unobserved_state_specs


def observed_model_fit(paths_dict, specs, params):
    est_model, model, params = specify_and_solve_model(
        path_dict=paths_dict,
        params=params,
        update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
        policy_state_trans_func=expected_SRA_probs_estimation,
        file_append="start",
        load_model=True,
        load_solution=True,
    )

    data_decision, states_dict = load_and_prep_data_for_model_fit(
        paths_dict, specs, params, model
    )

    unobserved_state_specs = create_unobserved_state_specs(data_decision, model)

    plot_observed_model_fit_choice_probs(
        paths_dict,
        specs,
        data_decision,
        states_dict,
        model,
        unobserved_state_specs,
        params,
        est_model,
        save_folder=paths_dict["plots"],
    )


def plot_observed_model_fit_choice_probs(
    paths_dict,
    specs,
    data_decision,
    states_dict,
    model,
    unobserved_state_specs,
    params,
    est_model,
    save_folder,
):
    for choice in range(specs["n_choices"]):
        choice_vals = np.ones_like(data_decision["choice"].values) * choice

        choice_probs_observations = choice_probs_for_choice_vals(
            choice_vals=choice_vals,
            states_dict=states_dict,
            model=model,
            unobserved_state_specs=unobserved_state_specs,
            params=params,
            est_model=est_model,
            use_probability_of_observed_states=False,
        )

        choice_probs_observations = np.nan_to_num(choice_probs_observations, nan=0.0)
        data_decision[f"choice_{choice}"] = choice_probs_observations

    data_decision["choice_likelihood"] = choice_probs_for_choice_vals(
        choice_vals=data_decision["choice"].values,
        states_dict=states_dict,
        model=model,
        unobserved_state_specs=unobserved_state_specs,
        params=params,
        est_model=est_model,
        use_probability_of_observed_states=True,
    )

    # for partner_val, partner_label in enumerate(partner_labels):
    for edu in range(2):
        fig, axes = plt.subplots(2, specs["n_choices"], figsize=(10, 5))
        for sex_var, sex_label in enumerate(specs["sex_labels"]):
            data_subset = data_decision[
                (data_decision["education"] == edu) & (data_decision["sex"] == sex_var)
            ]
            choice_shares_obs = (
                data_subset.groupby(["age"])["choice"]
                .value_counts(normalize=True)
                .unstack()
            )

            labels = specs["choice_labels"]
            for choice in range(specs["n_choices"]):
                ax = axes[sex_var, choice]
                choice_shares_predicted = data_subset.groupby(["age"])[
                    f"choice_{choice}"
                ].mean()
                choice_shares_predicted.plot(ax=ax, label="Simulated")
                if not ((sex_var == 0 and (choice == 2))):
                    choice_shares_obs[choice].plot(ax=ax, label="Observed", ls="--")
                ax.set_xlabel("Age")
                ax.set_ylabel("Choice share")
                ax.set_title(f"{labels[choice]}")
                ax.set_ylim([-0.05, 1.05])
                if choice == 0:
                    ax.legend(loc="upper left")
        # Fig title
        fig.tight_layout()

        file_append = ["low", "high"]
        fig.savefig(
            save_folder + f"observed_model_fit_{file_append[edu]}.png",
            transparent=True,
            dpi=300,
        )
        # fig.suptitle(f"Choice shares {specs['education_labels'][edu]}")
    breakpoint()


def load_and_prep_data_for_model_fit(paths_dict, specs, params, model):
    data_decision, _ = load_and_prep_data(
        paths_dict, params, model, drop_retirees=False
    )
    data_decision["age"] = data_decision["period"] + specs["start_age"]
    data_decision = data_decision[data_decision["age"] < 75]
    states_dict = {
        name: data_decision[name].values.copy()
        for name in model["model_structure"]["discrete_states_names"]
    }
    states_dict["experience"] = data_decision["experience"].values
    states_dict["wealth"] = data_decision["adjusted_wealth"].values
    return data_decision, states_dict


def choice_probs_for_choice_vals(
    choice_vals,
    states_dict,
    model,
    unobserved_state_specs,
    params,
    est_model,
    use_probability_of_observed_states=False,
):
    choice_prob_func = create_choice_prob_func_unobserved_states(
        model=model,
        observed_states=states_dict,
        observed_choices=choice_vals,
        unobserved_state_specs=unobserved_state_specs,
        use_probability_of_observed_states=use_probability_of_observed_states,
    )
    choice_probs_observations = choice_prob_func(
        value_in=est_model["value"],
        endog_grid_in=est_model["endog_grid"],
        params_in=params,
    )
    return choice_probs_observations
