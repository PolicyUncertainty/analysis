import jax

from model_code.stochastic_processes.health_transition import (
    calc_disability_probability,
)


def create_unobserved_state_specs(data_decision, model_class):
    model_funcs = model_class.model_funcs

    def weight_func(**kwargs):

        model_specs = kwargs["model_specs"]
        # We need to weight the unobserved job offer state for each of its possible values
        # The weight function is called with job offer new being the unobserved state
        job_offer_new = kwargs["job_offer_new"]
        # breakpoint()
        job_offer_weight = model_funcs["processed_stochastic_funcs"]["job_offer"](
            **kwargs
        )[job_offer_new]

        # For the informed state we use the share of this period. The period in the kwargs is the one from
        # before (see assignment below).
        current_age = model_specs["start_age"] + kwargs["period"] + 1
        informed_share = model_specs["informed_shares_in_ages"][
            current_age, kwargs["education"]
        ]
        informed_new = kwargs["informed_new"]
        informed_weight = informed_share * informed_new + (1 - informed_share) * (
            1 - informed_new
        )

        # Health is unobserved if it is either just bad or even disabled.
        # We choose a weight of 1 if it is observed. Start with calculating disability prob
        # and then select correct case
        disability_prob = calc_disability_probability(
            params=kwargs["params"],
            sex=kwargs["sex"],
            period=kwargs["period"],
            education=kwargs["education"],
            model_specs=kwargs["model_specs"],
        )
        disabled = kwargs["health_new"] == 2
        bad_health = kwargs["health_new"] == 1
        # If the health is observed, we set the weight to 1
        health_weight = jax.lax.select(disabled, on_true=disability_prob, on_false=1.0)
        health_weight = jax.lax.select(
            bad_health, on_true=1 - disability_prob, on_false=health_weight
        )

        return job_offer_weight * informed_weight * health_weight

    relevant_prev_period_state_choices_dict = {
        "period": data_decision["period"].values - 1,
        "education": data_decision["education"].values,
        "sex": data_decision["sex"].values,
    }

    unobserved_state_specs = {
        "observed_bools_states": {
            "job_offer": (data_decision["job_offer"] > -1).values,
            "informed": (data_decision["informed"] > -1).values,
            "health": (data_decision["health"] > -1).values,
        },
        "weight_func": weight_func,
        "state_choices_weighing": {
            "states": relevant_prev_period_state_choices_dict,
            "choices": data_decision["lagged_choice"].values,
        },
        # Bad health is unobserved if it is either just bad or even disabled.
        "custom_unobserved_states": {
            "health": [1, 2],
        },
    }
    return unobserved_state_specs
