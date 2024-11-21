import jax


def create_unobserved_state_specs(data_decision, model):
    def weight_func(**kwargs):
        # We need to weight the unobserved job offer state for each of its possible values
        # The weight function is called with job offer new beeing the unobserved state
        job_offer = kwargs["job_offer_new"]
        job_offer_weight_unobserved = model["model_funcs"]["processed_exog_funcs"][
            "job_offer"
        ](**kwargs)[job_offer]
        # We select the unobserved weight if job offer is unobserved.
        # If it is observed we have to assign the uniform weight for the number of values the state can take
        job_offer_weight = jax.lax.select(
            kwargs["job_offer_observed_bool"], 0.5, job_offer_weight_unobserved
        )

    relevant_prev_period_state_choices_dict = {
        "period": data_decision["period"].values - 1,
        "education": data_decision["education"].values,
    }
    unobserved_state_specs = {
        "observed_bools_states": {
            "job_offer": (data_decision["job_offer"] > -1).values,
            "informed": (data_decision["informed"] > -1).values,
        },
        "weight_func": weight_func,
        "state_choices_weighing": {
            "states": relevant_prev_period_state_choices_dict,
            "choices": data_decision["lagged_choice"].values,
        },
    }
    return unobserved_state_specs
