def create_unobserved_state_specs(data_decision, model):
    def weight_func(**kwargs):
        # We need to weight the unobserved job offer state for each of its possible values
        # The weight function is called with job offer new beeing the unobserved state
        job_offer = kwargs["job_offer_new"]
        return model["model_funcs"]["processed_exog_funcs"]["job_offer"](**kwargs)[
            job_offer
        ]

    relevant_prev_period_state_choices_dict = {
        "period": data_decision["period"].values - 1,
        "education": data_decision["education"].values,
    }
    unobserved_state_specs = {
        "observed_bool": data_decision["full_observed_state"].values,
        "weight_func": weight_func,
        "states": ["job_offer"],
        "pre_period_states": relevant_prev_period_state_choices_dict,
        "pre_period_choices": data_decision["lagged_choice"].values,
    }
    return unobserved_state_specs
