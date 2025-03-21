import jax


def create_unobserved_state_specs(data_decision, model):
    def weight_func(**kwargs):
        # We need to weight the unobserved job offer state for each of its possible values
        # The weight function is called with job offer new beeing the unobserved state
        job_offer_new = kwargs["job_offer_new"]
        job_offer_weight = model["model_funcs"]["processed_exog_funcs"]["job_offer"](
            **kwargs
        )[job_offer_new]

        # For the informed state we use the share of this period. The period in the kwargs is the one from
        # before (see assignment below).
        this_period = kwargs["period"] + 1
        education = kwargs["education"]
        informed_share = kwargs["options"]["informed_shares"][this_period, education]
        informed_new = kwargs["informed_new"]
        informed_weight = informed_share * informed_new + (1 - informed_share) * (
            1 - informed_new
        )

        return job_offer_weight * informed_weight

    relevant_prev_period_state_choices_dict = {
        "period": data_decision["period"].values - 1,
        "education": data_decision["education"].values,
        "sex": data_decision["sex"].values,
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
