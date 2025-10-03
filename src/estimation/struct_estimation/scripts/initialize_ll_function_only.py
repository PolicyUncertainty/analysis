from estimation.struct_estimation.scripts.estimate_setup import (
    est_class_from_paths,
    generate_print_func,
)
from specs.derive_specs import generate_derived_and_data_derived_specs


def initialize_est_class(
    path_dict,
    params_to_estimate_names,
    file_append,
    load_model,
    start_params_all,  #
    use_weights=True,
    save_results=True,
    print_men_examples=True,
    print_women_examples=False,
    use_observed_data=True,
    sim_data=None,
    old_only=False,
    sex_type="all",
    edu_type="all",
    util_type="add",
):

    specs = generate_derived_and_data_derived_specs(path_dict)

    print_function = generate_print_func(
        params_to_estimate_names,
        specs,
        print_men_examples=print_men_examples,
        print_women_examples=print_women_examples,
    )
    # Initialize estimation class
    est_class = est_class_from_paths(
        path_dict=path_dict,
        specs=specs,
        start_params_all=start_params_all,
        print_function=print_function,
        file_append=file_append,
        load_model=load_model,
        use_weights=use_weights,
        save_results=save_results,
        print_men_examples=print_men_examples,
        print_women_examples=print_women_examples,
        use_observed_data=use_observed_data,
        old_only=old_only,
        sim_data=sim_data,
        slow_version=False,
        sex_type=sex_type,
        edu_type=edu_type,
        util_type=util_type,
    )
    return est_class
