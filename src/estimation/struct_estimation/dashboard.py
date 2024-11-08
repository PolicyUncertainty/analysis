import pickle as pkl

import matplotlib.pyplot as plt
from estimation.struct_estimation.estimate_setup import create_unobserved_state_specs
from estimation.struct_estimation.start_params.set_start_params import (
    load_and_set_start_params,
)
from export_results.figures.observed_model_fit import load_and_prep_data_for_model_fit
from export_results.figures.observed_model_fit import (
    plot_observed_model_fit_choice_probs,
)
from model_code.specify_model import specify_model
from model_code.stochastic_processes.policy_states_belief import (
    expected_SRA_probs_estimation,
)
from model_code.stochastic_processes.policy_states_belief import (
    update_specs_exp_ret_age_trans_mat,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

paths_dict = create_path_dict(define_user=False)
specs = generate_derived_and_data_derived_specs(paths_dict)
start_params_all = load_and_set_start_params(paths_dict)
start_params_all["bequest_scale"] = 1

# Generate model_specs
model, start_params_all = specify_model(
    path_dict=paths_dict,
    update_spec_for_policy_state=update_specs_exp_ret_age_trans_mat,
    policy_state_trans_func=expected_SRA_probs_estimation,
    params=start_params_all,
    load_model=True,
)

data_decision, states_dict = load_and_prep_data_for_model_fit(
    paths_dict, specs, start_params_all, model
)

unobserved_state_specs = create_unobserved_state_specs(data_decision, model)
log_object = pkl.load(
    open(paths_dict["intermediate_est_data"] + "solving_log.pkl", "rb")
)
print(log_object["params"])
print(log_object["ll_value"])

start_params_all.update(log_object["params"])
plot_observed_model_fit_choice_probs(
    paths_dict,
    specs,
    data_decision,
    states_dict,
    model,
    unobserved_state_specs,
    start_params_all,
    log_object["model_sol"],
    save_folder=paths_dict["intermediate_est_data"],
)
plt.show()
