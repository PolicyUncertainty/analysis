import pickle as pkl

import matplotlib.pyplot as plt

from export_results.figures.job_offer_plots import (
    plot_job_offer_transitions,
    plot_job_separations,
)
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict, load_precomputed=True)
model_name = "msm_first"

if model_name == "start":
    from estimation.struct_estimation.start_params_and_bounds.set_start_params import (
        load_and_set_start_params,
    )

    params = load_and_set_start_params(path_dict)
else:
    params = pkl.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )

plot_job_separations(path_dict, specs)
plot_job_offer_transitions(path_dict, specs, params)
plt.show()
