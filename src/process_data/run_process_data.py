# %%
# set paths and parameters
# --------------------------------------------------------------------------------------
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=True)

# Load options and generate auxiliary options
from specs.derive_specs import read_and_derive_specs

specs = read_and_derive_specs(paths_dict["specs"])
string_in = input(
    """Which dataset should be created?
    \n\n- [a]ll
    \n- [s]tructural
    \n- [w]age
    \n- wage [p]artner
    \n- [j]ob separation
    \n- partner [t]ransition
    \n\n Please write the corresponding letter:"""
)


# %%
# Create relevant datasets.
# --------------------------------------------------------------------------------------
from process_data.create_structural_est_sample import create_structural_est_sample
from process_data.create_wage_est_sample import create_wage_est_sample
from process_data.create_partner_wage_est_sample import create_partner_wage_est_sample
from process_data.create_job_sep_sample import create_job_sep_sample
from process_data.create_partner_transition_sample import (
    create_partner_transition_sample,
)


if string_in == "a" or string_in == "s":
    create_structural_est_sample(paths_dict, options=specs, load_data=False)

if string_in == "a" or string_in == "w":
    create_wage_est_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "p":
    create_partner_wage_est_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "j":
    create_job_sep_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "t":
    create_partner_transition_sample(paths_dict, specs=specs, load_data=False)
# %%
