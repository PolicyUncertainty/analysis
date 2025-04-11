# %%
# set paths and parameters
# --------------------------------------------------------------------------------------
from set_paths import create_path_dict

paths_dict = create_path_dict(define_user=True)

# Load options and generate auxiliary options
from specs.derive_specs import read_and_derive_specs

specs = read_and_derive_specs(paths_dict["specs"])
string_in = input(
    """Which dataset should be created?\n
    \n- [a]ll
    \n- [s]tructural
    \n- [w]age
    \n- [p]artner wage
    \n- [j]ob separation
    \n- [f]amily transitions
    \n- [h]ealth transition
    \n- [m]ortality
    \n- [d]isability pension
    \n- [c]redited periods for long work life pensions
    \n\n Please write the corresponding letter:"""
)

USE_PROCESSED_PL = True
LOAD_WEALTH = True
LOAD_ARTKALEN_CHOICE = False


from process_data.first_step_sample_scripts.create_credited_periods_est_sample import (
    create_credited_periods_est_sample,
)
from process_data.first_step_sample_scripts.create_disability_pension_sample import (
    create_disability_pension_sample,
)
from process_data.first_step_sample_scripts.create_health_transition_sample import (
    create_health_transition_sample,
)
from process_data.first_step_sample_scripts.create_job_sep_sample import (
    create_job_sep_sample,
)
from process_data.first_step_sample_scripts.create_partner_transition_sample import (
    create_partner_transition_sample,
)
from process_data.first_step_sample_scripts.create_partner_wage_est_sample import (
    create_partner_wage_est_sample,
)
from process_data.first_step_sample_scripts.create_survival_transition_sample import (
    create_survival_transition_sample,
)
from process_data.first_step_sample_scripts.create_wage_est_sample import (
    create_wage_est_sample,
)

# %%
# Create relevant datasets.
# --------------------------------------------------------------------------------------
from process_data.structural_sample_scripts.create_structural_est_sample import (
    create_structural_est_sample,
)

if string_in == "a" or string_in == "s":
    create_structural_est_sample(
        paths_dict,
        specs=specs,
        load_data=False,
        load_artkalen_choice=LOAD_ARTKALEN_CHOICE,
        use_processed_pl=USE_PROCESSED_PL,
        load_wealth=LOAD_WEALTH,
    )

if string_in == "a" or string_in == "w":
    create_wage_est_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "f":
    create_partner_wage_est_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "j":
    create_job_sep_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "t":
    create_partner_transition_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "h":
    create_health_transition_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "m":
    create_survival_transition_sample(paths_dict, specs=specs, load_data=False)

if string_in == "a" or string_in == "c":
    create_credited_periods_est_sample(paths_dict, load_data=False)

if string_in == "a" or string_in == "d":
    create_disability_pension_sample(paths_dict, specs=specs, load_data=False)
# %%
