from set_paths import create_path_dict

paths_dict = create_path_dict()

from first_step_estimation.tables.children import create_nb_children_params_latex_table
from first_step_estimation.tables.health import (
    create_health_transition_params_latex_table,
)
from first_step_estimation.tables.job_separation import (
    create_job_sep_params_latex_table,
)
from first_step_estimation.tables.mortality import create_mortality_params_latex_table
from first_step_estimation.tables.partner_transition import (
    create_partner_transition_params_latex_table,
)
from first_step_estimation.tables.partner_wage import (
    create_partner_wage_params_latex_table,
)
from first_step_estimation.tables.wage_table import create_wage_params_latex_table
from specs.derive_specs import generate_derived_and_data_derived_specs

specs = generate_derived_and_data_derived_specs(paths_dict)

create_wage_params_latex_table(paths_dict)

create_partner_wage_params_latex_table(paths_dict)

create_job_sep_params_latex_table(paths_dict)
create_nb_children_params_latex_table(paths_dict, specs)

create_partner_transition_params_latex_table(paths_dict, specs)
create_health_transition_params_latex_table(paths_dict, specs)
create_mortality_params_latex_table(paths_dict, specs)
