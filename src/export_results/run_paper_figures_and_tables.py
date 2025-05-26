import matplotlib.pyplot as plt

from set_paths import create_path_dict, set_standard_matplotlib_specs
from specs.derive_specs import read_and_derive_specs

paths_dict = create_path_dict(define_user=True)
specs = read_and_derive_specs(paths_dict["specs"])

# Ch 2 Inst Background and Data -------------------------------------------------------------------

# chart: pension contribution and payout rates
# from export_results.figures.pension_rates import plot_pension_rates
# plot_pension_rates(paths_dict)
# plt.savefig(paths_dict["paper_plots"] + "pension_rates.png", transparent=True, dpi=300)

# chart: 2007 reform
from export_results.figures.policy_states import plot_SRA_2007_reform

plot_SRA_2007_reform(paths_dict)
plt.savefig(
    paths_dict["paper_plots"] + "SRA_2007_reform.png", transparent=True, dpi=300
)

# # table: datasets
# from export_results.tables.describe_datasets import create_table_describing_datasets
# df_description = create_table_describing_datasets(paths_dict, specs)
# df_description.to_latex(paths_dict["paper_tables"] + "datasets.tex", index=False)
# df_description = create_table_describing_datasets(paths_dict, specs, main=False)
# df_description.to_latex(paths_dict["paper_tables"] + "auxiliary_datasets.tex", index=False)

# Ch 3 Policy Beliefs  ---------------------------------------------------------------------------------------

# chart: SRA beliefs by cohort
plt.close("all")
from export_results.figures.expected_policy_plots import plot_sra_beliefs_by_cohort

plot_sra_beliefs_by_cohort(paths_dict)
plt.savefig(
    paths_dict["paper_plots"] + "sra_beliefs_by_cohort.png", transparent=True, dpi=300
)

# chart: ERP beliefs by cohort
plt.close("all")
from export_results.figures.expected_policy_plots import plot_erp_beliefs_by_cohort

plot_erp_beliefs_by_cohort(paths_dict)
plt.savefig(
    paths_dict["paper_plots"] + "erp_beliefs_by_cohort.png", transparent=True, dpi=300
)

# chart: example SRA evolution of expectation
from export_results.figures.expected_policy_plots import plot_example_sra_evolution

plot_example_sra_evolution(
    alpha=0.05, sigma_sq=0.05, alpha_star=0.0, SRA_30=67, resolution_age=70
)
plt.savefig(
    paths_dict["paper_plots"] + "example_sra_evolution_no_increase.png",
    transparent=True,
    dpi=300,
)

plot_example_sra_evolution(
    alpha=0.05, sigma_sq=0.05, alpha_star=0.05, SRA_30=67, resolution_age=70
)
plt.savefig(
    paths_dict["paper_plots"] + "example_sra_evolution_correct.png",
    transparent=True,
    dpi=300,
)
