import matplotlib.pyplot as plt
from set_paths import create_path_dict
from specs.derive_specs import read_and_derive_specs

paths_dict = create_path_dict(define_user=True)
specs = read_and_derive_specs(paths_dict["specs"])

# Ch 2 Inst Background and Data -------------------------------------------------------------------

# chart: pension contribution and payout rates
from export_results.figures.pension_rates import plot_pension_rates
plot_pension_rates(paths_dict)
plt.savefig(paths_dict["paper_plots"] + "pension_rates.png", transparent=True, dpi=300)
#plt.show()

# chart: 2007 reform
from export_results.figures.policy_states import plot_SRA_2007_reform
plot_SRA_2007_reform(paths_dict)
plt.savefig(paths_dict["paper_plots"] + "SRA_2007_reform.png", transparent=True, dpi=300)
#plt.show()

# table: datasets
from export_results.tables.describe_datasets import create_table_describing_datasets
df_description = create_table_describing_datasets(paths_dict, specs)
#print(df_description)
df_description.to_latex(paths_dict["paper_tables"] + "datasets.tex", index=False)
