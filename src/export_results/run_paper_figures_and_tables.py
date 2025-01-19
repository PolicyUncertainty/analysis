import pickle
import matplotlib.pyplot as plt
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

# Ch 2 Inst Background and Data

# chart: 2007 reform
from export_results.figures.policy_states import plot_SRA_2007_reform
path_dict = create_path_dict()
plot_SRA_2007_reform(path_dict)
plt.savefig(path_dict["paper_plots"] + "SRA_2007_reform.png", transparent=True, dpi=300)
plt.show()

# table: datasets