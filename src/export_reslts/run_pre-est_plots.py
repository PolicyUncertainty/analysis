# %% Set paths of project
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()


# %% ########################################
# # Utility plots
# ##########################################
from export_reslts.figures.utility import plot_utility


# %% ########################################
# # Budget plots
# ##########################################


from export_reslts.figures.income_plots import plot_incomes

plot_incomes(path_dict)
plt.show()
plt.close("all")


# %% ########################################
# # SRA plots
# ##########################################

from export_reslts.figures.expected_SRA_plots import plot_markov_process

plot_markov_process(path_dict)
plt.show()
plt.close("all")

# %% ##########################################
# # Policy state plot
# ##########################################
from export_reslts.figures.policy_states import plot_SRA_2007_reform

plot_SRA_2007_reform(path_dict)
plt.show()
plt.close("all")


#
# from export_reslts.figures.pension_npv import plot_pension_npv_by_age
#
# plot_pension_npv_by_age(path_dict)
# plt.show()
# plt.close("all")
# %%
