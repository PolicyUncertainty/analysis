# %% Set paths of project
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict()


# %% ########################################
# # Utility plots
# ##########################################
from export_results.figures.utility import plot_utility



###################
# Job offer plots
###################
from export_results.figures.job_offer_plots import plot_job_separation
plot_job_separation(path_dict)
plt.show()
plt.close("all")


# %% ########################################
# # Budget plots
# ##########################################


from export_results.figures.income_plots import plot_incomes

plot_incomes(path_dict)
plt.show()
plt.close("all")


# %% ########################################
# # SRA plots
# ##########################################

from export_results.figures.expected_SRA_plots import plot_markov_process

plot_markov_process(path_dict)
plt.show()
plt.close("all")

# %% ##########################################
# # Policy state plot
# ##########################################
from export_results.figures.policy_states import plot_SRA_2007_reform

plot_SRA_2007_reform(path_dict)
plt.show()
plt.close("all")


#
# from export_results.figures.pension_npv import plot_pension_npv_by_age
#
# plot_pension_npv_by_age(path_dict)
# plt.show()
# plt.close("all")
# %%