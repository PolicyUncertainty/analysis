# %% Set paths of project
import matplotlib.pyplot as plt
import pandas as pd

from set_paths import create_path_dict
from simulation.figures.announcment_timing import announcement_timing_lc_plot
from simulation.figures.commitment import commitment_lc_plot
from simulation.figures.debias import debias_lc_plot
from simulation.figures.sra_increase import sra_increase_aggregate_plot

path_dict = create_path_dict()

model_name = "disability"

# # Exclude the first rpw
# sra_increase_aggregate_plot(path_dict, model_name)
announcement_timing_lc_plot(path_dict, model_name)
# debias_lc_plot(path_dict, model_name)
# commitment_lc_plot(path_dict, model_name)

plt.show()
