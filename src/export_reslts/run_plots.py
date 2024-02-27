# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")
import matplotlib.pyplot as plt
from set_paths import create_path_dict

path_dict = create_path_dict(analysis_path)

from export_reslts.figures.policy_state_trajectories import trajectory_plot

trajectory_plot(path_dict)

from export_reslts.figures.sim_1_plots import (
    plot_average_savings,
    plot_full_time,
    plot_values_by_age,
)

plot_average_savings(path_dict)
plot_full_time(path_dict)
plot_values_by_age(path_dict)

# from export_reslts.figures.plot_wage import plot_wages
#
# plot_wages(path_dict)
plt.show()
