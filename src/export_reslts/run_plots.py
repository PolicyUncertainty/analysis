# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")

from set_paths import create_path_dict

path_dict = create_path_dict(analysis_path)

from export_reslts.figures.policy_state_trajectories import trajectory_plot

trajectory_plot(path_dict)
