# Set paths of project
from set_paths import create_path_dict

path_dict = create_path_dict()

from export_reslts.tables.cv import calc_compensated_variation

calc_compensated_variation(path_dict)
