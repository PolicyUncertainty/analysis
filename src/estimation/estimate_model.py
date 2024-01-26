import os
import pandas as pd
analysis_path = os.path.abspath(os.getcwd() + "/../../") + "/"

import sys
sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")


data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")

from model_code.specify_model import specify_model

model = specify_model()

breakpoint()