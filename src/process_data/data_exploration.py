# %%
# import libraries, load data

from pathlib import Path

import numpy as np
import pandas as pd

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
file_dir_path = str(Path(__file__).resolve().parents[0]) + "/"
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")


from set_paths import create_path_dict
USER = "b"
paths_dict = create_path_dict(USER, analysis_path)

data_decision = pd.read_pickle(analysis_path + "output/decision_data.pkl")
# %%
# generate age
data_decision["age"] = data_decision["period"] + 30

# load soep
core_data = pd.read_stata(
    f"{soep_c38}/pgen.dta",
    columns=["syear", "pid", "hid", "pgemplst", "pgexpft", "pgstib"],
    convert_categoricals=False,
)
pathl_data = pd.read_stata(
    f"{soep_c38}/ppathl.dta",
    columns=["pid", "hid", "syear", "sex", "gebjahr", "rv_id"],
    convert_categoricals=False,
)

# Merge core data with pathl data
merged_data = pd.merge(
    core_data, pathl_data, on=["pid", "hid", "syear"], how="inner"
)

# %%
# plot counts of observations by age
data_decision["age"].plot.hist(bins=100)

# %%
# group decision data by period, plot bar chart of choice shares (adding up to 100%) for every period
# x-axis: show only periods with 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
age_range = np.arange(0, 80, 1)
ax = data_decision.groupby("age")["choice"].value_counts(normalize=True).loc[age_range].unstack().plot(kind="bar", stacked=True)
#ax.set_xticks(np.arange(0, 10, 1))



# %%
