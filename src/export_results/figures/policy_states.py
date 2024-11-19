import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process_data.structural_sample_scripts.policy_state import (
    assign_policy_state_by_gebjahr,
)


def plot_SRA_2007_reform(path_dict):
    gebjahr = pd.Series(data=np.arange(1945, 1966, 1), name="gebjahr")
    policy_states = assign_policy_state_by_gebjahr(gebjahr)
    fig, ax = plt.subplots()
    ax.plot(gebjahr, policy_states)
    ax.set_xlim(1945, 1965)
    ax.set_ylim([64.8, 67.2])
    ax.set_xticks(np.arange(1945, 1966, 5))
    ax.set_yticks([65, 66, 67])
    ax.set_xlabel("Year of birth")
    ax.set_ylabel("SRA")
    fig.tight_layout()
    fig.savefig(path_dict["plots"] + "SRA_2007_reform.png", transparent=True, dpi=300)
