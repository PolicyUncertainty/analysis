# %% Set paths of project
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict(define_user=True)

specs = generate_derived_and_data_derived_specs(path_dict)

soep_is = path_dict["soep_is"]
relevant_cols = [
    "belief_pens_deduct",
    "age",
    "fweights",
    "education",
]
df = pd.read_stata(soep_is)[relevant_cols].astype(float)

# recode education
df["education"] = df["education"].replace({1: 0, 2: 0, 3: 1})

# Age as int
df["age"] = df["age"].astype(int)
# Restrict dataset to relevant age range and filter invalid beliefs
df = df[df["belief_pens_deduct"] >= 0]
df = df[df["age"] <= specs["max_ret_age"]]

predicted_shares = pd.read_csv(
    path_dict["est_results"] + "predicted_shares.csv", index_col=0
)
# Classify informed individuals
df["informed"] = df["belief_pens_deduct"] <= specs["informed_threshhold"]
informed_by_age = df.groupby(["age", "education"])["informed"].mean().rolling(3).mean()

# Create rolling average
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for edu_var, edu_label in enumerate(specs["education_labels"]):
    ax = axs[edu_var]
    ax.plot(informed_by_age.loc[(slice(None), edu_var)], label="Observed")
    ax.plot(predicted_shares[edu_label], label="Predicted")
    ax.set_title(edu_label)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Age")
    ax.set_ylabel("Share of informed")
    ax.legend()

fig.savefig(path_dict["plots"] + "informed_shares.png")
