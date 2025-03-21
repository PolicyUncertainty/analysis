import pandas as pd
import numpy as np
import statsmodels.api as sm
from process_data.first_step_sample_scripts.create_credited_periods_est_sample import create_credited_periods_est_sample

# delete later
from set_paths import create_path_dict
paths_dict = create_path_dict()


# estimate with ols: credited periods on experience, sex, education, has_partner

df = create_credited_periods_est_sample(paths_dict, load_data=True)

df["const"] = 1
edu_states = [0, 1]
sexes = [0, 1]
sub_group_names = ["sex", "education"]
multiindex = pd.MultiIndex.from_product(
    [sexes, edu_states],
    names=sub_group_names,
)
columns = [
    "const",
    "experience",
    "has_partner",
    "sex"
    ]
estimates = pd.DataFrame(index=multiindex, columns=columns)

#for sex in sexes:
#    for education in edu_states:
#        df_reduced = df[
#            (df["sex"] == sex)
#            & (df["education"] == education)
#        ]
#        X = df_reduced[columns]
#        Y = df_reduced["credited_periods"]
#        model = sm.OLS(Y, X).fit()
#        estimates.loc[(sex, education), columns] = model.params
#        print(f'sex: {sex} \n education: {education}')
#        print(model.summary())

X = df[columns]
Y = df["credited_periods"]
model = sm.OLS(Y, X).fit()
print(model.summary())


# predict credited periods for all observations
df["predicted_credited_periods"] = model.predict(df[columns])
# plot predicted vs actual credited periods by experience
import matplotlib.pyplot as plt
plt.scatter(df["experience"], df["credited_periods"], label="actual")
plt.scatter(df["experience"], df["predicted_credited_periods"], label="predicted")
plt.xlabel("experience")
plt.ylabel("credited periods")
plt.legend()
plt.show()