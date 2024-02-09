#%%
import numpy as np
import pandas as pd
from scipy.stats import norm

step_size = 0.25
min_ret_age = 63
max_ret_age = 72
alpha_hat = 0.03
sigma_sq_hat = 0.06

# create matrix of zeros and row/column labels
n_policy_states = int((max_ret_age - min_ret_age) / step_size + 1)
labels = np.arange(min_ret_age, max_ret_age + step_size, step_size)

ret_age_exp_transition_matrix = pd.DataFrame(
    np.zeros((n_policy_states, n_policy_states)),
    index=labels,
    columns=labels,
)
#%%

# fill in the matrix with the transition probabilities from the normal CDF
# if the column is min ret age, p = CDF(celta - step_size/2)
# if the column is max ret age, p = 1 - CDF(celta + step_size/2)
# otherwise, p = CDF(celta + step_size/2) - CDF(celta - step_size/2)
for i in range(n_policy_states):
    for j in range(n_policy_states):
        if j == 0:
            ret_age_exp_transition_matrix.iloc[i, j] = norm.cdf(
                labels[j] - alpha_hat * labels[i] + step_size / 2
            )
        elif j == n_policy_states - 1:
            ret_age_exp_transition_matrix.iloc[i, j] = 1 - norm.cdf(
                labels[j] - alpha_hat * labels[i] - step_size / 2
            )
        else:
            ret_age_exp_transition_matrix.iloc[i, j] = (
                norm.cdf(labels[j] - alpha_hat * labels[i] + step_size / 2)
                - norm.cdf(labels[j] - alpha_hat * labels[i] - step_size / 2)
            )