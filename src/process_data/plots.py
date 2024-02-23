fig, ax = plt.subplots()
ax.scatter(df[x_var], df[y_var], s=(df[weights] / df[weights].sum()) * 5000)
ax.plot(
    df[x_var],
    alpha_hat[0] * df[x_var],
    "r",
)
ax.set_xlabel("Time to retirement")
ax.set_ylabel(
    "Expected retirement age",
)
plt.savefig(paths["project_path"] + "output/exp_val_plot.png")

fig, ax = plt.subplots()
ax.scatter(df[x_var], df[y_var], s=(df[weights] / df[weights].sum()) * 5000)
ax.plot(
    df[x_var],
    sigma_sq_hat[0] * df[x_var],
    "r",
)
ax.set_xlabel("Time to retirement")
ax.set_ylabel(
    "Expected retirement age",
)
plt.savefig(paths["project_path"] + "output/var_plot.png")
# plt.show()
