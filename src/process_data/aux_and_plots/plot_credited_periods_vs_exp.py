import matplotlib.pyplot as plt


def plot_credited_periods_vs_exp(df):
    """ Plot credited periods vs experience """
    men_mask = df["sex"]==0
    plt.scatter(df[men_mask]["working_years"], df[men_mask]["credited_periods"], label="men: experience vs credited periods")
    plt.scatter(df[~men_mask]["working_years"], df[~men_mask]["credited_periods"], label="women: experience vs credited periods")
    plt.plot([0, 45], [0, 45], label="y=x", color="red")
    plt.xlabel("ft_exp_plus_pt_exp")
    plt.ylabel("credited periods")
    plt.legend()
    plt.show()
    