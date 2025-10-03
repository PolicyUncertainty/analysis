import matplotlib.pyplot as plt
import numpy as np


def one_std_cash(c, util_scale, mu, lam):
    minus_mu = 1 - mu
    return (c**minus_mu + (np.pi / np.sqrt(6)) * lam * minus_mu / util_scale) ** (
        1 / minus_mu
    ) - c


def plot_cons(mu, scale):
    cons = np.arange(0, 1, 0.01) / scale

    util = (cons ** (1 - mu) - 1) / (1 - mu)
    fig, ax = plt.subplots()
    ax.plot(cons, util)
    plt.show()
