import matplotlib.pyplot as plt
import numpy as np


def plot_percentage_change(df):
    """Plot the change in baseline outcomes as a percentage of the baseline outcome."""

    for var in [
        "below_sixty_savings",
        "ret_age",
        "sra_at_ret",
        "working_hours",
    ]:
        fig, ax = plt.subplots(figsize=(12, 8))

        change = df[var] / df["base_" + var] - 1
        ax.plot(df["alpha"], change, label=var)

        # ax.plot(df["alpha"], df["cv"], label="Compensated Variation")
        ax.set_xlabel("Simulated yearly SRA increase")
        ax.set_ylabel("Percentage change")
        ax.legend()
        plt.show()
