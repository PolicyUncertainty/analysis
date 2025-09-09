import matplotlib.pyplot as plt


def set_plot_defaults():
    """wrapper to set standard matplotlib specifications for plots."""
    set_standard_matplotlib_specs()
    set_colors()
    return


def set_standard_matplotlib_specs(plot_type="paper"):
    """Set standard matplotlib specifications for plots.
    Parameters
    ----------
    plot_type : str
        The type of plot to create (can be "paper", "presentation").
    """
    # Set matplotlib fontsizes
    plt.rcParams.update(
        {
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        }
    )
    # Make lines of plots thicker
    plt.rcParams["lines.linewidth"] = 3

    # Set figure size
    if plot_type == "paper":
        plt.rcParams["figure.figsize"] = (12, 8)
    elif plot_type == "presentation":
        plt.rcParams["figure.figsize"] = (16, 9)

    # set resolution
    plt.rcParams["figure.dpi"] = 300

    # set transparent background
    plt.rcParams["figure.facecolor"] = "none"
    plt.rcParams["axes.facecolor"] = "none"

    # set common alpha values
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["legend.framealpha"] = 1

    # set grid and legend defaults
    plt.rcParams["legend.fancybox"] = True

    # set common line width for specific elements (overrides global when needed)
    plt.rcParams["grid.linewidth"] = 2
    return


def set_colors():
    JET_COLOR_MAP = [
        "#1f77b4", #0
        "#ff7f0e", #1
        "#2ca02c", #2
        "#d62728", #3
        "#9467bd", #4
        "#8c564b", #5
        "#e377c2", #6
        "#7f7f7f", #7
        "#bcbd22", #8
        "#17becf", #9
    ]
    LINE_STYLES = [
        "-",
        "--",
        "-.",
        ":",
    ]
    return JET_COLOR_MAP, LINE_STYLES
