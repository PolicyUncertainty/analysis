import numpy as np


def create_savings_grid():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=20, step=2)  # 10
    section_2 = np.arange(start=20, stop=50, step=5)  # 6
    section_3 = np.arange(start=50, stop=100, step=20)  # 3
    section_4 = np.arange(start=100, stop=500, step=100)  # 4
    section_5 = np.arange(start=500, stop=1000, step=250)  # 2
    savings_grid = np.concatenate(
        [section_1, section_2, section_3, section_4, section_5]
    )
    return savings_grid
