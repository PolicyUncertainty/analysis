import numpy as np


def create_end_of_period_assets():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=10, step=1)  # 10
    section_2 = np.arange(start=10, stop=20, step=2)  # 5
    section_3 = np.arange(start=20, stop=50, step=5)  # 6
    section_4 = np.arange(start=50, stop=100, step=10)  # 3
    section_5 = np.arange(start=100, stop=500, step=100)  # 4
    section_6 = np.arange(start=500, stop=1000, step=200)  # 3
    # Add 5, 10 and a 100 million to the grid
    section_7 = np.array([5_000, 10_000, 100_000])
    savings_grid = np.concatenate(
        [section_1, section_2, section_3, section_4, section_5, section_6, section_7]
    ).astype(float)
    # The steps above are made in thousands, so we multiply by 1000
    savings_grid = savings_grid * 1000
    return savings_grid
