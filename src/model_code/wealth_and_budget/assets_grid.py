import numpy as np


def create_end_of_period_assets():
    """Create a saving grid with sections."""
    section_1 = np.arange(start=0, stop=10, step=0.5)  # 10
    section_2 = np.arange(start=10, stop=20, step=2.5)  # 4
    section_3 = np.arange(start=20, stop=50, step=15)  # 2
    section_4 = np.arange(start=50, stop=100, step=20)  # 2
    section_5 = [100, 500, 10_000]  # 1
    savings_grid = np.concatenate(
        [
            section_1,
            section_2,
            section_3,
            section_4,
            section_5,
        ]
    ).astype(float)
    # The steps above are made in thousands, so we multiply by 1000
    return savings_grid * 1000
