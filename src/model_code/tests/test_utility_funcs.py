import sys
from pathlib import Path

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))

from model_code.utility_functions import utility_func, inverse_marginal, marg_utility
import numpy as np


def test_utility_func():
    dis_util_work = 0.3
    dis_util_unemployed = 0.8
    mu = 0.5

    params = {
        "mu": mu,
        "dis_util_work": dis_util_work,
        "dis_util_unemployed": dis_util_unemployed,
    }
    cons = 2
    cons_utility = cons ** (1 - mu) / mu

    np.testing.assert_almost_equal(
        utility_func(cons, 1, params), cons_utility - dis_util_work
    )
    np.testing.assert_almost_equal(
        utility_func(cons, 0, params), cons_utility - dis_util_unemployed
    )
