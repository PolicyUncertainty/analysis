import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))

from model_code.state_space import get_next_state, get_next_state_jax
