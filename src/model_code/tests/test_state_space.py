# import sys
# from itertools import product
# from pathlib import Path
#
# import numpy as np
# import pytest
#
# src_folder = Path(__file__).resolve().parents[2]
# sys.path.append(str(src_folder))


import sys
from pathlib import Path

src_folder = Path(__file__).resolve().parents[2]
sys.path.append(str(src_folder))
import pytest
import numpy as np
from itertools import product
import jax

from model_code.state_space import sparsity_condition, update_state_space, create_state_space_functions