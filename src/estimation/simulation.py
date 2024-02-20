# Set paths of project
import sys
from pathlib import Path

analysis_path = str(Path(__file__).resolve().parents[2]) + "/"
file_dir_path = str(Path(__file__).resolve().parents[0]) + "/"

sys.path.insert(0, analysis_path + "submodules/dcegm/src/")
sys.path.insert(0, analysis_path + "src/")
# Import jax and set jax to work with 64bit
import jax

jax.config.update("jax_enable_x64", True)

import pickle
