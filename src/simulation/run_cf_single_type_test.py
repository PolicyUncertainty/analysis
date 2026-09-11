# %%
"""Test: solve one edu x sex type and record its solve-time GPU/memory stats.

Purpose: verify that a single type-restricted solve fits inside one MIG 1g.23gb
slice, so the four edu x sex types can each run on their own slice in parallel.
sex and education are permanent (non-transitioning) types, so the pooled DP is
block-diagonal across them and a per-type solve is exact, not an approximation.

Only the solve is measured. Per-type simulation is out of scope here: the start-
state pipeline (load_scale_and_correct_data / generate_start_states_from_obs) does
not restrict observed agents to the model's sex/edu grids, so feeding them into a
type-restricted solution would be invalid without a separate filtering step.

Run one type per slice, selecting it via argv:
    CUDA_VISIBLE_DEVICES=MIG-<uuid> python -m simulation.run_cf_single_type_test women high

Saves one row to:
    <sim_results>/<model_name>/type_test/solve_stats_<sex>_<edu>.csv
"""
import pickle as pkl
import sys
import time
from pathlib import Path

import pandas as pd

from set_paths import create_path_dict
from specs.derive_specs import generate_derived_and_data_derived_specs

path_dict = create_path_dict()
specs = generate_derived_and_data_derived_specs(path_dict)

import jax

jax.config.update("jax_enable_x64", True)

from model_code.specify_model import specify_model

# %%
# Type selection. Defaults to one type; override per slice via argv.
sex_type = sys.argv[1] if len(sys.argv) > 1 else "women"
edu_type = sys.argv[2] if len(sys.argv) > 2 else "high"

# Budget of a single MIG 1g.23gb slice, for the fit verdict below.
MIG_SLICE_GB = 23.0

model_name = specs["model_name"]
util_type = specs["util_type"]

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

# %%
# Build the type-restricted model fresh (prints "sparsified from X to Y").
print(f"\n=== Building model for type: sex={sex_type}, edu={edu_type} ===", flush=True)
model = specify_model(
    path_dict=path_dict,
    specs=specs,
    subj_unc=True,
    custom_resolution_age=None,
    sim_specs={
        "announcement_age": None,
        "SRA_at_start": 67,
        "SRA_at_retirement": 67,
    },
    load_model=False,
    sex_type=sex_type,
    edu_type=edu_type,
    util_type=util_type,
)

# %%
# Warm-up solve (compiles), then a timed solve. peak_bytes_in_use is a process
# high-water mark, so it covers whichever of the two allocated the most.
print("=== warm-up solve (JIT compile) ===", flush=True)
warmup_solved = model.solve(params)
jax.block_until_ready((warmup_solved.value, warmup_solved.policy))
del warmup_solved

print("=== timed solve ===", flush=True)
start = time.perf_counter()
model_solved = model.solve(params)
jax.block_until_ready((model_solved.value, model_solved.policy))
solve_seconds = time.perf_counter() - start

# %%
# Collect stats.
device = jax.local_devices()[0]
mem_stats = device.memory_stats() or {}
peak_bytes = mem_stats.get("peak_bytes_in_use")

n_state_choices, n_cont_combos, n_total_wealth_grid = model_solved.value.shape

# Actual resident solution containers (endog_grid is None under druedahl_jorgensen).
containers = [model_solved.value, model_solved.policy]
if model_solved.endog_grid is not None:
    containers.append(model_solved.endog_grid)
container_gb = sum(c.size * c.dtype.itemsize for c in containers) / 1e9

# Stochastic fan-out and largest batch, read from batch_info when present.
stochastic_fanout = None
max_batch_size = None
try:
    batch_info = model.batch_info
    fanouts, batches = [], []
    for key, seg in batch_info.items():
        if not (
            isinstance(seg, dict) and "child_states_to_integrate_stochastic" in seg
        ):
            continue
        fanouts.append(int(seg["child_states_to_integrate_stochastic"].shape[-1]))
        batches.append(int(seg["batches_state_choice_idx"].shape[-1]))
    if fanouts:
        stochastic_fanout = max(fanouts)
        max_batch_size = max(batches)
except (AttributeError, KeyError, TypeError):
    pass

peak_gpu_gb = None if peak_bytes is None else peak_bytes / 1e9

row = {
    "sex_type": sex_type,
    "edu_type": edu_type,
    "n_state_choices": int(n_state_choices),
    "n_cont_combos": int(n_cont_combos),
    "n_total_wealth_grid": int(n_total_wealth_grid),
    "n_containers": len(containers),
    "stochastic_fanout": stochastic_fanout,
    "max_batch_size": max_batch_size,
    "container_gb": round(container_gb, 3),
    "peak_gpu_gb": None if peak_gpu_gb is None else round(peak_gpu_gb, 3),
    "fits_23gb_slice": (
        None if peak_gpu_gb is None else bool(peak_gpu_gb < MIG_SLICE_GB)
    ),
    "solve_seconds": round(solve_seconds, 2),
    "device": device.device_kind,
}

# %%
out_dir = Path(path_dict["sim_results"]) / model_name / "type_test"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"solve_stats_{sex_type}_{edu_type}.csv"
pd.DataFrame([row]).to_csv(out_path, index=False)

print("\n=== single-type solve stats ===")
print(pd.DataFrame([row]).to_string(index=False))
if peak_gpu_gb is not None:
    verdict = "FITS" if peak_gpu_gb < MIG_SLICE_GB else "DOES NOT FIT"
    print(f"\nPeak {peak_gpu_gb:.1f} GB vs {MIG_SLICE_GB:.0f} GB slice -> {verdict}")
else:
    print("\nNo GPU memory_stats available (not on a GPU device).")
print(f"Saved: {out_path}")
