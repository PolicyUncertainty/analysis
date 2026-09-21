# %%
"""What the model costs to build, solve, evaluate and simulate on this machine.

Times the three things the project actually waits on -- the solve, the likelihood,
and solve-and-simulate -- each of them jit-compiled, and records the GPU memory they
take. Every stage is called ``N_CALLS`` times: the first call pays for tracing and
XLA compilation and is reported on its own, the rest are the steady-state cost that
an estimation loop or a counterfactual sweep really pays.

Set ``RUN_NAME`` to whatever identifies the machine to you -- it is written into
every row and into the output filename, so results from different devices can be
concatenated and compared without further bookkeeping.

``SOLVE_MODE`` and ``SMALL_RAM`` are the two knobs under test:

``SOLVE_MODE``
    ``"pooled"`` solves the whole state space in one go. ``"chunked_seq"`` and
    ``"chunked_parallel"`` assemble the same solution from the four sex x education
    sub-model solves (``dcegm.get_solve_from_small_models``), one at a time or
    dispatched across ``jax.devices()``. Chunking is what makes the solve fit where
    the pooled one does not; the parallel variant only helps with several devices.
``SMALL_RAM``
    Whether the solve interpolates the income-shock draws in blocks of one instead
    of all at once (``income_shock_batch_size``). Cuts the peak of the interpolation
    step at the cost of parallelism; the solution is unchanged.

``SOLVE_MODE`` reaches the solve only. The likelihood and solve-and-simulate are
built from a single model object -- ``get_solve_from_small_models`` hands back a
solved model, not a solve function to compile into them -- so they always run the
pooled solve, and their rows say so in ``solve_mode_applies``. ``SMALL_RAM`` reaches
all three.

Results: src/benchmarks/output/benchmark_<RUN_NAME>_<SOLVE_MODE>_<ram>.csv
"""
import os
import pickle as pkl
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import pandas as pd

from set_paths import create_path_dict

# =====================================================================================
# Set before every run
# =====================================================================================

# Manual identifier for the machine this is running on. Free-form, but keep it stable
# per device so rows from repeated runs line up (e.g. "hu_h100nvl", "hu_a100_80gb",
# "hu_a100_mig_20gb", "laptop_cpu").
RUN_NAME = "b200_split"

# "pooled" | "chunked_seq" | "chunked_parallel"
SOLVE_MODE = "chunked_parallel"

# True -> income_shock_batch_size = 1 (block the income-shock draws)
SMALL_RAM = True

# First call is compilation, the remaining ones are the steady-state cost.
N_CALLS = 5

# Simulation seed; fixed so solve-and-simulate is comparable across runs.
SEED = 123

SOLVE_MODES = ("pooled", "chunked_seq", "chunked_parallel")
if SOLVE_MODE not in SOLVE_MODES:
    raise ValueError(f"SOLVE_MODE must be one of {SOLVE_MODES}, got {SOLVE_MODE!r}.")

INCOME_SHOCK_BATCH_SIZE = 1 if SMALL_RAM else None

BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BENCHMARK_DIR / "output"
MODEL_CACHE_DIR = BENCHMARK_DIR / "model_cache"

# %%
# =====================================================================================
# Timing and memory
# =====================================================================================


def peak_device_bytes():
    """High-water mark of device memory, or None off-GPU.

    JAX never resets ``peak_bytes_in_use`` within a process, so this is the ceiling
    reached by everything up to this point in the run, not an isolate for the stage
    just finished. Read the first stage's number as that stage's peak and the later
    ones as the running ceiling; to isolate a stage, run it in its own process.
    """
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    return None if stats is None else stats.get("peak_bytes_in_use")


def current_device_bytes():
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    return None if stats is None else stats.get("bytes_in_use")


def peak_host_bytes():
    """Peak resident set size of this process. ru_maxrss is KiB on Linux."""
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss * 1024 if platform.system() == "Linux" else max_rss


ROWS = []


def record(stage, call_index, seconds, solve_mode_applies=True, note=""):
    phase = "compile" if call_index == 1 else "steady"
    ROWS.append(
        {
            "run_name": RUN_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "solve_mode": SOLVE_MODE,
            "solve_mode_applies": solve_mode_applies,
            "small_ram": SMALL_RAM,
            "income_shock_batch_size": INCOME_SHOCK_BATCH_SIZE,
            "n_devices": len(jax.devices()),
            "device_kind": jax.devices()[0].device_kind,
            "stage": stage,
            "call_index": call_index,
            "phase": phase,
            "seconds": seconds,
            "peak_device_bytes": peak_device_bytes(),
            "device_bytes_in_use": current_device_bytes(),
            "peak_host_bytes": peak_host_bytes(),
            "note": note,
        }
    )
    print(f"  {stage} call {call_index} ({phase}): {seconds:8.2f}s", flush=True)
    save_rows()


def save_rows():
    """Written after every call, so a run that dies late still leaves its timings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(output_csv_path, index=False)


def time_repeatedly(stage, func, block, solve_mode_applies=True, note=""):
    """Call ``func`` N_CALLS times, timing each; ``block`` waits for the result."""
    print(f"\n=== {stage} ===", flush=True)
    for call_index in range(1, N_CALLS + 1):
        start = time.perf_counter()
        result = func()
        block(result)
        record(
            stage=stage,
            call_index=call_index,
            seconds=time.perf_counter() - start,
            solve_mode_applies=solve_mode_applies,
            note=note,
        )
        del result
    return None


def block_on_solution(model_solved):
    jax.block_until_ready((model_solved.value, model_solved.policy))


def block_on_array(array):
    jax.block_until_ready(array)


def block_on_frame(df):
    # simulate() already pulls the panel to host, so there is nothing left to wait on.
    assert isinstance(df, pd.DataFrame)


# %%
# =====================================================================================
# Setup
# =====================================================================================

ram_tag = "smallram" if SMALL_RAM else "bigram"
output_csv_path = OUTPUT_DIR / f"benchmark_{RUN_NAME}_{SOLVE_MODE}_{ram_tag}.csv"

path_dict = create_path_dict()

from specs.derive_specs import generate_derived_and_data_derived_specs  # noqa: E402

specs = generate_derived_and_data_derived_specs(path_dict)
model_name = specs["model_name"]

# Build models into a benchmark-local cache so a benchmark run never overwrites the
# production model pickles.
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
bench_path_dict = dict(path_dict)
bench_path_dict["intermediate_data"] = str(MODEL_CACHE_DIR) + "/"

params = pkl.load(
    open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
)

print(
    f"Benchmark run '{RUN_NAME}': solve_mode={SOLVE_MODE}, small_ram={SMALL_RAM} "
    f"(income_shock_batch_size={INCOME_SHOCK_BATCH_SIZE}), "
    f"{len(jax.devices())}x {jax.devices()[0].device_kind}",
    flush=True,
)

from model_code.specify_model import specify_model  # noqa: E402


def build_model(sex_type="all", edu_type="all"):
    return specify_model(
        path_dict=bench_path_dict,
        specs=specs,
        subj_unc=True,
        custom_resolution_age=None,
        load_model=False,
        sim_specs=None,
        sex_type=sex_type,
        edu_type=edu_type,
        util_type=specs["util_type"],
        income_shock_batch_size=INCOME_SHOCK_BATCH_SIZE,
    )


# %%
# =====================================================================================
# 1) Building the model
# =====================================================================================

print("\n=== build model ===", flush=True)
start = time.perf_counter()
model = build_model()
record(stage="build_model", call_index=1, seconds=time.perf_counter() - start)

if SOLVE_MODE == "pooled":
    solve_func = model.get_solve_func()
else:
    import dcegm

    print("\n=== build sub-models ===", flush=True)
    start = time.perf_counter()
    sub_models = [
        build_model(sex_type=sub_sex_type, edu_type=sub_edu_type)
        for sub_sex_type in ("men", "women")
        for sub_edu_type in ("low", "high")
    ]
    record(
        stage="build_sub_models",
        call_index=1,
        seconds=time.perf_counter() - start,
        note="four sex x education sub-models for the chunked solve",
    )
    solve_func = dcegm.get_solve_from_small_models(
        small_models=sub_models,
        parallel=SOLVE_MODE == "chunked_parallel",
        big_model=model,
    )

# %%
# =====================================================================================
# 2) Solve
# =====================================================================================

time_repeatedly(
    stage="solve",
    func=lambda: solve_func(params),
    block=block_on_solution,
)

# %%
# =====================================================================================
# 3) Likelihood
# =====================================================================================

from model_code.transform_data_from_model import (  # noqa: E402
    create_states_dict,
    load_scale_and_correct_data,
)
from model_code.unobserved_state_weighting import (  # noqa: E402
    create_unobserved_state_specs,
)

print("\n=== build likelihood function ===", flush=True)
data_decision = load_scale_and_correct_data(path_dict=path_dict, model_class=model)
# Already retired individuals hold no identification; the estimation drops them too.
data_decision = data_decision[data_decision["lagged_choice"] != 0]
states_dict = create_states_dict(data_decision, model_class=model)

start = time.perf_counter()
ll_func = model.create_experimental_ll_func(
    observed_states=states_dict,
    observed_choices=data_decision["choice"].values,
    unobserved_state_specs=create_unobserved_state_specs(data_decision=data_decision),
    params_all=params,
    return_model_solution=False,
    slow_version=False,
)
record(
    stage="build_likelihood_func",
    call_index=1,
    seconds=time.perf_counter() - start,
    solve_mode_applies=False,
    note=f"{len(data_decision)} observations",
)

time_repeatedly(
    stage="likelihood",
    func=lambda: ll_func(params),
    block=block_on_array,
    solve_mode_applies=False,
    note="always the pooled solve; get_solve_from_small_models is solve-only",
)

# %%
# =====================================================================================
# 4) Solve and simulate
# =====================================================================================

from simulation.sim_tools.start_obs_for_sim import (  # noqa: E402
    generate_start_states_from_obs,
)

print("\n=== build solve-and-simulate function ===", flush=True)
initial_states = generate_start_states_from_obs(
    path_dict=path_dict,
    params=params,
    model_class=model,
    inital_SRA=specs["min_SRA"],
    seed=SEED,
)
start = time.perf_counter()
solve_and_simulate_func = model.get_solve_and_simulate_func(
    states_initial=initial_states,
    seed=SEED,
    slow_version=False,
)
record(
    stage="build_solve_simulate_func",
    call_index=1,
    seconds=time.perf_counter() - start,
    solve_mode_applies=False,
    note=f"{len(initial_states['period'])} simulated agents",
)

time_repeatedly(
    stage="solve_simulate",
    func=lambda: solve_and_simulate_func(params),
    block=block_on_frame,
    solve_mode_applies=False,
    note="always the pooled solve; get_solve_from_small_models is solve-only",
)

# %%
# =====================================================================================
# Summary
# =====================================================================================

results = pd.DataFrame(ROWS)
save_rows()

summary = (
    results[results["stage"].isin(["solve", "likelihood", "solve_simulate"])]
    .groupby(["stage", "phase"])["seconds"]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)
print("\n" + "=" * 70)
print(f"Run '{RUN_NAME}' -- solve_mode={SOLVE_MODE}, small_ram={SMALL_RAM}")
print("=" * 70)
print(summary.to_string())

peak = results["peak_device_bytes"].dropna()
if len(peak):
    print(f"\nPeak device memory over the run: {peak.max() / 2**30:.2f} GiB")
print(
    f"Peak host memory over the run:   {results['peak_host_bytes'].max() / 2**30:.2f} GiB"
)
print(f"\nWritten to {output_csv_path}")
