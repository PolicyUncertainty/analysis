"""Inspect discontinuities in the experience dimension of the solved value function
and how they travel backwards in age.

Motivation
----------
The very-long-insured (VLI) pension gives a deduction-free early retirement to
workers who reach 45 credited years (a sex-specific experience threshold, ~42
model-years for men) and retire within 2 years of the SRA. That makes the
*retirement* payoff jump at the threshold. A still-working man some years before
the VLI window inherits that jump: by working full time he can still reach the
threshold, so the value of continuing to work jumps at the experience level from
which the threshold is exactly reachable by the window. Because men can only work
full time (the model already forbids part-time for men), the reachable increments
are whole years, so in real experience-years the jump should sit one year further
below the threshold for each year before the window, i.e. at

    experience-years  ==  threshold - (first_window_age - age)      (age <= window)

and travel down one year per year of age. Two things complicate this: the
experience state is stored *normalized* to [0, 1] by a period-dependent scale
(``max_exps_period_working[period]``), and that scale jumps from 48 to 59 exactly
at age 63 (``min_period_very_long_insured``). So a *fixed* normalized grid images
onto a period-dependent real-year range, and the clean travel is only exact in the
constant-scale region (age >= 63).

This script solves a low-education men's model at a *fixed, known SRA* (no
subjective uncertainty, so the jump is not blurred by belief-averaging), on a grid
dense at integer experience-years, reads the solved value along experience at each
age 50..67, and reports the largest experience jump within the dense band and how
it travels backwards.

Design decisions (made here, documented for the report)
-------------------------------------------------------
* Type: low-education men (``sex_type='men'``, ``edu_type='low'``).
* SRA fixed via ``subj_unc=False``; inspection at the policy state whose SRA is
  ``INSPECT_SRA``. No belief uncertainty -> sharpest possible jump.
* "From age 50 onwards": we solve the full backward induction (start_age
  unchanged, so all age-dependent economics stay correct) and only *inspect* ages
  >= 50. Backward induction makes the age-50+ solution independent of earlier ages,
  exactly as the task notes -- faithful, and no fragile period truncation.
* Experience grid: dense integer model-years ``DENSE_YEARS`` (25..45 -- the task's
  30..45 band, extended a little below so we can follow the jump as it travels down
  past 30 at younger ages), plus a few coarse anchors so the normalized grid still
  spans [0, 1].
* We read the *choice-specific* value of continuing full time (choice 3) -- the
  object that carries the travelling jump -- and, in the window, the value of
  retiring (choice 0). A true discontinuity shows as a large value change across one
  grid cell (linear interpolation cannot represent a jump inside a cell), so we look
  for the largest cell-to-cell change within the dense integer band.
* x-axis is real experience-years (``construct_experience_years`` = normalized x
  period scale), the frame in which the jump travels linearly.
"""

import os
import pickle as pkl
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from matplotlib import pyplot as plt

from model_code.specify_model import specify_model
from model_code.state_space.experience import construct_experience_years
from set_paths import create_path_dict
from set_styles import get_figsize, set_colors, set_plot_defaults
from specs.derive_specs import generate_derived_and_data_derived_specs

# ----------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------
SEX_TYPE, EDU_TYPE = "men", "low"
SEX, EDU = 0, 0
INSPECT_SRA = 67
INSPECT_AGES = list(range(50, 68))  # 50 .. 67
DENSE_YEARS = np.arange(25, 46)  # dense integer band (model-years)
COARSE_YEARS = np.array([0, 12, 50, 59])  # anchors so the normalized grid spans [0,1]
FIXED_STATE = dict(
    health=0,
    partner_state=1,
    job_offer=1,
    informed=1,
    alg_1_claim=0,
    lagged_choice=3,
)
ASSET_MAIN = 25.0

BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BENCHMARK_DIR / "output_experience_discontinuities"


def build_dense_experience_grid(scale):
    years = np.unique(np.concatenate([COARSE_YEARS, DENSE_YEARS])).astype(float)
    return years / scale


def solve_model(path_dict, specs, params):
    scale = float(specs["max_exps_period_working"][-1])
    grid = build_dense_experience_grid(scale)

    model = specify_model(
        path_dict=path_dict,
        specs=specs,
        subj_unc=False,
        custom_resolution_age=None,
        load_model=False,
        sex_type=SEX_TYPE,
        edu_type=EDU_TYPE,
        util_type=specs["util_type"],
        experience_grid=grid,
    )
    sol_path = str(OUTPUT_DIR / "sol_cache.pkl")
    if os.path.exists(sol_path):
        print("Loading cached solution ...", flush=True)
        model_solved = model.solve(params, load_sol_path=sol_path)
    else:
        print("Solving ...", flush=True)
        model_solved = model.solve(params, save_sol_path=sol_path)
    print("Solved.", flush=True)
    return model_solved, grid, scale


def policy_state_for_sra(specs, sra):
    return int(round((sra - specs["min_SRA"]) / specs["SRA_grid_size"]))


def value_over_experience(model_solved, grid, period, choice, asset, policy_state):
    n = len(grid)
    ones = np.ones(n, dtype=int)
    states = {
        "period": ones * period,
        "lagged_choice": ones * FIXED_STATE["lagged_choice"],
        "education": ones * EDU,
        "sex": ones * SEX,
        "informed": ones * FIXED_STATE["informed"],
        "policy_state": ones * policy_state,
        "job_offer": ones * FIXED_STATE["job_offer"],
        "partner_state": ones * FIXED_STATE["partner_state"],
        "health": ones * FIXED_STATE["health"],
        "alg_1_claim": ones * FIXED_STATE["alg_1_claim"],
        "assets_begin_of_period": np.ones(n) * asset,
        "experience": np.asarray(grid, dtype=float),
    }
    _, value = model_solved.policy_and_value_for_states_and_choices(
        states=states, choices=ones * choice
    )
    return np.asarray(value)


def real_years(grid, period, specs):
    return np.asarray(
        construct_experience_years(
            float_experience=grid,
            period=np.full(len(grid), period, dtype=int),
            is_retired=np.zeros(len(grid), dtype=bool),
            model_specs=specs,
        )
    )


def largest_jump_in_band(x, value, in_band):
    """Largest cell-to-cell value change among consecutive nodes that are both in
    the dense band -> (mid-x location, size, is_at_band_edge)."""
    idx = np.where(in_band & np.isfinite(value))[0]
    if len(idx) < 2:
        return np.nan, 0.0, True
    best = None
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            continue
        dv = value[b] - value[a]
        if best is None or abs(dv) > abs(best[1]):
            best = (0.5 * (x[a] + x[b]), dv, a)
    if best is None:
        return np.nan, 0.0, True
    loc, size, a = best
    at_edge = (a == idx[0]) or (a + 1 == idx[-1])
    return loc, size, at_edge


def run():
    set_plot_defaults()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path_dict = create_path_dict()
    specs = generate_derived_and_data_derived_specs(path_dict)
    model_name = specs["model_name"]
    params = pkl.load(
        open(path_dict["struct_results"] + f"est_params_{model_name}.pkl", "rb")
    )

    model_solved, grid, scale = solve_model(path_dict, specs, params)
    policy_state = policy_state_for_sra(specs, INSPECT_SRA)
    threshold = float(specs["experience_threshold_very_long_insured"][SEX])
    start_age = specs["start_age"]
    first_window_age = INSPECT_SRA - 2  # earliest deduction-free VLI age
    # Which grid nodes are in the dense integer band (age-independent property).
    in_band = np.isin(np.round(grid * scale).astype(int), DENSE_YEARS)

    # ---- Full-time (continuation) value along experience, per age ----
    records = []
    fig, ax = plt.subplots(figsize=get_figsize(1, 1))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(INSPECT_AGES)))
    for c, age in zip(cmap, INSPECT_AGES):
        period = age - start_age
        xy = real_years(grid, period, specs)
        val = value_over_experience(
            model_solved, grid, period, 3, ASSET_MAIN, policy_state
        )
        ax.plot(xy, val, color=c, label=f"age {age}" if age % 3 == 0 else None, lw=2)
        loc, size, edge = largest_jump_in_band(xy, val, in_band)
        records.append((age, loc, size, edge))

    ax.axvline(
        threshold, color="red", ls="--", lw=1.5, label=f"threshold ({threshold:.0f}y)"
    )
    ax.set_xlabel("Experience (real model-years)")
    ax.set_ylabel("Value of working full time")
    ax.set_title(f"Full-time value vs experience, men low edu, SRA {INSPECT_SRA}")
    ax.set_xlim(20, 48)
    ax.legend(ncol=2, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fulltime_value_by_age.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- First differences (the jump shows as a spike), window + nearby ages ----
    fig, ax = plt.subplots(figsize=get_figsize(1, 1))
    for c, age in zip(plt.cm.plasma(np.linspace(0, 1, 6)), [58, 60, 62, 64, 65, 66]):
        period = age - start_age
        xy = real_years(grid, period, specs)
        val = value_over_experience(
            model_solved, grid, period, 3, ASSET_MAIN, policy_state
        )
        m = in_band
        xs, vs = xy[m], val[m]
        mid = 0.5 * (xs[:-1] + xs[1:])
        dv = np.diff(vs) / np.diff(xs)
        ax.plot(mid, dv, color=c, marker="o", ms=4, label=f"age {age}")
    ax.axvline(threshold, color="red", ls="--", lw=1.5, label="threshold")
    ax.set_xlabel("Experience (real model-years)")
    ax.set_ylabel("dV / d(experience)  (full time)")
    ax.set_title("Slope of full-time value across the dense band")
    ax.set_xlim(20, 46)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fulltime_slope_by_age.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- Where the largest in-band jump sits, by age ----
    rec = np.array([(a, l, s) for a, l, s, e in records], dtype=float)
    edges = np.array([e for *_, e in records])
    fig, ax = plt.subplots(figsize=get_figsize(1, 1))
    good = ~edges
    ax.scatter(
        rec[good, 0],
        rec[good, 1],
        c=np.abs(rec[good, 2]),
        cmap="magma",
        s=90,
        label="jump inside band",
    )
    if edges.any():
        ax.scatter(
            rec[edges, 0],
            rec[edges, 1],
            facecolors="none",
            edgecolors="grey",
            s=90,
            label="jump at band edge (grid may miss it)",
        )
    ax.axhline(threshold, color="red", ls="--", label="VLI threshold")
    ref_ages = np.array(INSPECT_AGES, dtype=float)
    ax.plot(
        ref_ages,
        threshold - np.maximum(first_window_age - ref_ages, 0),
        color="grey",
        ls=":",
        lw=2,
        label="threshold - (65 - age)",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel("Experience-years of largest in-band jump")
    ax.set_title("How the experience jump travels backwards")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "jump_location_by_age.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # ---- Retirement value in the window (sanity: jump exactly at threshold) ----
    fig, ax = plt.subplots(figsize=get_figsize(1, 1))
    for c, age in zip(plt.cm.plasma(np.linspace(0, 1, 4)), [64, 65, 66, 67]):
        period = age - start_age
        xy = real_years(grid, period, specs)
        val = value_over_experience(
            model_solved, grid, period, 0, ASSET_MAIN, policy_state
        )
        ax.plot(xy, val, color=c, label=f"retire at {age}", lw=2)
    ax.axvline(threshold, color="red", ls="--", label="VLI threshold")
    ax.set_xlabel("Experience (real model-years)")
    ax.set_ylabel("Value of retiring")
    ax.set_title("Retirement value vs experience (window ages)")
    ax.set_xlim(30, 48)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "retirement_value_window.png", bbox_inches="tight", dpi=200
    )
    plt.close(fig)

    # ---- Text summary ----
    lines = [
        f"Type: men, low edu. SRA={INSPECT_SRA} (policy_state={policy_state}), "
        f"subj_unc=False. Threshold={threshold:.1f} model-yr. Norm scale (max)={scale:.0f}.",
        f"Dense band (model-years): {DENSE_YEARS.min()}..{DENSE_YEARS.max()}.",
        f"Grid (model-years): {np.round(grid * scale, 2).tolist()}",
        "",
        "Largest in-band experience jump in the FULL-TIME value, by age:",
        "  age | jump @ exp-years | jump size | at band edge?",
    ]
    for age, loc, size, edge in records:
        lines.append(
            f"  {int(age):3d} | {loc:14.2f} | {size:+9.4f} | {'YES' if edge else 'no'}"
        )
    summary = "\n".join(lines)
    print(summary, flush=True)
    (OUTPUT_DIR / "summary.txt").write_text(summary)
    print(f"\nSaved plots + summary to {OUTPUT_DIR}", flush=True)
    return records


if __name__ == "__main__":
    run()
