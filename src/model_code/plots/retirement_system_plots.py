"""Illustrative plots of the retirement-system features that shape the experience
state and its grid.

Every figure is a 2x2 grid of panels over (sex x education) -- the four solved
types. The panel selects `sex` and `education`; the
text below lists what else each sweep holds fixed and what each line varies. The
sweeps call the same pension functions the solver uses (`experience_stock.py`), so
the plots show the actual model, not a stylized redraw.

Policy/state values held fixed across every panel of plots 1-3 (the ones that
call the pension functions), unless a line says otherwise:
  - SRA = 67 (the 1964+ cohort), i.e. `policy_state = 8` under min_SRA=65,
    SRA_grid_size=0.25. The SRA sets where deductions/bonus and the VLI window sit.
  - informed = 1 (informed): early-retirement deductions use the statutory ERP,
    not the estimated uninformed penalty. Plot 4 is the exception -- it varies
    exactly this.
  - partner_state = 1 (working partner): fixes the mothers'-pension top-up for
    women (added inside the retirement pension function; zero for men).
  - `experience` is measured in years; the VLI threshold is sex-specific
    (`experience_threshold_very_long_insured[sex]`, ~42 for men, ~31.5 for women).
  - `period` enters only through the retirement age: a fresh retirement at age A
    corresponds to `period = A - start_age + 1` (actual_retirement_age =
    start_age + period - 1).

--------------------------------------------------------------------------------
Plot 1 -- `plot_pension_points_from_experience`
  Raw pension points (Entgeltpunkte) accrued from experience, BEFORE any
  retirement adjustment (no deductions/bonus/VLI, and no mothers' pension).
  x-axis: experience years, 0 .. max (=59). y-axis: pension points.
  Held fixed: everything except experience -- this is a pure function of
  (sex, education, experience).
  Lines:
    - solid: pension points vs experience (`calc_pension_points_form_experience`).
      Convex because wages rise with experience; the small slope changes are the
      piecewise-linear interpolation between integer-year nodes.
    - scatter: the sex's production experience-grid nodes (real years, at the last
      working period), showing where the grid samples this curve.

Plot 2 -- `plot_pension_vs_retirement_age`
  Pension points a person WOULD receive if they retired fresh at each age,
  holding their accumulated experience fixed. Isolates the retirement-age
  schedule (deductions below the SRA, deduction-free VLI window, late bonus) and
  the disability fill-up.
  x-axis: retirement age, from min_long_insured_age (63) to max_ret_age (72).
  y-axis: pension points (includes retirement adjustment + mothers' pension).
  Held fixed per line: SRA=67, informed=1, partner=working; experience and health
  are what differ between the three lines:
    - "Just below VLI threshold": experience = threshold[sex] - 0.5 yr, good
      health. Never VLI-eligible, so it carries early-retirement deductions all
      the way up to the SRA, then the late bonus above it.
    - "At/above VLI threshold": experience = threshold[sex] + 0.5 yr, good health.
      Same accrued level (half a year apart), but VLI-eligible: deduction-free in
      the [SRA-2, SRA] window (the flat segment). The vertical gap to the line
      above, inside the shaded window, IS the VLI benefit.
    - "Disability": experience = threshold[sex] + 0.5 yr, disabled health. Points
      are filled up as if working to a 47-year span, with the capped disability
      penalty -- elevated at young retirement ages, converging as age rises.
  Shading: the [SRA-2, SRA] VLI window; the vertical line marks the SRA.

Plot 3 -- `plot_vli_discontinuity`
  The VLI jump as a function of experience, at the binding decision age.
  x-axis: experience years, 0 .. 59. y-axis: pension points at retirement.
  Held fixed: retirement age = SRA-2 = 65 (`period = 65 - start_age + 1`), SRA=67,
  informed=1, partner=working, good health. ONLY experience varies along x.
  Lines / marks:
    - solid: retirement pension points vs experience. It jumps up at
      `threshold[sex]`: below it the person retires with the SRA-2 deduction,
      at/above it they qualify for the deduction-free VLI pension.
    - vertical line: the sex-specific VLI threshold.
    - shaded band [threshold-2, threshold]: the "one more year to qualify" region
      -- experience levels from which the threshold is still reachable within the
      2-year VLI window.
    - dotted verticals: the type's experience-grid nodes, to show whether the grid
      brackets the jump and resolves the band.

Plot 4 -- `plot_informed_vs_uninformed_deductions`
  The early-retirement pension FACTOR (the multiplier on total pension points on
  the deductible long-insured path), varying the information state. This is the
  one plot that does NOT hold `informed` fixed, and it does not call the pension
  function -- it evaluates the factor `clip(1 - penalty * years_early, 0, 1)`
  directly.
  x-axis: retirement age, from 4 years before the SRA up to the SRA. y-axis:
  pension factor (1.0 = no deduction).
  Held fixed: SRA=67; the factor is experience- and sex-independent (it depends on
  the penalty rate only), so the two sex rows coincide by construction.
  Lines:
    - "Informed": penalty = statutory ERP (0.036/yr).
    - "Uninformed": penalty = estimated `uninformed_ERP[education]` -- steeper, and
      education-specific, which is why the two education columns differ.

Plot 5 -- `plot_pension_by_retirement_timing`
  Pension points vs current experience at a FIXED current age (default 60), with
  one line per retirement-timing choice. Shows the timing trade-off directly:
  working longer raises experience AND changes the retirement age (smaller
  deduction, and the VLI jump once the retirement age is within 2 years of the
  SRA).
  x-axis: experience the person has TODAY (at `age_now`), 0 .. the feasible max at
  that age. y-axis: pension points at the eventual retirement.
  Held fixed: current age = 60, SRA=67, informed=1, partner=working, good health,
  and full-time work in every interim year (so N interim years add N years of
  experience). Each line varies N = years until retirement:
    - one line per N in `years_until_retirement` (default 3..7): the person works
      full time for N years then retires at age 60+N with experience (today + N).
      Higher N sits higher (more experience, smaller deduction).
    - dotted vertical (colored to match a line): for early-retirement timings that
      fall inside the VLI window (retirement age < SRA and within 2 years of it --
      from age 60 that means retiring at 65 or 66, N=5 or 6), the current
      experience `threshold - N` from which N more full-time years exactly reach
      the VLI threshold. The line jumps up there.
  With age_now=60 and SRA=67, the "retire at 65" and "retire at 66" lines show the
  VLI jump; retiring at 67 is the SRA (full pension, no jump); retiring at 63/64 is
  outside the 2-year window and always deducted.
"""

import os

import numpy as np
from matplotlib import pyplot as plt

from model_code.pension_system.experience_stock import (
    calc_pension_points_for_experience,
    calc_pension_points_form_experience,
)
from set_styles import get_figsize, set_colors

SEX_LABELS = ["Men", "Women"]
EDU_LABELS = ["Low Education", "High Education"]
# Fixed illustrative policy environment: SRA = 67 (the 1964+ cohort).
_ILLUSTRATIVE_SRA = 66
_PARTNER_STATE = np.array(1)  # working partner, as in the other retirement plots


def _sra_policy_state(specs):
    """Policy-state index whose SRA equals the illustrative SRA."""
    return int(round((_ILLUSTRATIVE_SRA - specs["min_SRA"]) / specs["SRA_grid_size"]))


def _axis_max(specs):
    """Max experience (years) for the plot x-axis range."""
    return float(specs["max_exps_period_working"][-1])


def _grid_nodes(specs, sex):
    """Production experience-grid nodes (real years) for this sex at the last working
    period -- the widest grid, spanning the full attainable range with the VLI
    bracket. The grid is per (sex, period) and does not depend on education."""
    period = specs["max_ret_age"] - specs["start_age"]
    return np.asarray(specs["experience_grid_working_by_sex_period"])[sex, period]


def _save(fig, path_dict, filename_base):
    out_dir = path_dict["misc_plots"]
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_dir + f"{filename_base}.pdf", bbox_inches="tight")
    fig.savefig(out_dir + f"{filename_base}.png", bbox_inches="tight", dpi=300)


def _finish(fig, path_dict, filename_base, show, save):
    fig.tight_layout()
    if save:
        _save(fig, path_dict, filename_base)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _retirement_pension_points(specs, experience_years, period, sex, education, health):
    """Fresh-retirement pension points at the illustrative SRA for one scalar state.

    The model's pension functions assume fully-scalar (vmapped) inputs, so callers
    vectorize over the swept dimension with ``_sweep`` below rather than passing an
    array for a single argument.
    """
    return float(
        calc_pension_points_for_experience(
            period=np.asarray(period),
            sex=sex,
            experience_years=np.asarray(experience_years, dtype=float),
            education=education,
            partner_state=_PARTNER_STATE,
            policy_state=_sra_policy_state(specs),
            informed=1,
            health=health,
            model_specs=specs,
        )
    )


def _sweep(specs, experience_values, period_values, sex, education, health):
    """Map ``_retirement_pension_points`` over paired experience/period arrays."""
    experience_values, period_values = np.broadcast_arrays(
        np.asarray(experience_values, dtype=float), np.asarray(period_values)
    )
    return np.array(
        [
            _retirement_pension_points(specs, e, p, sex, education, health)
            for e, p in zip(np.ravel(experience_values), np.ravel(period_values))
        ]
    )


# =====================================================================================
# 1. Pension points accrued from experience
# =====================================================================================
def plot_pension_points_from_experience(path_dict, specs, show=False, save=False):
    colors, _ = set_colors()
    scale = _axis_max(specs)
    exp_years = np.linspace(0.0, scale, 400)

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    fig.suptitle("Pension points accrued from experience", fontweight="bold")

    for sex in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            ax = axes[sex, edu]
            pp = np.asarray(
                calc_pension_points_form_experience(
                    education=edu,
                    sex=sex,
                    experience_years=exp_years,
                    model_specs=specs,
                )
            )
            ax.plot(exp_years, pp, color=colors[0])

            # Production experience-grid nodes (real years) as ticks along the curve.
            nodes = _grid_nodes(specs, sex)
            node_pp = np.interp(nodes, exp_years, pp)
            ax.scatter(
                nodes, node_pp, color=colors[3], zorder=3, s=40, label="Grid nodes"
            )

            ax.set_title(f"{SEX_LABELS[sex]}, {EDU_LABELS[edu]}")
            ax.set_xlabel("Experience (years)")
            ax.set_ylabel("Pension points")
            ax.grid(True, alpha=0.3)
            if sex == 0 and edu == 0:
                ax.legend()

    _finish(fig, path_dict, "retirement_pp_from_experience", show, save)


# =====================================================================================
# 2. Pension at retirement vs retirement age
# =====================================================================================
def plot_pension_vs_retirement_age(path_dict, specs, show=False, save=False):
    colors, _ = set_colors()
    start_age = specs["start_age"]
    # Retirement ages from the earliest long-insured age to the max, and the period
    # index each maps to (actual_retirement_age = start_age + period - 1).
    ret_ages = np.arange(specs["min_long_insured_age"], specs["max_ret_age"] + 1)
    periods = ret_ages - start_age + 1

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    fig.suptitle(
        f"Pension points at retirement vs retirement age (SRA = {_ILLUSTRATIVE_SRA})",
        fontweight="bold",
    )

    for sex in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            ax = axes[sex, edu]
            threshold = float(specs["experience_threshold_very_long_insured"][sex])

            for exp_years, label, color in [
                (threshold - 0.5, "Just below VLI threshold", colors[1]),
                (threshold + 0.5, "At/above VLI threshold", colors[0]),
            ]:
                pp = _sweep(specs, exp_years, periods, sex, edu, health=0)
                ax.plot(ret_ages, pp, color=color, label=label)

            # Disability at the same (above-threshold) experience, for comparison.
            pp_dis = _sweep(
                specs,
                threshold + 0.5,
                periods,
                sex,
                edu,
                health=specs["disabled_health_var"],
            )
            ax.plot(
                ret_ages, pp_dis, color=colors[2], linestyle="--", label="Disability"
            )

            ax.axvline(_ILLUSTRATIVE_SRA, color="grey", linewidth=1.5, alpha=0.6)
            ax.axvspan(
                _ILLUSTRATIVE_SRA - 2, _ILLUSTRATIVE_SRA, color="grey", alpha=0.12
            )
            ax.set_title(f"{SEX_LABELS[sex]}, {EDU_LABELS[edu]}")
            ax.set_xlabel("Retirement age")
            ax.set_ylabel("Pension points")
            ax.grid(True, alpha=0.3)
            if sex == 0 and edu == 0:
                ax.legend()

    _finish(fig, path_dict, "retirement_pp_vs_age", show, save)


# =====================================================================================
# 3. VLI discontinuity + reachability band + grid nodes
# =====================================================================================
def plot_vli_discontinuity(path_dict, specs, show=False, save=False):
    colors, _ = set_colors()
    scale = _axis_max(specs)
    start_age = specs["start_age"]
    # The binding VLI decision: retire at SRA - 2 (the deduction-free VLI age).
    ret_age = _ILLUSTRATIVE_SRA - 2
    period = ret_age - start_age + 1
    exp_years = np.linspace(0.0, scale, 600)

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    fig.suptitle(
        f"VLI discontinuity at retirement age {ret_age} (= SRA - 2)",
        fontweight="bold",
    )

    for sex in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            ax = axes[sex, edu]
            threshold = float(specs["experience_threshold_very_long_insured"][sex])

            pp = _sweep(specs, exp_years, period, sex, edu, health=0)
            ax.plot(exp_years, pp, color=colors[0])

            # Reachability band: experience from which the threshold is still reachable
            # within the 2-year VLI window.
            ax.axvspan(
                threshold - 2.0,
                threshold,
                color=colors[1],
                alpha=0.15,
                label="Reachability band",
            )
            ax.axvline(threshold, color=colors[3], linewidth=2, label="VLI threshold")

            # Production experience-grid nodes for this sex (real years).
            nodes = _grid_nodes(specs, sex)
            for node in nodes:
                ax.axvline(node, color="grey", linewidth=1, linestyle=":", alpha=0.7)
            ax.plot([], [], color="grey", linestyle=":", label="Grid nodes")

            ax.set_title(f"{SEX_LABELS[sex]}, {EDU_LABELS[edu]}")
            ax.set_xlabel("Experience (years)")
            ax.set_ylabel("Pension points at retirement")
            ax.grid(True, alpha=0.3)
            if sex == 0 and edu == 0:
                ax.legend()

    _finish(fig, path_dict, "retirement_vli_discontinuity", show, save)


# =====================================================================================
# 4. Informed vs uninformed early-retirement deductions
# =====================================================================================
def plot_informed_vs_uninformed_deductions(path_dict, specs, show=False, save=False):
    colors, _ = set_colors()
    # Years of early retirement before the SRA (long-insured path allows up to 4).
    years_early = np.linspace(0.0, 4.0, 200)

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    fig.suptitle("Early-retirement factor: informed vs uninformed", fontweight="bold")

    for sex in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            ax = axes[sex, edu]

            informed_factor = np.clip(1 - specs["ERP"] * years_early, 0.0, 1.0)
            uninformed_factor = np.clip(
                1 - specs["uninformed_ERP"][edu] * years_early, 0.0, 1.0
            )

            ax.plot(
                _ILLUSTRATIVE_SRA - years_early,
                informed_factor,
                color=colors[0],
                label="Informed",
            )
            ax.plot(
                _ILLUSTRATIVE_SRA - years_early,
                uninformed_factor,
                color=colors[1],
                linestyle="--",
                label="Uninformed",
            )

            ax.set_title(f"{SEX_LABELS[sex]}, {EDU_LABELS[edu]}")
            ax.set_xlabel("Retirement age")
            ax.set_ylabel("Pension factor")
            ax.grid(True, alpha=0.3)
            if sex == 0 and edu == 0:
                ax.legend()

    _finish(fig, path_dict, "retirement_deduction_information", show, save)


# =====================================================================================
# 5. Pension points by retirement timing at a fixed current age
# =====================================================================================
def plot_pension_by_retirement_timing(
    path_dict,
    specs,
    show=False,
    save=False,
    age_now=60,
    years_until_retirement=(3, 4, 5, 6, 7),
):
    """Pension points vs current experience, one line per retirement-timing choice.

    Fixes the person's current age and asks: if you work full time for N more years
    and then retire, how do your pension points depend on the experience you have
    today? Each line is a different N. Working longer both raises experience (shifts
    the curve up) and changes the retirement age (less deduction, and -- once the
    retirement age reaches SRA - 2 -- the deduction-free VLI jump). The dotted
    reachability marks show, for VLI-eligible lines, the current experience from
    which N more full-time years just reach the threshold.
    """
    colors, _ = set_colors()
    start_age = specs["start_age"]
    sra = _ILLUSTRATIVE_SRA
    period_now = age_now - start_age
    max_exp_now = float(specs["max_exps_period_working"][period_now])
    exp_now = np.linspace(0.0, max_exp_now, 400)

    fig, axes = plt.subplots(2, 2, figsize=get_figsize(2, 2))
    fig.suptitle(
        f"Pension points by retirement timing (current age {age_now}, SRA = {sra})",
        fontweight="bold",
    )

    for sex in range(specs["n_sexes"]):
        for edu in range(specs["n_education_types"]):
            ax = axes[sex, edu]
            threshold = float(specs["experience_threshold_very_long_insured"][sex])

            for i, years in enumerate(sorted(years_until_retirement)):
                ret_age = age_now + years
                ret_period = ret_age - start_age + 1
                # Full time each interim year adds one year of experience.
                exp_at_retirement = exp_now + years
                pp = _sweep(specs, exp_at_retirement, ret_period, sex, edu, health=0)
                color = colors[i % len(colors)]
                ax.plot(
                    exp_now,
                    pp,
                    color=color,
                    label=f"Retire at {ret_age} (in {years} yr)",
                )

                # The VLI jump appears only when retiring early (before the SRA) but
                # within its 2-year window; retiring exactly at the SRA is full-
                # pension regardless of experience, so it has no jump.
                if ret_age < sra and (sra - ret_age) <= 2:
                    exp_reach = threshold - years
                    if 0.0 <= exp_reach <= max_exp_now:
                        ax.axvline(exp_reach, color=color, linestyle=":", alpha=0.6)

            ax.set_title(f"{SEX_LABELS[sex]}, {EDU_LABELS[edu]}")
            ax.set_xlabel(f"Experience at age {age_now} (years)")
            ax.set_ylabel("Pension points at retirement")
            ax.grid(True, alpha=0.3)
            if sex == 0 and edu == 0:
                ax.legend()

    _finish(fig, path_dict, "retirement_pp_by_timing", show, save)


def plot_all_retirement_system(path_dict, specs, show=False, save=True):
    """Produce the full retirement-system plot suite."""
    plot_pension_points_from_experience(path_dict, specs, show=show, save=save)
    plot_pension_vs_retirement_age(path_dict, specs, show=show, save=save)
    plot_vli_discontinuity(path_dict, specs, show=show, save=save)
    plot_informed_vs_uninformed_deductions(path_dict, specs, show=show, save=save)
    plot_pension_by_retirement_timing(path_dict, specs, show=show, save=save)
