#!/usr/bin/env python3
"""Stage 7. Recompute every published number and report PASS, FAIL or SKIPPED.

Reads ``paper_numbers.json``, recomputes each entry from the outputs the earlier
stages wrote, and prints one row per claim. Exits 0 when nothing failed and 1
otherwise, so continuous integration can gate on it.

Each entry names a recipe rather than carrying code. Recipes live in the
``RECIPES`` registry below, so nothing in the JSON is evaluated and a reader can
follow any claim from the number in the paper to the arithmetic behind it.

Tolerances are Monte Carlo, at two standard errors on the reported quantity, not
an arbitrary epsilon. The paper's Appendix A states the yardstick, Monte Carlo
standard error on any reported rate is at most .008 at 4,000 replicates and .004
at 20,000. Integer counts carry a tolerance of zero.

A claim whose inputs are absent is SKIPPED, not FAILED. That is how the
MD-Agreement rows behave when the corpus is not present, since the corpus is
distributed by its own authors and is not redistributed here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from importlib import resources
from pathlib import Path

REPRO = Path(__file__).resolve().parent
ROOT = REPRO.parent
RESULTS = Path(os.environ.get("SG_RESULTS", ROOT / "results"))
#: Searched when a file is not in RESULTS. A smoke run writes reduced-replicate
#: outputs, so it verifies the deposited reference outputs instead and points
#: this at its own scratch directory for anything only it produced.
EXTRA: list[Path] = []

SEED = 20260101


# --------------------------------------------------------------------------- #
# Terminal presentation
# --------------------------------------------------------------------------- #
class Style:
    """Colour, but only when a terminal is there to render it."""

    def __init__(self, stream) -> None:
        self.on = (
            hasattr(stream, "isatty")
            and stream.isatty()
            and os.environ.get("TERM", "") not in ("", "dumb")
            and "NO_COLOR" not in os.environ
        )

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def ok(self, t): return self(t, "32")
    def bad(self, t): return self(t, "31;1")
    def dim(self, t): return self(t, "2")
    def head(self, t): return self(t, "1")


class Missing(Exception):
    """An input this claim needs is not present, so the claim is skipped."""


# --------------------------------------------------------------------------- #
# Loading, with one cache per file so 40 claims read each input once
# --------------------------------------------------------------------------- #
_CACHE: dict[str, object] = {}


def _load(name: str):
    if name in _CACHE:
        return _CACHE[name]
    if name == "worked_case":
        from stabilitygate import screen_dataframe
        with resources.files("stabilitygate.data").joinpath(
            "four_model_stability.csv"
        ).open("r") as f:
            rows = list(csv.DictReader(f))
        value = (rows, screen_dataframe(rows, compute_gate_false_exclusion=True))
    else:
        for directory in [RESULTS, *EXTRA]:
            path = directory / name
            if path.exists():
                break
        else:
            raise Missing(str(RESULTS / name))
        if path.suffix == ".json":
            value = json.loads(path.read_text())
        else:
            value = list(csv.DictReader(path.open()))
    _CACHE[name] = value
    return value


def _design_rows(mode: str) -> list[dict]:
    return [r for r in _load("extension_design.csv") if r["mode"] == mode]


def _joint_index() -> dict:
    rows = [r for r in _load("extension_joint_counts.csv") if r["arm"] == "design"]
    return {
        (r["baseline"], r["sigma_logit"], r["K_cells"], r["n_per_cell"], r["mode"]): r
        for r in rows
    }


def _key(r: dict) -> tuple:
    return (r["baseline"], r["sigma_logit"], r["K_cells"], r["n_per_cell"], r["mode"])


def _resolvable_and_excluded(j: dict) -> tuple[int, int]:
    g = lambda c: int(j[c])  # noqa: E731
    resolvable = (g("n_admit_gatepass") + g("n_admit_gatefail")
                  + g("n_exclude_gatepass") + g("n_exclude_gatefail"))
    excluded = g("n_exclude_gatepass") + g("n_exclude_gatefail")
    return resolvable, excluded


def _gate_failed(j: dict) -> int:
    return int(j["n_admit_gatefail"]) + int(j["n_exclude_gatefail"])


def _conditional(r: dict, numerator: str) -> float:
    admit, exclude = float(r["proposed_admit"]), float(r["proposed_exclude"])
    denom = admit + exclude
    if denom <= 0:
        return float("nan")
    return (exclude if numerator == "exclude" else admit) / denom


def _well_resolved_keys() -> set:
    return {
        _key(r)[:4] for r in _design_rows("stable")
        if float(r["proposed_indeterminate"]) < 0.10
    }


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
def r_worked_case(model: str, field: str) -> float:
    rows, results = _load("worked_case")
    res = results[model]
    if field == "identical_conditions":
        grp = [x for x in rows if x["model"] == model]
        return float(sum(1 for x in grp
                         if float(x["main_rate"]) == float(x["stability_rate"])))
    return float(getattr(res, field))


def r_grid_stat(screen: str, stat: str) -> float:
    values = [float(r["err_" + screen])
              for r in _load("simulation_grid.csv") if r["mode"] == "stable"]
    return {"median": statistics.median, "min": min, "max": max}[stat](values)


def r_grid_count(what: str) -> float:
    rows = _load("simulation_grid.csv")
    if what == "rows":
        return float(len(rows))
    return float(sum(1 for r in rows if r["mode"] == "stable"))


def r_fig3(series: str, baseline: float) -> float:
    values = _load("fig3_values.json")
    if series not in values:
        raise Missing(f"fig3_values.json has no series {series!r}")
    return float(values[series][values["bl"].index(baseline)])


def r_pooled_conditional_size(quantity: str) -> float:
    joint = _joint_index()
    resolvable = excluded = 0
    for r in _design_rows("stable"):
        res, exc = _resolvable_and_excluded(joint[_key(r)])
        resolvable += res
        excluded += exc
    return float(resolvable) if quantity == "draws" else excluded / resolvable


def r_conditional_size(stat: str, threshold: float = 0.10) -> float:
    values = [_conditional(r, "exclude") for r in _design_rows("stable")]
    values = [v for v in values if v == v]
    if stat == "max":
        return max(values)
    return float(sum(1 for v in values if v > threshold))


def r_size_by_resolution(subset: str, quantity: str) -> float:
    well = _well_resolved_keys()
    rows = [r for r in _design_rows("stable")
            if (_key(r)[:4] in well) == (subset == "well_resolved")]
    if quantity == "configurations":
        return float(len(rows))
    joint = _joint_index()
    resolvable = excluded = 0
    for r in rows:
        res, exc = _resolvable_and_excluded(joint[_key(r)])
        resolvable += res
        excluded += exc
    return excluded / resolvable


def r_false_admission(denominator: str, stat: str, threshold: float = 0.05) -> float:
    rows = _design_rows("shift_1.0")
    if denominator == "conditional":
        values = [v for v in (_conditional(r, "admit") for r in rows) if v == v]
    else:
        values = [float(r["proposed_admit"]) for r in rows]
    if stat == "max":
        return max(values)
    return float(sum(1 for v in values if v > threshold))


def r_indeterminate(stat: str) -> float:
    values = [float(r["proposed_indeterminate"]) for r in _design_rows("stable")]
    return {"median": statistics.median, "max": max}[stat](values)


def r_well_resolved(who: str, quantity: str, mode: str = "shift_1.0") -> float:
    """Median over the well-resolved configurations.

    ``incumbent`` is the unconditional gate rate, which is the construction
    behind the published figure and the only one the marginals support.
    ``incumbent_conditional`` restricts to draws the screen found resolvable,
    which is the like-for-like comparison and needs the joint counts.
    """
    well = _well_resolved_keys()
    rows = [r for r in _design_rows("stable" if quantity == "size" else mode)
            if _key(r)[:4] in well]
    if who == "proposed":
        values = [_conditional(r, "exclude") for r in rows]
    elif who == "incumbent":
        values = [1 - float(r["pearson_pass"]) for r in rows]
    else:
        joint = _joint_index()
        values = []
        for r in rows:
            j = joint[_key(r)]
            res, _ = _resolvable_and_excluded(j)
            values.append(_gate_failed(j) / res if res else float("nan"))
    values = [v for v in values if v == v]
    return statistics.median(values)


def r_incumbent_full_grid(mode: str) -> float:
    return statistics.median(
        [1 - float(r["pearson_pass"]) for r in _design_rows(mode)]
    )


def r_direction_symmetry(mode: str) -> float:
    well = _well_resolved_keys()
    rows = [r for r in _design_rows(mode) if _key(r)[:4] in well]
    return statistics.median([v for v in (_conditional(r, "exclude") for r in rows) if v == v])


def r_incumbent_pooled_conditional(mode: str) -> float:
    joint = _joint_index()
    resolvable = failed = 0
    for r in _design_rows(mode):
        j = joint[_key(r)]
        res, _ = _resolvable_and_excluded(j)
        resolvable += res
        failed += _gate_failed(j)
    return failed / resolvable


def r_joint_rows_partition() -> float:
    """1.0 when every joint row's six screen-by-gate cells sum to its n_rep."""
    for j in _load("extension_joint_counts.csv"):
        res, _ = _resolvable_and_excluded(j)
        total = res + int(j["n_indet_gatepass"]) + int(j["n_indet_gatefail"])
        if total != int(j["n_rep"]):
            return 0.0
    return 1.0


def r_panel(field: str) -> float:
    return float(_load("mdagreement_summary.json")[field])


RECIPES = {
    "worked_case": r_worked_case,
    "grid_stat": r_grid_stat,
    "grid_count": r_grid_count,
    "fig3": r_fig3,
    "pooled_conditional_size": r_pooled_conditional_size,
    "conditional_size": r_conditional_size,
    "size_by_resolution": r_size_by_resolution,
    "false_admission": r_false_admission,
    "indeterminate": r_indeterminate,
    "well_resolved": r_well_resolved,
    "incumbent_full_grid": r_incumbent_full_grid,
    "direction_symmetry": r_direction_symmetry,
    "incumbent_pooled_conditional": r_incumbent_pooled_conditional,
    "joint_rows_partition": r_joint_rows_partition,
    "panel": r_panel,
}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _fmt(value: float | None, integer: bool) -> str:
    if value is None:
        return "--"
    if integer:
        return f"{int(round(value)):,}"
    return f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recompute every published number from the reproduction outputs."
    )
    parser.add_argument("--claims", default=str(REPRO / "paper_numbers.json"),
                        help="path to paper_numbers.json")
    parser.add_argument("--results", default=None,
                        help="directory holding the reproduction outputs "
                             "(default: results/, or $SG_RESULTS)")
    parser.add_argument("--extra", default=None,
                        help="a second directory searched when a file is not in "
                             "--results, used by the smoke run")
    parser.add_argument("--verbose", action="store_true",
                        help="print the source and recipe of every claim")
    args = parser.parse_args(argv)

    global RESULTS
    if args.results:
        RESULTS = Path(args.results)
    if args.extra:
        EXTRA.append(Path(args.extra))

    style = Style(sys.stdout)
    claims = json.loads(Path(args.claims).read_text())["claims"]

    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:                                   # pragma: no cover
        numpy_version = "absent"
    python_version = ".".join(str(v) for v in sys.version_info[:3])

    print(style.head("stabilitygate reproduction check"))
    print(style.dim(f"seed {SEED}   numpy {numpy_version}   python {python_version}"))
    print(style.dim(f"results {RESULTS}"))
    for directory in EXTRA:
        print(style.dim(f"        {directory}"))
    print()

    id_width = max(34, min(46, max(len(c["id"]) for c in claims)))
    header = (f"  {'id':<{id_width}}{'claim':>12}{'computed':>12}"
              f"{'tol':>10}   status")
    print(style.head(header))
    print("  " + "-" * (len(header) - 2))

    tally = {"PASS": 0, "FAIL": 0, "SKIPPED": 0}
    failures: list[str] = []
    for claim in claims:
        integer = bool(claim.get("integer"))
        tolerance = float(claim["tolerance"])
        try:
            computed = float(RECIPES[claim["recipe"]](**claim.get("args", {})))
        except Missing as exc:
            status, computed, note = "SKIPPED", None, str(exc)
        else:
            ok = abs(computed - float(claim["claim"])) <= tolerance
            status, note = ("PASS" if ok else "FAIL"), ""
        tally[status] += 1
        painted = {"PASS": style.ok, "FAIL": style.bad, "SKIPPED": style.dim}[status](status)
        print(f"  {claim['id']:<{id_width}}{_fmt(float(claim['claim']), integer):>12}"
              f"{_fmt(computed, integer):>12}{_fmt(tolerance, integer):>10}   {painted}"
              + (style.dim(f"  ({Path(note).name})") if status == "SKIPPED" and note else ""))
        if args.verbose:
            print(style.dim(f"      source   {claim['source']}"))
            print(style.dim(f"      from     {claim['computed_from']}"))
            print(style.dim(f"      recipe   {claim['recipe']} {claim.get('args', {})}"))
        if status == "FAIL":
            failures.append(
                f"{claim['id']}: claimed {claim['claim']}, computed {computed:.6g}, "
                f"tolerance {tolerance:g} ({claim['tolerance_basis']})"
            )

    print()
    summary = (f"  {tally['PASS']} PASS, {tally['FAIL']} FAIL, "
               f"{tally['SKIPPED']} SKIPPED")
    print(style.bad(summary) if tally["FAIL"] else style.ok(summary))
    if failures:
        print()
        for line in failures:
            print(style.bad("  FAIL ") + line)
    return 1 if tally["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
