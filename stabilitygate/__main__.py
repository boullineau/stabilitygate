"""Command-line interface for the stability screen.

Examples
--------
Run the screen on a long-format CSV (one row per cell, a column per model)::

    python -m stabilitygate screen mydata.csv --n-col main_n

Reproduce the paper's four-model worked case::

    python -m stabilitygate demo

Map the operating characteristics of a planned design::

    python -m stabilitygate power --K 14 --n 20 --baseline 0.9

Recommend the smallest design meeting a false-exclusion and power target::

    python -m stabilitygate recommend --baseline 0.9 --drift 0.5 \
        --max-false-exclusion 0.05 --alpha 0.025
"""

from __future__ import annotations

import argparse
import csv
import sys
from importlib import resources
from pathlib import Path

from .power import power as run_power
from . import _core, recommend_design, screen, screen_dataframe


def _load_rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


#: The decision-rule flags shared by the screen, power and recommend subcommands.
_RULE_FLAGS = (
    "alpha_split",
    "run_sided",
    "combine",
    "precedence",
    "level_statistic",
    "heterogeneity_statistic",
    "null",
)


def _rule_kwargs(args: argparse.Namespace) -> dict:
    """Collect whichever decision-rule flags this subcommand declared."""
    return {k: getattr(args, k) for k in _RULE_FLAGS if hasattr(args, k)}


def _add_rule_flags(parser: argparse.ArgumentParser, with_null: bool) -> None:
    """Add the decision-rule flags, with the package defaults."""
    parser.add_argument(
        "--alpha-split", choices=list(_core.ALPHA_SPLITS), default="bonferroni",
        help="how alpha is divided between the two component tests "
             "(default: bonferroni, each at alpha/2)",
    )
    parser.add_argument(
        "--run-sided", choices=list(_core.RUN_SIDES), default="two",
        help="tail of the level test (default: two)",
    )
    parser.add_argument(
        "--combine", choices=list(_core.COMBINES), default="or",
        help="exclude when either test rejects, or only when both do (default: or)",
    )
    parser.add_argument(
        "--precedence", choices=list(_core.PRECEDENCES), default="resolution_first",
        help="whether a detected run effect may rule below the information floor "
             "(default: resolution_first, it may not)",
    )
    parser.add_argument(
        "--level-statistic", choices=list(_core.LEVEL_STATISTICS), default="woolf_z",
        help="the level statistic (default: woolf_z)",
    )
    parser.add_argument(
        "--heterogeneity-statistic", choices=list(_core.HETEROGENEITY_STATISTICS),
        default="cochran_q", help="the heterogeneity statistic (default: cochran_q)",
    )
    if with_null:
        parser.add_argument(
            "--null", choices=list(_core.NULLS), default="bootstrap",
            help="the null the tests are calibrated against. 'oracle' draws from the "
                 "true design parameters and is available only in simulation "
                 "(default: bootstrap)",
        )


def _cmd_screen(args: argparse.Namespace) -> int:
    rows = _load_rows(args.csv)
    results = screen_dataframe(
        rows,
        group_key=args.group_key,
        main_col=args.main_col,
        retest_col=args.retest_col,
        n_col=args.n_col,
        input_kind=args.input_kind,
        lambda_floor=args.lambda_floor,
        alpha=args.alpha,
        lambda_rule=args.lambda_rule,
        gate_threshold=args.gate_threshold,
        compute_gate_false_exclusion=args.gate_false_exclusion,
        seed=args.seed,
        **_rule_kwargs(args),
    )
    for key, res in results.items():
        print(f"\n=== {key} ===")
        print(res)
    if args.out:
        _write_results_csv(results, args.out)
        print(f"\nWrote {args.out}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    with resources.files("stabilitygate.data").joinpath(
        "four_model_stability.csv"
    ).open("r") as f:
        rows = list(csv.DictReader(f))
    results = screen_dataframe(
        rows, compute_gate_false_exclusion=True, seed=args.seed
    )
    order = ["claude-opus-4-7", "gemini-3.1-pro", "deepseek-v4-flash", "gpt-5.5"]
    order += [k for k in results if k not in order]
    print("Four-model worked case from The Screen Decides the Verdict.\n")
    header = (
        f"{'model':20}{'lambda':>8}{'z_run':>8}{'Q':>7}{'r':>8}"
        f"{'bare gate':>11}{'P(fail|stable)':>16}   screen verdict"
    )
    # the bare gate's own labels stay PASS and FAIL: they describe the incumbent
    # correlation gate, not this screen's verdict.
    print(header)
    print("-" * len(header))
    for key in order:
        if key not in results:
            continue
        r = results[key]
        pf = f"{r.p_fail_if_stable * 100:.1f}%" if r.p_fail_if_stable is not None else "-"
        print(
            f"{key:20}{r.lambda_point:>8.2f}{r.z_run:>8.2f}{r.Q_interaction:>7.1f}"
            f"{r.pearson_r:>8.3f}{r.registered_gate_verdict:>11}{pf:>16}   {r.verdict}"
        )
    print(
        "\nThe bare correlation gate PASSES Claude, the model with the largest mean\n"
        "absolute change in response rate (.129), and FAILS GPT-5.5, the model with\n"
        "the smallest (.018). The screen FAILS Claude on its run effect and returns\n"
        "INDETERMINATE for GPT-5.5, whose design carried too little information to rule.\n"
        "\nThe admitting verdict is 'no_drift_detected', not 'pass'. Both component\n"
        "tests are calibrated against the design's own no-drift null, so it records a\n"
        "failure to reject no drift, not a rejection of consequential drift."
    )
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    rows = run_power(
        K=args.K,
        n_calls=args.n,
        baseline=args.baseline,
        sigma_logit=args.sigma,
        lambda_floor=args.lambda_floor,
        alpha=args.alpha,
        lambda_rule=args.lambda_rule,
        n_rep=args.n_rep,
        n_boot=args.n_boot,
        seed=args.seed,
        **_rule_kwargs(args),
    )
    header = (
        f"{'baseline':>9}{'mode':>11}{'no drift':>10}{'fail':>7}"
        f"{'indet':>7}{'lam>=floor':>11}  role"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.baseline:>9.2f}{r.mode:>11}{r.final_pass:>10.2f}{r.final_fail:>7.2f}"
            f"{r.final_indeterminate:>7.2f}{r.pct_lambda_ge_floor:>11.2f}  {r.role}"
        )
    if args.out:
        _write_rows_csv([r.to_dict() for r in rows], args.out)
        print(f"\nWrote {args.out}")
    return 0


def _cmd_recommend(args: argparse.Namespace) -> int:
    rec = recommend_design(
        baseline=args.baseline,
        drift_logit_that_matters=args.drift,
        sigma_logit=args.sigma,
        target_false_exclusion=args.max_false_exclusion,
        target_power=args.min_power,
        lambda_floor=args.lambda_floor,
        alpha=args.alpha,
        lambda_rule=args.lambda_rule,
        n_rep=args.n_rep,
        n_boot=args.n_boot,
        seed=args.seed,
        **_rule_kwargs(args),
    )
    print(rec)
    if args.out and rec.evaluated:
        _write_rows_csv(rec.evaluated, args.out)
        print(f"\nWrote {args.out}")
    return 0


def _write_results_csv(results: dict, path: str) -> None:
    rows = []
    for key, res in results.items():
        d = res.to_dict()
        d = {"group": key, **d}
        rows.append(d)
    _write_rows_csv(rows, path)


def _write_rows_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stabilitygate",
        description="Three-outcome test-retest stability screen for repeated-call LLM experiments.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common_seed = dict(type=int, default=20260101, help="base random seed")

    s = sub.add_parser("screen", help="run the screen on observed cell rates in a CSV")
    s.add_argument("csv", help="long-format CSV, one row per matched cell")
    s.add_argument("--group-key", default="model")
    s.add_argument("--main-col", default="main_rate")
    s.add_argument("--retest-col", default="stability_rate")
    s.add_argument("--n-col", default="main_n")
    s.add_argument("--input-kind", choices=["auto", "rate", "count"], default="auto",
                   help="interpret cell values as rates, counts, or auto-detect")
    s.add_argument("--lambda-floor", type=float, default=0.50)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--lambda-rule", choices=["point", "lower"], default="point")
    s.add_argument("--gate-threshold", type=float, default=0.80)
    s.add_argument("--gate-false-exclusion", action="store_true",
                   help="also estimate the bare gate's false-exclusion rate")
    _add_rule_flags(s, with_null=False)
    s.add_argument("--out", help="optional path to write a results CSV")
    s.add_argument("--seed", **common_seed)
    s.set_defaults(func=_cmd_screen)

    d = sub.add_parser("demo", help="reproduce the paper's four-model worked case")
    d.add_argument("--seed", **common_seed)
    d.set_defaults(func=_cmd_demo)

    w = sub.add_parser("power", help="operating characteristics of a planned design")
    w.add_argument("--K", type=int, required=True, help="number of matched cells")
    w.add_argument("--n", type=int, required=True, help="calls per cell")
    w.add_argument("--baseline", type=float, nargs="+", default=[0.90],
                   help="one or more expected response rates")
    w.add_argument("--sigma", type=float, default=0.50, help="between-cell spread, log-odds")
    w.add_argument("--lambda-floor", type=float, default=0.50)
    w.add_argument("--alpha", type=float, default=0.05)
    w.add_argument("--lambda-rule", choices=["point", "lower"], default="point")
    _add_rule_flags(w, with_null=True)
    w.add_argument("--n-rep", type=int, default=1500)
    w.add_argument("--n-boot", type=int, default=300)
    w.add_argument("--out", help="optional path to write a results CSV")
    w.add_argument("--seed", **common_seed)
    w.set_defaults(func=_cmd_power)

    r = sub.add_parser("recommend", help="smallest design meeting a false-exclusion and power target")
    r.add_argument("--baseline", type=float, required=True)
    r.add_argument("--drift", type=float, required=True,
                   help="the log-odds shift you want to be able to detect")
    r.add_argument("--sigma", type=float, default=0.50)
    r.add_argument("--max-false-exclusion", type=float, default=0.05)
    r.add_argument("--min-power", type=float, default=0.80)
    r.add_argument("--lambda-floor", type=float, default=0.50)
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--lambda-rule", choices=["point", "lower"], default="point")
    _add_rule_flags(r, with_null=True)
    r.add_argument("--n-rep", type=int, default=1200)
    r.add_argument("--n-boot", type=int, default=250)
    r.add_argument("--out", help="optional path to write the evaluated-designs CSV")
    r.add_argument("--seed", **common_seed)
    r.set_defaults(func=_cmd_recommend)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
