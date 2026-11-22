#!/usr/bin/env python3
"""Stage 6. The paper's three figures.

Promoted into this harness from three separate figure scripts of 27 August 2026
and merged, so one command draws all three and each reads a stage output rather
than a path into someone's working tree.

  fig1_inversion   the four-model worked case, from the bundled data file, with
                   every annotation computed by ``stabilitygate.screen_dataframe``
                   rather than read from a precomputed summary
  fig2_size        false exclusion of the six screens under no drift, from
                   ``results/simulation_grid.csv`` written by stage 2
  fig3_regime      the size-matched comparison at the realised design, from
                   ``results/fig3_values.json`` written by stage 3

Read-only on all inputs. No randomness, so no seed.
"""
from __future__ import annotations

import csv
import json
import os
from importlib import resources
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from stabilitygate import screen_dataframe  # noqa: E402

RESULTS = Path(os.environ.get("SG_RESULTS", Path(__file__).resolve().parent.parent / "results"))
FIGURES = Path(os.environ.get("SG_FIGURES", Path(__file__).resolve().parent.parent / "figures"))
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": .6,
                     "xtick.major.width": .6, "ytick.major.width": .6, "figure.dpi": 300})
GREY = "#4a4a4a"
ACC = "#000000"

DISPLAY = {"claude-opus-4-7": "Claude Opus 4.7", "gemini-3.1-pro": "Gemini 3.1 Pro",
           "deepseek-v4-flash": "DeepSeek-V4-Flash", "gpt-5.5": "GPT-5.5"}


def _save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {FIGURES / stem}.pdf and .png")


def figure_1_inversion() -> None:
    """The worked case. A correlation gate and observed movement rank models differently."""
    with resources.files("stabilitygate.data").joinpath(
        "four_model_stability.csv"
    ).open("r") as f:
        rows = list(csv.DictReader(f))
    results = screen_dataframe(rows)
    cells = {}
    for key in DISPLAY:
        grp = [r for r in rows if r["model"] == key]
        cells[key] = (np.array([float(r["main_rate"]) for r in grp]),
                      np.array([float(r["stability_rate"]) for r in grp]))
    models = sorted(DISPLAY, key=lambda m: -results[m].mean_abs_drift)

    fig, ax = plt.subplots(2, 4, figsize=(5.5, 2.85), gridspec_kw={"height_ratios": [1, .8]})
    for j, m in enumerate(models):
        a, b = cells[m]
        res = results[m]
        gate = res.registered_gate_verdict
        top = ax[0, j]
        top.plot([0, 1], [0, 1], lw=.6, color="0.75", zorder=1)
        top.plot(a, b, "o", ms=3.2, mfc="none" if gate == "PASS" else ACC,
                 mec=ACC, mew=.8, zorder=3)
        top.set_xlim(-.05, 1.05)
        top.set_ylim(-.05, 1.05)
        top.set_title(f"{DISPLAY[m]}\n{gate}", fontsize=7.2,
                      fontweight="bold" if gate == "FAIL" else "normal")
        top.annotate(f"$r$={res.pearson_r:.3f}\n$\\lambda$={res.lambda_point:.2f}",
                     (.05, .93), fontsize=6.3, va="top")
        top.set_xticks([0, .5, 1])
        top.set_yticks([0, .5, 1])
        top.tick_params(length=2.2, labelsize=6.4)
        top.spines[["top", "right"]].set_visible(False)
        if j == 0:
            top.set_ylabel("re-run rate", fontsize=7)

        bot = ax[1, j]
        diff = np.abs(b - a)
        order = np.argsort(-diff)
        k = len(diff)
        bot.axhline(0, lw=.6, color="0.75")
        bot.vlines(np.arange(k), 0, diff[order], lw=1.0, color=GREY)
        bot.plot(np.arange(k), diff[order], "o", ms=2.4, color=ACC)
        bot.axhline(res.mean_abs_drift, ls=(0, (3, 2)), lw=.8, color=ACC)
        nz = int((diff == 0).sum())
        bot.annotate(f"mean |$\\Delta$|={res.mean_abs_drift:.3f}\n{nz}/{k} identical",
                     (k - 1, .60), fontsize=6.3, ha="right", va="top")
        bot.set_ylim(-.03, .62)
        bot.set_xlim(-1, k)
        bot.set_xticks([])
        bot.tick_params(length=2.2, labelsize=6.4)
        bot.spines[["top", "right", "bottom"]].set_visible(False)
        if j == 0:
            bot.set_ylabel("|change|", fontsize=7)
        bot.set_xlabel(f"{k} conditions", fontsize=6.6)
    ax[0, 0].set_xlabel("main-run rate", fontsize=7)
    fig.tight_layout(pad=.3, h_pad=.6, w_pad=.5)
    _save(fig, "fig1_inversion")


def figure_2_size() -> None:
    """False exclusion under no drift, median and range over the stable configurations."""
    path = RESULTS / "simulation_grid.csv"
    rows = [r for r in csv.DictReader(path.open()) if r["mode"] == "stable"]
    names = [("pearson", "Pearson r > .80"), ("spearman", "Spearman > .80"),
             ("ccc", "Lin's CCC > .80"), ("absdrift", "Absolute drift"),
             ("logit", "Log-odds drift"), ("chi2", "Unsigned, calibrated")]
    stats = []
    for key, label in names:
        v = np.array([float(r["err_" + key]) for r in rows])
        stats.append((label, np.median(v), v.min(), v.max()))
    stats.sort(key=lambda t: -t[1])

    fig, ax = plt.subplots(figsize=(5.4, 2.0))
    for i, (label, med, lo, hi) in enumerate(stats):
        ax.plot([lo, hi], [i, i], color=GREY, lw=1.1, solid_capstyle="butt", zorder=2)
        ax.plot([lo, lo], [i - .14, i + .14], color=GREY, lw=1.1)
        ax.plot([hi, hi], [i - .14, i + .14], color=GREY, lw=1.1)
        ax.plot(med, i, "o", color=ACC, ms=4.6, zorder=3)
        ax.annotate(f"{med:.3f}", (med, i), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.6)
    ax.axvline(.05, color=ACC, ls=(0, (3, 2)), lw=.8, zorder=1)
    ax.annotate("nominal .05", (.05, len(stats) - .35), textcoords="offset points",
                xytext=(5, 0), fontsize=6.6, va="center")
    ax.set_yticks(np.arange(len(stats)))
    ax.set_yticklabels([s[0] for s in stats])
    ax.invert_yaxis()
    ax.set_xlim(-.02, 1.05)
    ax.set_xlabel(f"False-exclusion rate under no drift "
                  f"(median, min-max over {len(rows)} designs)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5)
    fig.tight_layout(pad=.3)
    _save(fig, "fig2_size")


def figure_3_regime() -> None:
    """Size calibration at the realised design, from the stage 3 series."""
    values = json.loads((RESULTS / "fig3_values.json").read_text())
    bl = values["bl"]
    reg_fe, pw_p, pw_c, thr = values["reg_fe"], values["pw_p"], values["pw_c"], values["thr"]

    fig, ax = plt.subplots(1, 3, figsize=(5.5, 1.95))
    ax[0].plot(bl, reg_fe, "-o", ms=3, lw=1.1, color=ACC)
    ax[0].axhline(.05, ls=(0, (3, 2)), lw=.8, color=GREY)
    ax[0].annotate("nominal .05", (.52, .10), fontsize=6.2, color=GREY)
    ax[0].set_ylim(-.03, 1.03)
    ax[0].set_ylabel("rate")
    ax[0].set_title("(a) registered gate,\nfalse exclusion", fontsize=7.4)

    ax[1].plot(bl, pw_c, ls=(0, (1, 1.4)), marker="^", ms=3, lw=1.2, color=ACC, label="unsigned")
    ax[1].plot(bl, pw_p, "-o", ms=3, lw=1.1, color=GREY, label="Pearson")
    ax[1].axhline(.05, ls=(0, (3, 2)), lw=.8, color=GREY)
    ax[1].set_ylim(-.03, 1.03)
    ax[1].legend(frameon=False, fontsize=6.4, loc="upper right", handlelength=1.6)
    ax[1].set_title("(b) power at matched\n5% false exclusion", fontsize=7.4)

    ax[2].plot(bl, thr, "-o", ms=3, lw=1.1, color=ACC)
    ax[2].axhline(0, lw=.7, color=GREY)
    ax[2].axhline(.80, ls=(0, (3, 2)), lw=.8, color=GREY)
    ax[2].annotate("registered .80", (.52, .84), fontsize=6.2, color=GREY)
    ax[2].fill_between(bl, thr, 0, where=np.array(thr) < 0, color="0.85", zorder=0)
    ax[2].set_ylim(-.45, 1.02)
    ax[2].set_title("(c) threshold that would\ngive 5% false exclusion", fontsize=7.4)

    for a_ in ax:
        a_.set_xlabel("baseline $p_0$", fontsize=7.4)
        a_.spines[["top", "right"]].set_visible(False)
        a_.tick_params(length=2.5, labelsize=7)
        a_.set_xlim(.47, 1.02)
    fig.tight_layout(pad=.35, w_pad=1.1)
    _save(fig, "fig3_regime")


def main() -> None:
    print("Stage 6, figures")
    figure_1_inversion()
    figure_2_size()
    figure_3_regime()


if __name__ == "__main__":
    main()
