"""stabilitygate: a three-outcome test-retest stability screen for repeated-call LLM experiments.

This package operationalises the screen described and measured in

    The Screen Decides the Verdict: Operating Characteristics of
    Model-Exclusion Rules in LLM Evaluation (2026).
    https://doi.org/10.5281/zenodo.21366659
    See README section "Citation".

A correlation-based test-retest gate becomes uninformative or inverted when
responses sit near a ceiling or floor, because matched cells then hold too
little reliable between-cell variance for a correlation to read. This package
replaces the single-coefficient gate with a screen that returns one of three
verdicts, ``no_drift_detected``, ``FAIL``, or ``INDETERMINATE``, and with a
design-time calibration routine that reports the operating characteristics of a
planned design before any data are collected.

The admitting verdict is called ``no_drift_detected`` and not PASS on purpose.
Both component tests are calibrated against the design's own no-drift null, so
admission is a failure to reject no drift rather than a rejection of
consequential drift. Those two criteria have different error-control properties
and this package implements the weaker one. It is not an equivalence test.

Two entry points cover the two moments in a study:

    screen(...)  runs the three-outcome screen on observed main-run and
                 retest cell rates and returns a structured verdict.

    power(...)   maps the operating characteristics of a planned design
                 (cells, calls per cell, expected baseline and spread), and
                 recommend_design(...) searches for the smallest design that
                 meets a stated false-exclusion budget and drift-detection power.

The only hard dependency is numpy.
"""

from .screen import (
    FAIL,
    INDETERMINATE,
    NO_DRIFT_DETECTED,
    VERDICTS,
    ScreenResult,
    screen,
    screen_dataframe,
)
from .power import (
    DesignRecommendation,
    PowerRow,
    power,
    recommend_design,
)
from . import presets
from .presets import OC_2026_08_28, PRESETS

__all__ = [
    "screen",
    "screen_dataframe",
    "ScreenResult",
    "power",
    "recommend_design",
    "PowerRow",
    "DesignRecommendation",
    "presets",
    "PRESETS",
    "OC_2026_08_28",
    "NO_DRIFT_DETECTED",
    "FAIL",
    "INDETERMINATE",
    "VERDICTS",
]

__version__ = "0.1.0"
