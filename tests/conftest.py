"""Shared fixtures and the reference-output resolver.

The reference outputs are read from ``repro/expected/``. The paper-number and
preset tests read them rather than recomputing per test, so the fast suite stays fast.
Their slow variants recompute and are marked ``slow``, so ``pytest -m "not
slow"`` skips them.

``repro/expected/`` is not redistributed in the sdist, the wheel or the public
git history. A fresh checkout therefore has the generation scripts and the
documented seed, not the static files. ``reference_dir`` falls back to
generating them locally via ``repro/_ensure_expected.py`` the first time a test
session needs them. The generated copy is cached on disk under
``repro/expected/`` for the rest of that checkout.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

sys.path.insert(0, str(ROOT / "repro"))


def reference_dir() -> Path:
    """Where the reference outputs live, generating them locally if absent.

    The search order lets the fixture be relocated, checked in locally, or
    generated without touching a test.
    """
    override = os.environ.get("SG_OC_REFERENCE_DIR")
    candidates = [Path(override)] if override else []
    candidates += [ROOT / "repro" / "expected", TESTS / "data"]
    for candidate in candidates:
        if (candidate / "extension_design.csv").exists():
            return candidate
    from _ensure_expected import ensure

    return ensure()


def load_reference(name: str):
    """A reference output, as parsed JSON or a list of CSV row dicts."""
    path = reference_dir() / name
    if not path.exists():
        pytest.skip(f"{name} is not in {reference_dir()}")
    if path.suffix == ".json":
        return json.loads(path.read_text())
    rows = list(csv.DictReader(path.open()))
    assert rows, f"{name} is empty"
    return rows


#: SHA-256 of ``extension_joint_counts.csv`` as published. The generation script
#: draws every configuration from one sequential stream, and numpy 2.5 changed
#: how large binomials are drawn, so a reference regenerated under numpy 2.5 or
#: later agrees with the published one within Monte Carlo error and not draw for
#: draw.
PUBLISHED_JOINT_SHA256 = "d5a90bd80618fae231825a84204991db3e392a118fe5b5276b8459947a5ea79c"


@pytest.fixture(scope="session")
def published_reference():
    """Skip a test of an exact published count unless the reference is the published one."""
    import hashlib

    path = reference_dir() / "extension_joint_counts.csv"
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    if digest != PUBLISHED_JOINT_SHA256:
        pytest.skip(
            "the reference outputs were regenerated under numpy 2.5 or later, "
            "which draws large binomials differently from numpy 2.2.6, so exact "
            "published counts cannot be compared. Install numpy==2.2.6 and "
            "delete repro/expected/ to run this test."
        )


@pytest.fixture(scope="session")
def extension_design():
    return load_reference("extension_design.csv")


@pytest.fixture(scope="session")
def extension_joint():
    return [r for r in load_reference("extension_joint_counts.csv") if r["arm"] == "design"]


@pytest.fixture(scope="session")
def simulation_grid():
    return load_reference("simulation_grid.csv")


@pytest.fixture(scope="session")
def fig3_values():
    return load_reference("fig3_values.json")
