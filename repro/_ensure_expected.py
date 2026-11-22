"""Ensure ``repro/expected/`` exists, generating it locally if it is absent.

``repro/expected/`` holds reference outputs for the paper's numbers. It is not
redistributed in the sdist, the wheel or the public git history. It is excluded
from ``MANIFEST.in`` and listed in ``.gitignore``.

Anyone building from source therefore starts with the scripts and the
documented seed, not the static files. This module regenerates the reference
outputs deterministically from them so the test suite and ``reproduce.sh`` work
on a fresh checkout. The result is cached on disk under ``repro/expected/`` once
per checkout. Delete the directory to force a fresh regeneration, for example
after changing the seed or the generation scripts.

MD-Agreement-derived files are not covered here: that corpus is distributed by
its own authors and is already handled as SKIPPED, not generated, when absent.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPRO = Path(__file__).resolve().parent
ROOT = REPRO.parent
EXPECTED = REPRO / "expected"

#: Files the fast test suite reads via ``conftest.load_reference``. Each is
#: produced by exactly one generation script, run once with results redirected
#: into ``EXPECTED``.
REQUIRED = (
    "extension_design.csv",
    "extension_realised.csv",
    "extension_joint_counts.csv",
    "simulation_grid.csv",
    "fig3_values.json",
)

#: Scripts that produce them, in dependency order. Each is deterministic given
#: the seeds it documents, so running them once under numpy 2.2.6 reproduces the
#: published numbers.
GENERATORS = ("run_grid.py", "run_realised.py", "run_extension_oc.py")


def ensure() -> Path:
    """Return ``repro/expected/``, generating any missing file into it first."""
    EXPECTED.mkdir(exist_ok=True)
    missing = [name for name in REQUIRED if not (EXPECTED / name).exists()]
    if not missing:
        return EXPECTED
    env = dict(os.environ, SG_RESULTS=str(EXPECTED))
    env.pop("SG_QUICK", None)
    for script in GENERATORS:
        completed = subprocess.run(
            [sys.executable, str(REPRO / script)], env=env, cwd=ROOT,
            capture_output=True, text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"generating the reference fixture failed running {script}:\n"
                f"{completed.stderr}"
            )
    still_missing = [name for name in REQUIRED if not (EXPECTED / name).exists()]
    if still_missing:
        raise RuntimeError(
            f"generation scripts ran but did not produce: {', '.join(still_missing)}"
        )
    return EXPECTED


if __name__ == "__main__":
    print(ensure())
