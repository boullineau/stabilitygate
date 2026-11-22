"""Hygiene and packaging metadata.

Ordinary release checks, and one that asserts the author metadata is present in
every file that should carry it.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT
import stabilitygate
from stabilitygate.__main__ import main

#: The command line is exercised in process rather than in a subprocess, so the
#: coverage run sees it and a failure surfaces as a traceback rather than as an
#: exit code. One test below still spawns a subprocess, to cover the
#: ``python -m stabilitygate`` entry path itself.
DATA = ROOT / "stabilitygate" / "data" / "four_model_stability.csv"


def _cli(*args: str) -> int:
    return main(list(args))


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #
SMOKE = {
    "demo": ("demo",),
    "screen": ("screen", str(DATA)),
    "power": ("power", "--K", "14", "--n", "20", "--baseline", "0.9",
              "--n-rep", "60", "--n-boot", "40"),
    "recommend": ("recommend", "--baseline", "0.7", "--drift", "0.5",
                  "--max-false-exclusion", "0.2", "--min-power", "0.5",
                  "--n-rep", "60", "--n-boot", "40"),
}


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_cli_all_subcommands_exit_zero(name, capsys):
    """Each of the four subcommands runs and returns 0, at smoke-test scale."""
    assert _cli(*SMOKE[name]) == 0
    assert capsys.readouterr().out.strip(), f"{name} printed nothing"


def test_cli_entry_point_runs_as_a_module():
    """``python -m stabilitygate`` reaches the same parser."""
    done = subprocess.run([sys.executable, "-m", "stabilitygate", "demo"],
                          capture_output=True, text=True, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr
    assert "no_drift_detected" in done.stdout


def test_cli_writes_csv(tmp_path, capsys):
    """``--out`` produces a readable CSV carrying the settings that made it."""
    out = tmp_path / "screened.csv"
    assert _cli("screen", str(DATA), "--out", str(out)) == 0
    capsys.readouterr()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 4
    for column in ("group", "verdict", "lambda_point", "alpha_component",
                   "alpha_split", "precedence", "combine", "run_sided"):
        assert column in rows[0], column
    assert {r["verdict"] for r in rows} <= set(stabilitygate.VERDICTS)

    power_out = tmp_path / "power.csv"
    assert _cli("power", "--K", "14", "--n", "20", "--n-rep", "60",
                "--n-boot", "40", "--out", str(power_out)) == 0
    capsys.readouterr()
    rows = list(csv.DictReader(power_out.open()))
    assert rows and "final_pass" in rows[0] and "precedence" in rows[0]

    rec_out = tmp_path / "designs.csv"
    assert _cli("recommend", "--baseline", "0.7", "--drift", "0.5",
                "--max-false-exclusion", "0.2", "--min-power", "0.5",
                "--n-rep", "60", "--n-boot", "40", "--out", str(rec_out)) == 0
    capsys.readouterr()
    rows = list(csv.DictReader(rec_out.open()))
    assert rows and "total_calls" in rows[0]


def test_cli_passes_the_decision_rule_through(capsys):
    """A flag on the command line reaches the screen and changes the output."""
    assert _cli("screen", str(DATA), "--precedence", "resolution_first",
                "--alpha-split", "none", "--combine", "and") == 0
    out = capsys.readouterr().out
    assert "resolution_first" in out
    assert "'none' split" in out
    assert "combined with 'and'" in out


def test_cli_rejects_an_unknown_setting():
    with pytest.raises(SystemExit) as raised:
        _cli("power", "--K", "14", "--n", "20", "--precedence", "floor_first")
    assert raised.value.code != 0


def test_cli_requires_a_subcommand():
    with pytest.raises(SystemExit):
        _cli()


def test_demo_names_the_verdict_it_returns(capsys):
    """The worked-example output reads as something a reader can paste and trust."""
    assert _cli("demo") == 0
    out = capsys.readouterr().out
    assert "no_drift_detected" in out
    assert "INDETERMINATE" in out
    assert "failure to reject no drift" in out
    # the bare gate keeps its own labels, and the inversion is visible
    assert "bare gate" in out
    # fixed-width columns, so the table lines up
    lines = [line for line in out.splitlines() if line.startswith("claude-opus-4-7")]
    assert lines and lines[0].startswith("claude-opus-4-7     ")


def test_power_cli_table_is_aligned(capsys):
    """Fixed-width columns and consistent precision, so it pastes into a report."""
    assert _cli("power", "--K", "14", "--n", "20", "--baseline", "0.9",
                "--n-rep", "60", "--n-boot", "40") == 0
    lines = capsys.readouterr().out.splitlines()
    header, rule = lines[0], lines[1]
    assert len(rule) == len(header) and set(rule) == {"-"}
    body = [line for line in lines[2:] if line.strip()]
    assert len({len(line.rstrip()) - len(line.rstrip().split()[-1]) for line in body}) == 1


# --------------------------------------------------------------------------- #
# Packaging metadata
# --------------------------------------------------------------------------- #
def _pyproject() -> dict:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def test_version_matches_pyproject():
    assert stabilitygate.__version__ == _pyproject()["project"]["version"]


def test_setuptools_floor_supports_pep_639():
    """PEP 639 metadata needs setuptools 77.0.3, not the 61.0 the package declared."""
    requires = _pyproject()["build-system"]["requires"]
    floor = next(r for r in requires if r.startswith("setuptools"))
    version = tuple(int(p) for p in floor.split(">=")[1].split("."))
    assert version >= (77, 0, 3), floor


def test_py_typed_marker_is_present_and_packaged():
    assert (ROOT / "stabilitygate" / "py.typed").exists()
    package_data = _pyproject()["tool"]["setuptools"]["package-data"]["stabilitygate"]
    assert "py.typed" in package_data


def test_bundled_data_carries_a_licence():
    """Code is MIT and the bundled data file is CC BY 4.0, as elsewhere in the programme."""
    licence = ROOT / "stabilitygate" / "data" / "LICENSE"
    assert licence.exists()
    text = licence.read_text()
    assert "CC BY 4.0" in text or "Attribution 4.0" in text
    assert "MIT" in (ROOT / "LICENSE").read_text()


def test_citation_cff_parses():
    """Valid YAML with the keys GitHub and Zenodo read, and no placeholder left."""
    yaml = pytest.importorskip("yaml")
    path = ROOT / "CITATION.cff"
    assert path.exists()
    data = yaml.safe_load(path.read_text())
    for key in ("cff-version", "type", "title", "version", "license",
                "authors", "repository-code", "date-released",
                "preferred-citation"):
        assert key in data, key
    assert data["cff-version"] == "1.2.0"
    assert data["type"] == "software"
    assert data["license"] == "MIT"
    assert data["version"] == stabilitygate.__version__
    assert len(data["authors"]) == 2
    assert data["repository-code"].startswith("https://github.com/")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data["date-released"]))
    assert all("family-names" in a and "orcid" in a for a in data["authors"])
    assert "PENDING" not in path.read_text()


def test_changelog_is_keep_a_changelog_with_iso_dates():
    text = (ROOT / "CHANGELOG.md").read_text()
    assert "Keep a Changelog" in text
    headings = re.findall(r"^## \[?([^\]\s]+)\]?(?: - (\S+))?", text, re.MULTILINE)
    assert headings, "no release headings found"
    for version, date in headings:
        assert date, f"heading {version!r} carries no date"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date), f"{version}: {date!r} is not ISO"


# --------------------------------------------------------------------------- #
# Author metadata
# --------------------------------------------------------------------------- #
def test_author_metadata_is_present():
    """The licence, the citation file and the build metadata name the authors."""
    for name in ("LICENSE", "CITATION.cff", "pyproject.toml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "Boullineau" in text, name
        assert "Arciniegas" in text, name
        assert "PENDING" not in text, name
    assert "The stabilitygate authors" not in (ROOT / "LICENSE").read_text()


def test_no_build_artefacts_are_tracked():
    """The stale build artefacts are gone and the ignore file covers their return."""
    ignore = (ROOT / ".gitignore").read_text()
    for pattern in ("dist/", "build/", "__pycache__/", "*.egg-info/"):
        assert pattern in ignore, pattern
    # a freshly built pair for the current version is fine, since the release
    # build writes into the same directory.
    dist = ROOT / "dist"
    if dist.exists():
        wrong = [p.name for p in dist.iterdir()
                 if p.suffix in {".whl", ".gz"} and stabilitygate.__version__ not in p.name]
        assert not wrong, f"dist/ carries artefacts of another version: {wrong}"
    assert not (ROOT / "build").exists() or not any((ROOT / "build").iterdir())
