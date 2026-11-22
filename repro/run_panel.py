#!/usr/bin/env python3
"""Stage 5. Panel loss in a published five-annotator corpus.

Promoted into this harness from the quorum analysis of 27 August 2026, rewritten
to take the corpus as an argument and to use only the standard library, so the
harness keeps numpy as its one dependency.

WHAT IT COMPUTES
MD-Agreement is a published offensive-language corpus in which each item carries
labels from exactly five identified annotators drawn from a pool of 819. For
every single-annotator removal the script asks two questions.

  1. Do the four survivors split two-two, so that a recomputed majority has no
     majority to recompute. Those are the tie events.
  2. Carrying the literal k = 3 to the four survivors, does the published
     verdict change, and in which direction.

Summed over annotators these give the counts Section 5 and Appendix F report,
8,433 tie events out of 49,070 removals, 3,843 one-way flips from offensive to
not offensive and none the other way, and 2,811 distinct items that can reach a
tie, which is the corpus's whole three-two stratum.

THE CORPUS IS NOT REDISTRIBUTED
MD-Agreement is distributed by its own authors. Point ``SG_MDAGREEMENT`` at a
locally obtained copy, either the released zip or the extracted
``MD-Agreement_dataset.tsv``. With the variable unset this stage prints why it
is skipping and exits 0, so the reproduction runs end to end without it and the
verification marks the affected rows SKIPPED rather than FAILED.

    SG_MDAGREEMENT=/path/to/MD-Agreement_datasets.zip python3 repro/run_panel.py

Read-only on all inputs. No randomness, so no seed.
"""
from __future__ import annotations

import csv
import io
import json
import os
import sys
import zipfile
from pathlib import Path

OUT = Path(os.environ.get("SG_RESULTS", Path(__file__).resolve().parent.parent / "results"))

#: The published label is a majority of five, so the literal threshold is 3.
K_LITERAL = 3
PANEL = 5
TSV_NAME = "MD-Agreement_dataset.tsv"


def _open_corpus(path: Path) -> io.TextIOBase:
    """Return a text handle on the corpus TSV, from a zip or a plain file."""
    if path.is_dir():
        candidates = sorted(path.rglob(TSV_NAME))
        if not candidates:
            raise SystemExit(f"no {TSV_NAME} under {path}")
        return candidates[0].open(newline="", encoding="utf-8")
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        names = [n for n in zf.namelist() if n.endswith(TSV_NAME)]
        if not names:
            raise SystemExit(f"no {TSV_NAME} inside {path}")
        return io.TextIOWrapper(zf.open(names[0]), encoding="utf-8", newline="")
    return path.open(newline="", encoding="utf-8")


def load_items(handle: io.TextIOBase) -> list[dict]:
    items = []
    for row in csv.DictReader(handle, delimiter="\t"):
        labels = [int(x.strip()) for x in row["Individual_Annotations"].split(",")]
        annotators = [x.strip() for x in row["Annotators_ID"].split(",")]
        if len(labels) != PANEL or len(annotators) != PANEL:
            raise SystemExit(
                f"item {row['ID']} does not carry exactly {PANEL} labels and annotators"
            )
        published = int(row["Offensive_binary_label"])
        if int(sum(labels) >= K_LITERAL) != published:
            raise SystemExit(
                f"item {row['ID']}: published label is not the five-annotator majority"
            )
        items.append({
            "id": row["ID"],
            "labels": labels,
            "annotators": annotators,
            "published": published,
            "agreement": row.get("Agreement_level", ""),
        })
    if not items:
        raise SystemExit("corpus is empty")
    return items


def analyse(items: list[dict]) -> tuple[list[dict], dict]:
    per_annotator: dict[str, dict] = {}
    tie_items: set[str] = set()
    tie_events = flips_1_to_0 = flips_0_to_1 = 0

    for item in items:
        for pos, annotator in enumerate(item["annotators"]):
            counts = per_annotator.setdefault(
                annotator,
                {"annotator": annotator, "items_labelled": 0,
                 "tie_events_created": 0, "pass_to_fail_flips": 0},
            )
            counts["items_labelled"] += 1
            survivors = item["labels"][:pos] + item["labels"][pos + 1:]
            s = sum(survivors)
            if s == PANEL - K_LITERAL:          # a two-two split among four
                counts["tie_events_created"] += 1
                tie_events += 1
                tie_items.add(item["id"])
            literal = int(s >= K_LITERAL)
            if literal != item["published"]:
                if item["published"] == 1:
                    counts["pass_to_fail_flips"] += 1
                    flips_1_to_0 += 1
                else:
                    flips_0_to_1 += 1

    rows = [per_annotator[a] for a in sorted(per_annotator)]
    removals = len(items) * PANEL
    summary = {
        "corpus_items": len(items),
        "annotators": len(rows),
        "single_annotator_removal_events": removals,
        "tie_events": tie_events,
        "tie_event_rate": round(tie_events / removals, 4),
        "unique_items_that_can_tie": len(tie_items),
        "exposed_share_of_corpus": round(len(tie_items) / len(items), 4),
        "literal_k_flips_offensive_to_not": flips_1_to_0,
        "literal_k_flips_not_to_offensive": flips_0_to_1,
    }
    return rows, summary


def main() -> int:
    corpus = os.environ.get("SG_MDAGREEMENT")
    if not corpus:
        print("Stage 5 skipped. MD-Agreement is distributed by its own authors and is")
        print("not redistributed here. Point SG_MDAGREEMENT at a local copy of the")
        print("released zip, the extracted directory, or MD-Agreement_dataset.tsv to")
        print("enable this stage. The verification marks the affected rows SKIPPED.")
        return 0

    path = Path(corpus).expanduser()
    if not path.exists():
        print(f"Stage 5 skipped. SG_MDAGREEMENT points at {path}, which does not exist.")
        return 0

    with _open_corpus(path) as handle:
        items = load_items(handle)
    rows, summary = analyse(items)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "mdagreement_removal_counts.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "mdagreement_summary.json").open("w") as f:
        json.dump(summary, f, indent=1)

    print(f"{'corpus items':<42}{summary['corpus_items']:>8,}")
    print(f"{'annotators':<42}{summary['annotators']:>8,}")
    print(f"{'single-annotator removal events':<42}{summary['single_annotator_removal_events']:>8,}")
    print(f"{'two-two tie events':<42}{summary['tie_events']:>8,}"
          f"   {summary['tie_event_rate']:.1%} of removals")
    print(f"{'distinct items that can tie':<42}{summary['unique_items_that_can_tie']:>8,}"
          f"   {summary['exposed_share_of_corpus']:.1%} of the corpus")
    print(f"{'literal k = 3 flips, offensive to not':<42}{summary['literal_k_flips_offensive_to_not']:>8,}")
    print(f"{'literal k = 3 flips, not to offensive':<42}{summary['literal_k_flips_not_to_offensive']:>8,}")
    print()
    print(f"Wrote {OUT / 'mdagreement_removal_counts.csv'} ({len(rows)} annotators)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
