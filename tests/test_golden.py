"""Checked-in expectations, generated from histo_hmm (see `tools/make_golden.py`).

Unlike `test_parity.py`, these need nothing but the plugin, so the promise "we reproduce
histo_hmm" stays testable on a machine that has never had histo_hmm installed -- and stays
pinned to the reference's behaviour at a known commit.

Writes a full comparison to `tmp/04_golden.csv`.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import polars_mhci_hmm  # noqa: F401

from conftest import PROB_TOL, SCORE_TOL

GOLDEN = Path(__file__).parent / "data" / "golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN.exists():
        pytest.skip(f"{GOLDEN.name} not generated; run tools/make_golden.py")
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module")
def results(golden, sequences) -> list[tuple[dict, dict]]:
    """Pair each golden case with our classification of the same sequence."""
    by_name = {s["name"]: s["sequence"] for s in sequences}
    cases = golden["cases"]
    n_top = golden["generated_from"]["n_top"]

    seqs = [by_name[c["name"]] for c in cases]
    mine = (
        pl.DataFrame({"s": seqs}, schema={"s": pl.String})
        .select(pl.col("s").mhci.classify(n_top=n_top))
        .to_series()
        .to_list()
    )
    return list(zip(cases, mine))


def test_golden_was_generated_from_our_models(golden):
    """A golden file describing someone else's models would silently weaken every assertion."""
    manifest = json.loads((Path(polars_mhci_hmm.model_dir()) / "manifest.json").read_text())
    assert golden["generated_from"]["models_commit"] == manifest["source"]["commit"], (
        "golden.json was generated against different models; re-run tools/make_golden.py"
    )


def test_golden_classifications(results, artefacts):
    rows = []
    for want, got in results:
        rows.append(
            {
                "name": want["name"],
                "kind": want["kind"],
                "expected_locus": want["expected_locus"],
                "golden_top": want["top_loci"][0]["locus"] if want["top_loci"] else None,
                "ours_top": got["top_loci"][0]["locus"] if got["top_loci"] else None,
                "golden_best_score": want["best_score"],
                "ours_best_score": got["best_score"],
                "score_diff": abs(want["best_score"] - got["best_score"])
                if want["best_score"] != float("-inf")
                else 0.0,
                "golden_is_class_i": want["is_class_i"],
                "ours_is_class_i": got["is_class_i"],
            }
        )
    artefacts("04_golden.csv", pl.DataFrame(rows, infer_schema_length=None))

    for want, got in results:
        label = want["name"]
        assert got["is_class_i"] == want["is_class_i"], f"{label}: is_class_i"
        assert got["region_start"] == want["region_start"], f"{label}: region_start"
        assert got["region_end"] == want["region_end"], f"{label}: region_end"
        assert [d["locus"] for d in got["top_loci"]] == [
            d["locus"] for d in want["top_loci"]
        ], f"{label}: top_loci order"
        assert got["best_score"] == pytest.approx(want["best_score"], abs=SCORE_TOL), (
            f"{label}: best_score"
        )
        assert got["confidence"] == pytest.approx(want["confidence"], abs=PROB_TOL), (
            f"{label}: confidence"
        )
        for a, b in zip(got["top_loci"], want["top_loci"]):
            assert a["probability"] == pytest.approx(b["probability"], abs=PROB_TOL), (
                f"{label}: probability for {b['locus']}"
            )


def test_golden_covers_both_kinds(golden):
    kinds = {c["kind"] for c in golden["cases"]}
    assert kinds == {"class_i", "negative"}


def test_readme_example(class_i):
    """The example in the README must actually produce what the README says it does."""
    seq = next(s["sequence"] for s in class_i if s["locus"] == "hla_a")
    result = (
        pl.DataFrame({"sequence": [seq]})
        .select(pl.col("sequence").mhci.classify(n_top=3))
        .to_series()
        .to_list()[0]
    )
    assert result["is_class_i"] is True
    assert result["top_loci"][0]["locus"] == "hla_a"
    assert result["confidence"] == pytest.approx(1.0, abs=1e-6)
