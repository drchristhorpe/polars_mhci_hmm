"""Shared fixtures and helpers.

The suite has two halves:

* **Parity** -- runs the real ``histo_hmm`` alongside the plugin and compares. This is the test
  the project rests on. It skips if ``histo_hmm`` is not installed (``uv sync`` installs it from
  git as a dev dependency).
* **Golden / integration** -- checked-in expectations and Polars behaviour, which run anywhere.

Every test that produces something worth looking at writes it to ``tmp/`` via the ``artefacts``
fixture, so a failure can be read rather than re-derived.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

import polars_mhci_hmm  # noqa: F401  -- registers the .mhci namespace

ROOT = Path(__file__).resolve().parent.parent
# Test artefacts live under tmp/tests/ so they do not collide with the numbered ones
# tools/validate.py writes to tmp/ itself.
TMP = ROOT / "tmp" / "tests"
DATA = Path(__file__).parent / "data" / "mhci_sequences.json"

# PLAN.md §5.1: Viterbi is max-plus and reproduces exactly, but histo_hmm's null score and
# softmax denominator go through numpy's pairwise `np.sum`, and its `exp` may differ from Rust's
# libm by an ULP. Observed drift is ~1e-12 on log-odds of ~700; these leave four orders of room.
SCORE_TOL = 1e-9
PROB_TOL = 1e-12


def pytest_configure(config):
    TMP.mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def artefacts():
    """A place under ``tmp/`` to write test output for inspection."""
    TMP.mkdir(parents=True, exist_ok=True)

    def _write(name: str, df: pl.DataFrame) -> Path:
        path = TMP / name
        if path.suffix == ".parquet":
            df.write_parquet(path)
        else:
            df.write_csv(path)
        return path

    return _write


@pytest.fixture(scope="session")
def sequences() -> list[dict]:
    """The curated sample: real Class I sequences plus non-MHC negatives."""
    return json.loads(DATA.read_text())["sequences"]


@pytest.fixture(scope="session")
def class_i(sequences) -> list[dict]:
    return [s for s in sequences if s["kind"] == "class_i"]


@pytest.fixture(scope="session")
def negatives(sequences) -> list[dict]:
    return [s for s in sequences if s["kind"] == "negative"]


@pytest.fixture(scope="session")
def tied_models(tmp_path_factory) -> Path:
    """A model directory whose loci score *identically*, to test tie-breaking.

    Real loci never tie, so the rule that ties resolve to manifest order (PLAN.md §5.2) is
    untestable on real data. Three byte-identical copies of hla_a, named to sort in a known
    order, make the tie exact and the correct answer unambiguous.
    """
    import shutil

    src = Path(polars_mhci_hmm.model_dir())
    dst = tmp_path_factory.mktemp("tied_models")

    names = ["aaa_copy", "mmm_copy", "zzz_copy"]
    for name in names:
        shutil.copy2(src / "hla_a.npz", dst / f"{name}.npz")
    shutil.copy2(src / "background.npy", dst / "background.npy")

    # Manifest order is the tie-break order, so write it sorted -- as histo_hmm's save_models
    # does -- rather than in whatever order the loop happened to run.
    (dst / "manifest.json").write_text(
        json.dumps(
            {
                "n_models": len(names),
                "classes": sorted(names),
                "lengths": dict.fromkeys(names, 275),
            },
            indent=2,
        )
    )
    return dst


@pytest.fixture(scope="session")
def reference():
    """A live ``histo_hmm`` classifier, or skip.

    Built against *our* vendored models rather than histo_hmm's bundled copy, so a parity failure
    means the port is wrong rather than that the two models directories drifted.
    """
    histo = pytest.importorskip("histo_hmm", reason="histo_hmm not installed; skipping parity")
    return histo.MHCClassIClassifier(model_dir=polars_mhci_hmm.model_dir())


def ours(seqs, **kwargs) -> list[dict | None]:
    """Classify via the plugin."""
    return (
        pl.DataFrame({"s": seqs}, schema={"s": pl.String})
        .select(pl.col("s").mhci.classify(**kwargs))
        .to_series()
        .to_list()
    )


def theirs(clf, seqs, **kwargs) -> list[dict]:
    """Classify via histo_hmm, row by row, shaped like our struct for comparison."""
    out = []
    for s in seqs:
        r = clf.classify(s, **kwargs)
        out.append(
            {
                "is_class_i": r.is_class_i,
                "confidence": r.confidence,
                "best_score": r.best_score,
                "region_start": r.region_start,
                "region_end": r.region_end,
                "top_loci": [{"locus": k, "probability": p} for k, p in r.top_loci],
            }
        )
    return out


def assert_same(mine: dict, ref: dict, label: str) -> None:
    """Assert one classification matches the reference.

    Booleans, regions and locus *order* must be exact; only the floats get a tolerance.
    """
    assert mine["is_class_i"] == ref["is_class_i"], f"{label}: is_class_i"
    assert mine["region_start"] == ref["region_start"], f"{label}: region_start"
    assert mine["region_end"] == ref["region_end"], f"{label}: region_end"

    assert [d["locus"] for d in mine["top_loci"]] == [
        d["locus"] for d in ref["top_loci"]
    ], f"{label}: top_loci order"

    assert mine["best_score"] == pytest.approx(ref["best_score"], abs=SCORE_TOL), (
        f"{label}: best_score"
    )
    assert mine["confidence"] == pytest.approx(ref["confidence"], abs=PROB_TOL), (
        f"{label}: confidence"
    )
    for a, b in zip(mine["top_loci"], ref["top_loci"]):
        assert a["probability"] == pytest.approx(b["probability"], abs=PROB_TOL), (
            f"{label}: probability for {b['locus']}"
        )
