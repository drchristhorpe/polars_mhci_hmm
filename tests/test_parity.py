"""Differential test against histo_hmm. This is the test the whole project rests on.

It asserts not only that we return the same top locus, but that every field of the struct matches
the reference's `ClassificationResult` -- the region, the ordering of all ten loci, the
confidence. A plugin that agreed on the winner and disagreed on the ranking would be just as
wrong as one that mis-scored.

Booleans, regions and locus order are compared exactly; floats get the tolerance justified in
PLAN.md §5.1. Comparison tables land in `tmp/` so a failure can be read.

Skips entirely if histo_hmm is not installed.
"""

from __future__ import annotations

import polars as pl
import pytest

from conftest import PROB_TOL, SCORE_TOL, assert_same, ours, theirs


def _table(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def test_parity_full_length(reference, class_i, artefacts):
    """Real Class I sequences, scored whole (the common case)."""
    seqs = [s["sequence"] for s in class_i]
    mine = ours(seqs, n_top=10)
    ref = theirs(reference, seqs, n_top=10)

    rows = []
    for s, m, r in zip(class_i, mine, ref):
        rows.append(
            {
                "name": s["name"],
                "expected_locus": s["locus"],
                "ours_top": m["top_loci"][0]["locus"],
                "ref_top": r["top_loci"][0]["locus"],
                "ours_best_score": m["best_score"],
                "ref_best_score": r["best_score"],
                "score_diff": abs(m["best_score"] - r["best_score"]),
                "prob_diff": max(
                    abs(a["probability"] - b["probability"])
                    for a, b in zip(m["top_loci"], r["top_loci"])
                ),
            }
        )
    artefacts("01_parity_full_length.csv", _table(rows))

    for s, m, r in zip(class_i, mine, ref):
        assert_same(m, r, s["name"])


def test_parity_negatives(reference, negatives, artefacts):
    """Non-MHC sequences: the reference and the plugin must agree on rejecting them too."""
    seqs = [s["sequence"] for s in negatives]
    mine = ours(seqs, n_top=5)
    ref = theirs(reference, seqs, n_top=5)

    artefacts(
        "02_parity_negatives.csv",
        _table(
            [
                {
                    "name": s["name"],
                    "ours_is_class_i": m["is_class_i"],
                    "ref_is_class_i": r["is_class_i"],
                    "ours_best_score": m["best_score"],
                    "ref_best_score": r["best_score"],
                }
                for s, m, r in zip(negatives, mine, ref)
            ]
        ),
    )

    for s, m, r in zip(negatives, mine, ref):
        assert_same(m, r, s["name"])


def test_parity_truncated(reference, class_i):
    """Partial sequences -- the reference explicitly supports these."""
    seq = class_i[0]["sequence"]
    variants = [seq[:100], seq[:180], seq[50:250], seq[-120:]]

    mine = ours(variants, n_top=5)
    ref = theirs(reference, variants, n_top=5)
    for i, (m, r) in enumerate(zip(mine, ref)):
        assert_same(m, r, f"truncated[{i}] (len {len(variants[i])})")


@pytest.mark.slow
def test_parity_construct_scan(reference, class_i, artefacts):
    """The sliding-window scan path.

    Slow -- the reference takes ~20 s per sequence here, because it scores ~104 windows against
    all 251 loci. That cost is exactly why this plugin exists, so it is worth paying once.
    """
    seq = class_i[0]["sequence"]
    constructs = [
        "M" * 60 + seq + "GGGSGGGS" * 10,  # MHC embedded in a fusion construct
        "MGSSHHHHHHSSGLVPRGSH" + seq,  # His-tagged
    ]

    mine = ours(constructs, n_top=5)
    ref = theirs(reference, constructs, n_top=5)

    artefacts(
        "03_parity_constructs.csv",
        _table(
            [
                {
                    "length": len(c),
                    "ours_region": f"{m['region_start']}:{m['region_end']}",
                    "ref_region": f"{r['region_start']}:{r['region_end']}",
                    "ours_top": m["top_loci"][0]["locus"],
                    "ref_top": r["top_loci"][0]["locus"],
                    "score_diff": abs(m["best_score"] - r["best_score"]),
                }
                for c, m, r in zip(constructs, mine, ref)
            ]
        ),
    )

    for i, (m, r) in enumerate(zip(mine, ref)):
        assert_same(m, r, f"construct[{i}]")


def test_parity_scan_disabled(reference, class_i):
    """`scan_constructs=False` must score the full sequence, however long it is."""
    seq = "M" * 60 + class_i[0]["sequence"] + "GGGSGGGS" * 10

    mine = ours([seq], n_top=5, scan_constructs=False)[0]
    ref = theirs(reference, [seq], n_top=5, scan_constructs=False)[0]

    assert_same(mine, ref, "scan disabled")
    assert mine["region_start"] == 0 and mine["region_end"] == len(seq)


def test_parity_messy_input(reference, class_i):
    """Case and punctuation are cleaned by both sides identically, so results must agree."""
    seq = class_i[0]["sequence"]
    variants = [
        seq.lower(),
        seq[:150] + "X" * 10 + seq[150:],  # unknown residues, kept and scored uniformly
        "-" * 5 + seq + "***",  # dropped entirely
        seq[:100] + "BZJ" + seq[100:],  # ambiguity codes: dropped, shortening the sequence
    ]

    mine = ours(variants, n_top=5)
    ref = theirs(reference, variants, n_top=5)
    for i, (m, r) in enumerate(zip(mine, ref)):
        assert_same(m, r, f"messy[{i}]")


def test_parity_non_ascii(reference, class_i):
    """Non-ASCII input must clean the way Python's str.upper() cleans it.

    'ß' uppercases to "SS" and 'ı' to "I" -- letters that survive cleaning and get scored. A
    bytewise ASCII uppercase would drop them and score a shorter sequence, which is a silently
    different answer rather than a loud failure.
    """
    seq = class_i[0]["sequence"]
    variants = ["MAKß", "MAKı", "MAKΩ", seq[:100] + "ß" + seq[100:]]

    mine = ours(variants, n_top=3)
    ref = theirs(reference, variants, n_top=3)
    for i, (m, r) in enumerate(zip(mine, ref)):
        assert_same(m, r, f"non-ascii[{i}] ({variants[i][:8]!r})")


def test_parity_empty_and_unscorable(reference):
    """Empty and all-invalid inputs hit the reference's degenerate branch."""
    variants = ["", "---", "***", "123"]

    mine = ours(variants)
    ref = theirs(reference, variants)
    for i, (m, r) in enumerate(zip(mine, ref)):
        assert_same(m, r, f"unscorable[{i}]")


@pytest.mark.parametrize("n_top", [1, 3, 10, 25])
def test_parity_n_top(reference, class_i, n_top):
    seqs = [s["sequence"] for s in class_i[:4]]
    mine = ours(seqs, n_top=n_top)
    ref = theirs(reference, seqs, n_top=n_top)
    for m, r in zip(mine, ref):
        assert len(m["top_loci"]) == n_top
        assert_same(m, r, f"n_top={n_top}")


@pytest.mark.parametrize("threshold", [-1e9, 0.0, 500.0, 1e9])
def test_parity_threshold(reference, class_i, negatives, threshold):
    """`threshold` shifts the is_class_i call; both sides must shift together."""
    seqs = [s["sequence"] for s in class_i[:3]] + [s["sequence"] for s in negatives]

    mine = ours(seqs, n_top=3, threshold=threshold)
    # histo_hmm takes the threshold on the constructor, not on classify().
    import histo_hmm
    import polars_mhci_hmm

    clf = histo_hmm.MHCClassIClassifier(
        model_dir=polars_mhci_hmm.model_dir(), threshold=threshold
    )
    ref = theirs(clf, seqs, n_top=3)

    for i, (m, r) in enumerate(zip(mine, ref)):
        assert_same(m, r, f"threshold={threshold} seq[{i}]")


def test_parity_single_model_score(reference, class_i):
    """`.mhci.score(locus)` mirrors ProfileHMM.log_odds_score."""
    seqs = [s["sequence"] for s in class_i[:5]]

    for locus in ["hla_a", "hla_b", "h2_k", "sasa_uba"]:
        mine = (
            pl.DataFrame({"s": seqs})
            .select(pl.col("s").mhci.score(locus))
            .to_series()
            .to_list()
        )
        model = reference.models[locus]
        for seq, got in zip(seqs, mine):
            cleaned = reference._clean_sequence(seq)
            want = model.log_odds_score(cleaned)
            assert got == pytest.approx(want, abs=SCORE_TOL), f"score({locus})"


def test_probabilities_sum_to_one_over_all_loci(class_i):
    """The softmax normalises across every locus, so asking for all of them sums to 1.0."""
    import polars_mhci_hmm

    n = len(polars_mhci_hmm.loci())
    result = ours([class_i[0]["sequence"]], n_top=n)[0]
    total = sum(d["probability"] for d in result["top_loci"])
    assert total == pytest.approx(1.0, abs=1e-12)


def test_parity_is_deterministic(class_i):
    """Parallel reduction must not make the scan's tie-breaking depend on scheduling."""
    seq = "M" * 60 + class_i[0]["sequence"] + "GGGS" * 20
    runs = [ours([seq], n_top=5, scan_constructs=False)[0] for _ in range(3)]
    for r in runs[1:]:
        assert r == runs[0]
