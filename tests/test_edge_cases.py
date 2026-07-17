"""The faithfulness contract (PLAN.md §5), one test per rule.

These pin the behaviours a casual port gets wrong. They are written against the plugin alone --
no histo_hmm required -- because each encodes a rule read out of the reference's source, so they
keep their meaning even where the reference is not installed to compare against.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import polars_mhci_hmm  # noqa: F401


def classify_one(seq: str, **kwargs) -> dict:
    return (
        pl.DataFrame({"s": [seq]}, schema={"s": pl.String})
        .select(pl.col("s").mhci.classify(**kwargs))
        .to_series()
        .to_list()[0]
    )


def score_one(seq: str, locus: str = "hla_a") -> float:
    return (
        pl.DataFrame({"s": [seq]}, schema={"s": pl.String})
        .select(pl.col("s").mhci.score(locus))
        .to_series()
        .to_list()[0]
    )


class TestRule1SequenceCleaning:
    """§5.1 -- uppercase, keep the 20 AAs and X, drop everything else."""

    def test_case_is_normalised(self, class_i):
        seq = class_i[0]["sequence"]
        assert classify_one(seq.lower()) == classify_one(seq)

    def test_gaps_and_punctuation_are_dropped_not_scored(self, class_i):
        seq = class_i[0]["sequence"]
        gapped = "-".join(seq)  # a gap between every residue
        assert classify_one(gapped) == classify_one(seq)

    def test_ambiguity_codes_are_dropped(self, class_i):
        """B, Z and J are not in the alphabet and are *removed*, shortening the sequence."""
        seq = class_i[0]["sequence"]
        got = classify_one(seq[:100] + "BZJ" + seq[100:])
        want = classify_one(seq)
        assert got == want

    def test_x_is_kept_and_scored(self, class_i):
        """X survives cleaning and scores uniformly -- unlike B/Z/J, which vanish."""
        seq = class_i[0]["sequence"]
        with_x = seq[:100] + "X" * 5 + seq[105:]

        # X is kept, so the length is unchanged...
        assert classify_one(with_x)["region_end"] == len(seq)
        # ...but the score changes, because X emits uniformly rather than as the real residue.
        assert score_one(with_x) != score_one(seq)

    def test_dropped_characters_shorten_the_region(self, class_i):
        seq = class_i[0]["sequence"]
        assert classify_one(seq + "BZJ")["region_end"] == len(seq)

    def test_region_indexes_the_cleaned_sequence(self, class_i):
        """The offsets refer to the cleaned string, not the caller's input."""
        seq = class_i[0]["sequence"]
        result = classify_one("---" + seq)
        assert result["region_start"] == 0
        assert result["region_end"] == len(seq)  # not len(seq) + 3

    @pytest.mark.parametrize(
        ("probe", "expected_len"),
        [
            ("MAKß", 5),  # Python's str.upper(): 'ß' -> 'SS', so two extra residues survive
            ("MAKı", 4),  # dotless i -> 'I', which is an amino acid
            ("MAKΩ", 3),  # uppercases to itself, still not an amino acid: dropped
        ],
    )
    def test_unicode_uppercasing_matches_python(self, probe, expected_len):
        """Cleaning uppercases the Unicode way, because the reference's str.upper() does.

        A bytewise ASCII uppercase would drop these characters instead of expanding them, and
        silently score a shorter sequence than histo_hmm does.
        """
        assert classify_one(probe)["region_end"] == expected_len


class TestRule2LocusOrdering:
    """§5.2 -- ties keep manifest order; ranking is by probability descending."""

    def test_top_loci_are_ranked(self, class_i):
        result = classify_one(class_i[0]["sequence"], n_top=20)
        probs = [d["probability"] for d in result["top_loci"]]
        assert probs == sorted(probs, reverse=True)

    def test_ties_follow_manifest_order(self, tied_models, class_i):
        """Identical models tie exactly, and a tie must resolve to manifest order.

        Real loci never tie, so the only honest way to test the rule is to build models that do:
        `tied_models` is three byte-identical copies of hla_a under names that sort in a known
        order. Python's `sorted(..., reverse=True)` is stable, so the reference returns them in
        manifest order; an unstable sort in the port would scramble them, and nothing in the
        real-data tests would notice.
        """
        result = classify_one(
            class_i[0]["sequence"], n_top=3, model_dir=tied_models
        )
        probs = [d["probability"] for d in result["top_loci"]]
        loci = [d["locus"] for d in result["top_loci"]]

        assert probs[0] == pytest.approx(probs[-1], abs=1e-15), "expected an exact tie"
        assert loci == ["aaa_copy", "mmm_copy", "zzz_copy"]

    def test_ordering_is_stable_across_runs(self, tied_models, class_i):
        """A tie must break the same way every time, not by whichever thread finished first."""
        runs = [
            classify_one(class_i[0]["sequence"], n_top=3, model_dir=tied_models)
            for _ in range(5)
        ]
        for r in runs[1:]:
            assert r == runs[0]


class TestRule5ScanTrigger:
    """§5.5 -- scan iff len > 370 and scan_constructs is on."""

    def test_at_or_below_threshold_scores_whole(self, class_i):
        seq = (class_i[0]["sequence"] * 2)[:370]
        result = classify_one(seq)
        assert (result["region_start"], result["region_end"]) == (0, 370)

    def test_above_threshold_scans(self, class_i):
        seq = (class_i[0]["sequence"] * 2)[:371]
        result = classify_one(seq)
        # A scanned region is a window of 200..320, never the whole 371.
        assert result["region_end"] - result["region_start"] <= 320
        assert (result["region_start"], result["region_end"]) != (0, 371)

    def test_scan_disabled_scores_whole(self, class_i):
        seq = (class_i[0]["sequence"] * 2)[:500]
        result = classify_one(seq, scan_constructs=False)
        assert (result["region_start"], result["region_end"]) == (0, 500)

    def test_window_geometry(self, class_i):
        """Windows start on a multiple of 20 and are a multiple of 10 long, within 200..320."""
        seq = "M" * 60 + class_i[0]["sequence"] + "GGGS" * 25
        result = classify_one(seq)
        start, end = result["region_start"], result["region_end"]
        assert start % 20 == 0
        assert (end - start) % 10 == 0
        assert 200 <= end - start <= 320


class TestDegenerateInputs:
    """The reference's empty branch, reproduced exactly."""

    @pytest.mark.parametrize("seq", ["", "---", "***", "123", "!@#"])
    def test_unscorable_returns_degenerate_result(self, seq):
        result = classify_one(seq)
        assert result["is_class_i"] is False
        assert result["confidence"] == 0.0
        assert result["top_loci"] == []
        assert result["best_score"] == float("-inf")
        assert (result["region_start"], result["region_end"]) == (0, 0)

    def test_single_residue(self):
        """One residue is scorable, so it takes the normal path rather than the empty one."""
        result = classify_one("M", n_top=1)
        assert (result["region_start"], result["region_end"]) == (0, 1)
        assert len(result["top_loci"]) == 1
        assert math.isfinite(result["best_score"])

    def test_n_top_zero_returns_no_loci(self, class_i):
        result = classify_one(class_i[0]["sequence"], n_top=0)
        assert result["top_loci"] == []
        # The classification itself still happened.
        assert result["is_class_i"] is True

    def test_n_top_beyond_locus_count_is_capped(self, class_i):
        result = classify_one(class_i[0]["sequence"], n_top=10_000)
        assert len(result["top_loci"]) == 251


class TestRule7And8ScoreTransforms:
    """§5.7/§5.8 -- softmax temperature 30, confidence sigmoid clamped to ±50."""

    def test_confidence_is_sigmoid_of_best_score(self, class_i):
        result = classify_one(class_i[0]["sequence"], n_top=1)
        best = result["best_score"]
        clamped = max(-50.0, min(50.0, best))
        assert result["confidence"] == pytest.approx(1.0 / (1.0 + math.exp(-clamped)))

    def test_confidence_saturates_not_overflows(self, class_i):
        """A strong hit clamps at +50, giving a confidence of 1.0 rather than an overflow."""
        result = classify_one(class_i[0]["sequence"], n_top=1)
        assert result["best_score"] > 50.0
        assert result["confidence"] == pytest.approx(1.0)

    def test_probabilities_sum_to_one(self, class_i):
        result = classify_one(class_i[0]["sequence"], n_top=251)
        assert sum(d["probability"] for d in result["top_loci"]) == pytest.approx(1.0, abs=1e-12)

    def test_softmax_temperature_is_thirty(self, class_i):
        """Recover the temperature from the ratio of two probabilities.

        p_i/p_j == exp((s_i - s_j)/T), so T == (s_i - s_j) / log(p_i/p_j). Checking this against
        raw log-odds from `score()` pins the constant without reaching into the kernel.
        """
        seq = class_i[0]["sequence"]
        result = classify_one(seq, n_top=251)
        top, second = result["top_loci"][0], result["top_loci"][1]

        s_i, s_j = score_one(seq, top["locus"]), score_one(seq, second["locus"])
        t = (s_i - s_j) / math.log(top["probability"] / second["probability"])
        assert t == pytest.approx(30.0, rel=1e-6)


class TestThreshold:
    """`threshold` decides is_class_i, and nothing else."""

    def test_threshold_flips_the_call(self, class_i):
        seq = class_i[0]["sequence"]
        assert classify_one(seq, threshold=0.0)["is_class_i"] is True
        assert classify_one(seq, threshold=1e9)["is_class_i"] is False

    def test_threshold_is_inclusive(self, class_i):
        """`best_score >= threshold`, so a threshold exactly at the score still counts."""
        seq = class_i[0]["sequence"]
        best = classify_one(seq)["best_score"]
        assert classify_one(seq, threshold=best)["is_class_i"] is True
        assert classify_one(seq, threshold=math.nextafter(best, math.inf))["is_class_i"] is False

    def test_threshold_does_not_change_scores(self, class_i):
        seq = class_i[0]["sequence"]
        a = classify_one(seq, threshold=0.0)
        b = classify_one(seq, threshold=1e9)
        assert a["best_score"] == b["best_score"]
        assert a["top_loci"] == b["top_loci"]


class TestNegatives:
    """Non-MHC sequences must be rejected, and rejected for the right reason."""

    def test_negatives_are_not_class_i(self, negatives):
        for s in negatives:
            result = classify_one(s["sequence"], n_top=1)
            assert result["is_class_i"] is False, f"{s['name']} was called Class I"
            assert result["best_score"] < 0.0

    def test_class_i_sequences_are_class_i(self, class_i):
        for s in class_i:
            result = classify_one(s["sequence"], n_top=1)
            assert result["is_class_i"] is True, f"{s['name']} was not called Class I"


# Loci with no near-sister in the model set. For these -- and only these -- "recovers the locus
# it was drawn from" is a fair expectation of the reference classifier, which makes it a useful
# check that the models are loaded, indexed and oriented correctly.
#
# It is deliberately NOT asserted for the rest, because histo_hmm does not do it: it prefers
# mafa_a1 for a rhesus mamu_a1 sequence (sister macaque species), hla_a for a chimp patr_a and
# hla_b for a gorilla gogo_b, bola_2 over bola_1, gaga_bf2 over gaga_bf1, and it ranks the mouse
# H2 loci as low as 100th. That is the reference's accuracy, not this port's: reproducing it
# faithfully -- confusions included -- is the contract. test_golden.py pins every one of those
# rankings exactly, so a regression in the port still fails loudly.
WELL_SEPARATED = {"hla_a", "hla_b", "hla_c", "hla_e", "sla_1", "sasa_uba"}


class TestSanity:
    """Cheap checks that the models are wired up right, independent of the reference."""

    def test_well_separated_loci_are_recovered(self, class_i):
        wrong = []
        for s in class_i:
            if s["locus"] not in WELL_SEPARATED:
                continue
            top = classify_one(s["sequence"], n_top=1)["top_loci"][0]["locus"]
            if top != s["locus"]:
                wrong.append((s["name"], s["locus"], top))
        assert not wrong, f"mis-assigned loci: {wrong}"

    def test_confusions_stay_within_class_i(self, class_i):
        """Even where the locus is wrong, the sequence is still recognised as Class I.

        This is what separates "the reference is imprecise about sister taxa" from "the port is
        broken": a mis-loaded model would not merely rank loci oddly, it would stop scoring
        Class I sequences as Class I at all.
        """
        for s in class_i:
            result = classify_one(s["sequence"], n_top=1)
            assert result["is_class_i"] is True
            assert result["best_score"] > 0.0, f"{s['name']} scored below the null model"
