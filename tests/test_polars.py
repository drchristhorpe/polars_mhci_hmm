"""Polars integration: the things that make this a plugin rather than a function.

These run without histo_hmm installed.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_mhci_hmm

STRUCT_FIELDS = [
    "is_class_i",
    "confidence",
    "best_score",
    "region_start",
    "region_end",
    "top_loci",
]


@pytest.fixture(scope="module")
def seq(class_i) -> str:
    return class_i[0]["sequence"]


def test_schema_matches_declared_output(seq):
    df = pl.DataFrame({"s": [seq]}).select(pl.col("s").mhci.classify(n_top=3))
    dtype = df.schema["s"]

    assert isinstance(dtype, pl.Struct)
    assert [f.name for f in dtype.fields] == STRUCT_FIELDS

    by_name = {f.name: f.dtype for f in dtype.fields}
    assert by_name["is_class_i"] == pl.Boolean
    assert by_name["confidence"] == pl.Float64
    assert by_name["best_score"] == pl.Float64
    assert by_name["region_start"] == pl.UInt32
    assert by_name["region_end"] == pl.UInt32
    assert by_name["top_loci"] == pl.List(
        pl.Struct({"locus": pl.String, "probability": pl.Float64})
    )


def test_nulls_produce_null_structs(seq):
    """A null sequence is the absence of a classification, not a classification of nothing."""
    df = pl.DataFrame({"s": [seq, None, seq]}).select(
        r=pl.col("s").mhci.classify(n_top=1)
    )
    values = df["r"].to_list()

    assert values[1] is None
    assert values[0] is not None and values[2] is not None
    assert df["r"].null_count() == 1


def test_all_null_column(seq):
    """An all-null column still has to carry the declared inner dtype."""
    df = pl.DataFrame({"s": [None, None]}, schema={"s": pl.String}).select(
        r=pl.col("s").mhci.classify()
    )
    assert df["r"].to_list() == [None, None]
    assert df.schema["r"] == pl.Struct(
        {
            "is_class_i": pl.Boolean,
            "confidence": pl.Float64,
            "best_score": pl.Float64,
            "region_start": pl.UInt32,
            "region_end": pl.UInt32,
            "top_loci": pl.List(pl.Struct({"locus": pl.String, "probability": pl.Float64})),
        }
    )


def test_empty_frame():
    df = pl.DataFrame({"s": []}, schema={"s": pl.String}).select(
        r=pl.col("s").mhci.classify()
    )
    assert len(df) == 0
    assert isinstance(df.schema["r"], pl.Struct)


def test_lazy_frame(seq):
    got = (
        pl.LazyFrame({"s": [seq]})
        .select(pl.col("s").mhci.classify(n_top=1).struct.field("is_class_i"))
        .collect()
    )
    assert got.to_series().to_list() == [True]


def test_struct_field_unnest(seq):
    df = pl.DataFrame({"s": [seq]}).with_columns(
        r=pl.col("s").mhci.classify(n_top=1)
    ).unnest("r")
    assert set(STRUCT_FIELDS).issubset(df.columns)
    assert df["is_class_i"][0] is True


def test_top_loci_explodes(seq):
    df = (
        pl.DataFrame({"s": [seq]})
        .select(pl.col("s").mhci.classify(n_top=5).struct.field("top_loci"))
        .explode("top_loci")
        .unnest("top_loci")
    )
    assert len(df) == 5
    assert df.columns == ["locus", "probability"]
    # Ranked best-first.
    probs = df["probability"].to_list()
    assert probs == sorted(probs, reverse=True)


def test_multiple_chunks(seq, class_i):
    """A chunked column must classify the same as a contiguous one, in the same order."""
    a = pl.DataFrame({"s": [seq]})
    b = pl.DataFrame({"s": [class_i[1]["sequence"]]})
    chunked = pl.concat([a, b], rechunk=False)
    contiguous = pl.concat([a, b], rechunk=True)

    assert chunked.n_chunks() > 1
    got = chunked.select(r=pl.col("s").mhci.classify(n_top=3))["r"].to_list()
    want = contiguous.select(r=pl.col("s").mhci.classify(n_top=3))["r"].to_list()
    assert got == want


def test_row_order_is_preserved(class_i):
    """Parallelism inside the kernel must not reorder rows."""
    seqs = [s["sequence"] for s in class_i[:8]]
    df = pl.DataFrame({"name": [s["name"] for s in class_i[:8]], "s": seqs}).with_columns(
        top=pl.col("s").mhci.classify(n_top=1).struct.field("top_loci")
    )
    tops = [row[0]["locus"] for row in df["top"].to_list()]
    expected = [s["locus"] for s in class_i[:8]]
    assert tops == expected


def test_group_by(class_i):
    seqs = [s["sequence"] for s in class_i[:6]]
    df = pl.DataFrame({"g": ["a", "b"] * 3, "s": seqs}).group_by("g").agg(
        pl.col("s").mhci.classify(n_top=1).struct.field("best_score")
    )
    assert len(df) == 2


def test_score_expression(seq):
    df = pl.DataFrame({"s": [seq, None]}).select(
        hla_a=pl.col("s").mhci.score("hla_a"),
        h2_k=pl.col("s").mhci.score("h2_k"),
    )
    assert df.schema["hla_a"] == pl.Float64
    assert df["hla_a"][1] is None
    # An HLA-A sequence should look more like hla_a than like a mouse locus.
    assert df["hla_a"][0] > df["h2_k"][0]


def test_function_forms(seq):
    """The module-level forms are sugar over the namespace, and must agree with it."""
    df = pl.DataFrame({"s": [seq]})
    a = df.select(polars_mhci_hmm.classify("s", n_top=2))["s"][0]
    b = df.select(pl.col("s").mhci.classify(n_top=2))["s"][0]
    assert a == b

    c = df.select(polars_mhci_hmm.score("s", locus="hla_a"))["s"][0]
    d = df.select(pl.col("s").mhci.score("hla_a"))["s"][0]
    assert c == d


def test_loci_frame():
    df = polars_mhci_hmm.loci()
    assert df.columns == ["locus", "length"]
    assert len(df) == 251
    assert df["locus"].to_list() == sorted(df["locus"].to_list())
    assert "hla_a" in df["locus"].to_list()
    # PLAN.md §5.9: lengths vary; nothing may assume 275.
    assert df["length"].min() == 273
    assert df["length"].max() == 282


def test_model_dir_is_usable(seq):
    """`model_dir=` accepts the bundled directory explicitly, and matches the default."""
    df = pl.DataFrame({"s": [seq]})
    a = df.select(pl.col("s").mhci.classify(n_top=2))["s"][0]
    b = df.select(
        pl.col("s").mhci.classify(n_top=2, model_dir=polars_mhci_hmm.model_dir())
    )["s"][0]
    assert a == b


class TestValidation:
    """Bad arguments fail at expression-construction time, not inside the query engine."""

    def test_unknown_locus_suggests_alternatives(self):
        with pytest.raises(ValueError, match="unknown locus"):
            pl.col("s").mhci.score("hla_z")

    def test_unknown_locus_near_miss_hint(self):
        with pytest.raises(ValueError, match="did you mean"):
            pl.col("s").mhci.score("hla_A")

    def test_locus_must_be_str(self):
        with pytest.raises(TypeError, match="locus must be a str"):
            pl.col("s").mhci.score(1)

    def test_negative_n_top(self):
        with pytest.raises(ValueError, match="n_top must be >= 0"):
            pl.col("s").mhci.classify(n_top=-1)

    def test_n_top_must_be_int(self):
        with pytest.raises(TypeError, match="n_top must be an int"):
            pl.col("s").mhci.classify(n_top=2.5)

    def test_scan_constructs_must_be_bool(self):
        with pytest.raises(TypeError, match="scan_constructs must be a bool"):
            pl.col("s").mhci.classify(scan_constructs="yes")

    def test_threshold_must_be_number(self):
        with pytest.raises(TypeError, match="threshold must be a number"):
            pl.col("s").mhci.classify(threshold="high")

    def test_missing_model_dir(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            pl.col("s").mhci.classify(model_dir=tmp_path / "nope")

    def test_model_dir_without_manifest(self, tmp_path):
        with pytest.raises(ValueError, match="no manifest.json"):
            pl.col("s").mhci.classify(model_dir=tmp_path)
