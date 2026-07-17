"""Produce validation artefacts into `tmp/` so the histo_hmm-parity claim can be inspected.

Run:  uv run python tools/validate.py

Writes:
  tmp/00_SUMMARY.txt              read this one first
  tmp/01_parity_full_length.csv   every test sequence, ours vs histo_hmm
  tmp/02_parity_constructs.csv    the sliding-window scan path, ours vs histo_hmm
  tmp/03_locus_confusion.csv      what each sequence was assigned, and how confidently
  tmp/04_benchmark.txt            throughput vs histo_hmm
  tmp/05_classified.parquet       the full classified dataset

The parity sections need histo_hmm (`uv sync`); the benchmark and the classified dataset do not.
"""

from __future__ import annotations

import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import polars as pl

import polars_mhci_hmm

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)
DATA = ROOT / "tests" / "data" / "mhci_sequences.json"

summary = io.StringIO()


def say(msg: str = "") -> None:
    print(msg)
    summary.write(msg + "\n")


def load_sequences() -> list[dict]:
    return json.loads(DATA.read_text())["sequences"]


def ours(seqs: list[str], **kwargs) -> list[dict]:
    return (
        pl.DataFrame({"s": seqs}, schema={"s": pl.String})
        .select(pl.col("s").mhci.classify(**kwargs))
        .to_series()
        .to_list()
    )


def get_reference():
    try:
        import histo_hmm
    except ImportError:
        return None
    return histo_hmm.MHCClassIClassifier(model_dir=polars_mhci_hmm.model_dir())


def section_parity(records: list[dict], clf) -> None:
    say("=" * 78)
    say("PARITY vs histo_hmm -- full-length sequences")
    say("=" * 78)

    seqs = [r["sequence"] for r in records]
    mine = ours(seqs, n_top=10)

    rows = []
    worst_score, worst_prob = 0.0, 0.0
    for rec, m in zip(records, mine):
        ref = clf.classify(rec["sequence"], n_top=10)
        sd = abs(ref.best_score - m["best_score"])
        pd_ = max(
            (abs(p - d["probability"]) for (_, p), d in zip(ref.top_loci, m["top_loci"])),
            default=0.0,
        )
        worst_score = max(worst_score, sd)
        worst_prob = max(worst_prob, pd_)
        rows.append(
            {
                "name": rec["name"],
                "kind": rec["kind"],
                "expected_locus": rec["locus"],
                "ref_top": ref.top_loci[0][0] if ref.top_loci else None,
                "ours_top": m["top_loci"][0]["locus"] if m["top_loci"] else None,
                "order_match": [k for k, _ in ref.top_loci]
                == [d["locus"] for d in m["top_loci"]],
                "ref_best_score": ref.best_score,
                "ours_best_score": m["best_score"],
                "score_diff": sd,
                "prob_diff": pd_,
                "ref_is_class_i": ref.is_class_i,
                "ours_is_class_i": m["is_class_i"],
            }
        )

    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(TMP / "01_parity_full_length.csv")

    n = len(df)
    order_ok = int(df["order_match"].sum())
    call_ok = int((df["ref_is_class_i"] == df["ours_is_class_i"]).sum())
    say(f"  sequences               : {n}")
    say(f"  locus ordering matches  : {order_ok}/{n}")
    say(f"  is_class_i matches      : {call_ok}/{n}")
    say(f"  worst best_score diff   : {worst_score:.3e}   (tolerance 1e-9)")
    say(f"  worst probability diff  : {worst_prob:.3e}   (tolerance 1e-12)")
    say("")
    say("  Viterbi is max-plus, so model scores reproduce exactly. The residual drift is")
    say("  histo_hmm's np.sum being pairwise where ours is sequential, plus an ULP of exp.")
    say("  See PLAN.md section 5.1.")
    say("")


def section_constructs(records: list[dict], clf) -> None:
    say("=" * 78)
    say("PARITY vs histo_hmm -- sliding-window scan (constructs)")
    say("=" * 78)

    seq = next(r["sequence"] for r in records if r["locus"] == "hla_a")
    constructs = {
        "fusion_60aa_prefix": "M" * 60 + seq + "GGGSGGGS" * 10,
        "his_tagged": "MGSSHHHHHHSSGLVPRGSH" + seq,
    }

    rows = []
    for name, c in constructs.items():
        t = time.perf_counter()
        m = ours([c], n_top=5)[0]
        ours_s = time.perf_counter() - t

        t = time.perf_counter()
        ref = clf.classify(c, n_top=5)
        ref_s = time.perf_counter() - t

        rows.append(
            {
                "name": name,
                "length": len(c),
                "ref_region": f"{ref.region_start}:{ref.region_end}",
                "ours_region": f"{m['region_start']}:{m['region_end']}",
                "region_match": (ref.region_start, ref.region_end)
                == (m["region_start"], m["region_end"]),
                "ref_top": ref.top_loci[0][0],
                "ours_top": m["top_loci"][0]["locus"],
                "score_diff": abs(ref.best_score - m["best_score"]),
                "ref_seconds": round(ref_s, 3),
                "ours_seconds": round(ours_s, 3),
                "speedup": round(ref_s / ours_s, 1),
            }
        )
        say(f"  {name} (len {len(c)})")
        say(f"    region     ref {rows[-1]['ref_region']} | ours {rows[-1]['ours_region']}")
        say(f"    top locus  ref {rows[-1]['ref_top']} | ours {rows[-1]['ours_top']}")
        say(f"    time       ref {ref_s:.2f}s | ours {ours_s:.2f}s | {rows[-1]['speedup']}x")

    pl.DataFrame(rows, infer_schema_length=None).write_csv(TMP / "02_parity_constructs.csv")
    say("")


def section_assignments(records: list[dict]) -> None:
    say("=" * 78)
    say("LOCUS ASSIGNMENT")
    say("=" * 78)

    seqs = [r["sequence"] for r in records]
    mine = ours(seqs, n_top=3)

    rows = []
    for rec, m in zip(records, mine):
        top = m["top_loci"][0] if m["top_loci"] else {"locus": None, "probability": None}
        rows.append(
            {
                "name": rec["name"],
                "kind": rec["kind"],
                "expected_locus": rec["locus"],
                "assigned_locus": top["locus"],
                "correct": rec["locus"] == top["locus"] if rec["kind"] == "class_i" else None,
                "probability": top["probability"],
                "best_score": m["best_score"],
                "is_class_i": m["is_class_i"],
                "region": f"{m['region_start']}:{m['region_end']}",
            }
        )

    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(TMP / "03_locus_confusion.csv")
    df.write_parquet(TMP / "05_classified.parquet")

    pos = df.filter(pl.col("kind") == "class_i")
    neg = df.filter(pl.col("kind") == "negative")
    say(f"  Class I sequences assigned to the right locus : {int(pos['correct'].sum())}/{len(pos)}")
    say(f"  Class I sequences called Class I              : {int(pos['is_class_i'].sum())}/{len(pos)}")
    say(f"  negatives correctly rejected                  : {int((~neg['is_class_i']).sum())}/{len(neg)}")
    say("")


def section_benchmark(records: list[dict], clf) -> None:
    say("=" * 78)
    say("BENCHMARK")
    say("=" * 78)

    seqs = [r["sequence"] for r in records if r["kind"] == "class_i"]
    out = io.StringIO()

    def emit(msg: str) -> None:
        say(msg)
        out.write(msg + "\n")

    # sched_getaffinity would report the cores actually usable, but it is Linux-only and this
    # runs on macOS too.
    emit(f"  machine: {platform.processor() or platform.machine()}, "
         f"{os.cpu_count()} cores")
    emit(f"  polars {pl.__version__}, python {sys.version.split()[0]}")
    emit("")

    # Warm: model load on our side, JIT compile on theirs.
    ours(seqs[:1])
    if clf is not None:
        clf.classify(seqs[0])

    t = time.perf_counter()
    ours(seqs)
    ours_total = time.perf_counter() - t
    emit(f"  polars-mhci-hmm : {ours_total / len(seqs) * 1000:7.1f} ms/seq  "
         f"({len(seqs) / ours_total:6.1f} seq/s)  [{len(seqs)} sequences]")

    if clf is not None:
        t = time.perf_counter()
        for s in seqs:
            clf.classify(s)
        ref_total = time.perf_counter() - t
        emit(f"  histo_hmm       : {ref_total / len(seqs) * 1000:7.1f} ms/seq  "
             f"({len(seqs) / ref_total:6.1f} seq/s)")
        emit("")
        emit(f"  speedup         : {ref_total / ours_total:.1f}x")
    else:
        emit("  histo_hmm not installed; no comparison")

    emit("")
    emit("  Each sequence is scored against all 251 profile HMMs. histo_hmm's kernel is")
    emit("  numba-JIT'd and single-threaded; the win here is parallelism plus a cache-friendly")
    emit("  parameter layout, not compiled-vs-interpreted.")

    (TMP / "04_benchmark.txt").write_text(out.getvalue())
    say("")


def main() -> int:
    records = load_sequences()
    clf = get_reference()

    say("polars-mhci-hmm validation")
    say(f"  models    : {polars_mhci_hmm.model_dir()}")
    say(f"  loci      : {len(polars_mhci_hmm.loci())}")
    say(f"  sequences : {len(records)}")
    say(f"  histo_hmm : {'installed' if clf else 'NOT INSTALLED -- parity sections skipped'}")
    say("")

    if clf is not None:
        section_parity(records, clf)
        section_constructs(records, clf)
    section_assignments(records)
    section_benchmark(records, clf)

    say("=" * 78)
    say(f"artefacts in {TMP}")
    for f in sorted(TMP.iterdir()):
        if f.name != ".gitkeep":
            say(f"  {f.name:32s} {f.stat().st_size / 1024:8.1f} KB")

    (TMP / "00_SUMMARY.txt").write_text(summary.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
