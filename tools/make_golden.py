"""Regenerate `tests/data/golden.json` -- expected classifications, produced by histo_hmm.

The parity suite needs histo_hmm installed. These golden values are its fossil record: generated
once *from the reference*, checked in, and asserted against thereafter. That keeps the test suite
meaningful for anyone who has only the plugin -- a CI job on a machine without histo_hmm, a user
debugging a wheel -- and it pins the reference's behaviour at a known commit, so a silent change
upstream shows up as a diff here rather than as a mystery.

Regenerating requires histo_hmm, and the diff should be empty unless the models or the
reference's logic actually changed.

Run:  uv run python tools/make_golden.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars_mhci_hmm

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tests" / "data" / "golden.json"
SEQUENCES = ROOT / "tests" / "data" / "mhci_sequences.json"

N_TOP = 5


def main() -> int:
    try:
        import histo_hmm
    except ImportError:
        raise SystemExit(
            "histo_hmm is required to regenerate golden values:\n"
            "  uv sync   (installs it from git as a dev dependency)"
        )

    records = json.loads(SEQUENCES.read_text())["sequences"]

    # Score against *our* vendored models, so the golden file describes this repo's models.
    model_dir = polars_mhci_hmm.model_dir()
    clf = histo_hmm.MHCClassIClassifier(model_dir=model_dir)
    manifest = json.loads((Path(model_dir) / "manifest.json").read_text())

    # histo_hmm's fields are numpy scalars (is_class_i is a np.bool_, scores are np.float64).
    # json refuses those, and a golden file full of numpy reprs would be unreadable anyway.
    cases = []
    for rec in records:
        r = clf.classify(rec["sequence"], n_top=N_TOP)
        cases.append(
            {
                "name": rec["name"],
                "kind": rec["kind"],
                "expected_locus": rec["locus"],
                "is_class_i": bool(r.is_class_i),
                "confidence": float(r.confidence),
                "best_score": float(r.best_score),
                "region_start": int(r.region_start),
                "region_end": int(r.region_end),
                "top_loci": [{"locus": k, "probability": float(p)} for k, p in r.top_loci],
            }
        )

    payload = {
        "note": (
            "Expected classifications, generated from histo_hmm against this repo's vendored "
            "models. Regenerate with tools/make_golden.py; a non-empty diff means the reference "
            "or the models changed."
        ),
        "generated_from": {
            "histo_hmm_version": getattr(histo_hmm, "__version__", "unknown"),
            "models_commit": manifest.get("source", {}).get("commit", "unknown"),
            "n_top": N_TOP,
        },
        "cases": cases,
    }
    DEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {len(cases)} golden cases -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
