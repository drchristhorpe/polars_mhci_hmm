"""Regenerate `tests/data/mhci_sequences.json` from a histo_hmm checkout.

The test suite needs real MHC Class I sequences, but histo_hmm ships its training data in the
repository rather than in the installed package -- so a test that imported histo_hmm still could
not reach them. Rather than have the suite reach for the network, a small curated sample is
checked in.

Sampling is deterministic (a fixed seed and sorted inputs), so re-running this on the same
checkout reproduces the same file, and a diff means the upstream data actually changed.

Run:  uv run python tools/make_test_data.py --histo-hmm <path-to-checkout>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tests" / "data" / "mhci_sequences.json"

SEED = 20260716

# Loci chosen to span the tree: human, macaque, mouse, cow, pig, chicken, salmon -- so a
# mis-ported model or a locus-ordering bug shows up as a wrong *neighbour*, not just a wrong
# probability.
LOCI = [
    "hla_a",
    "hla_b",
    "hla_c",
    "hla_e",
    "mamu_a1",
    "mamu_b",
    "patr_a",
    "gogo_b",
    "h2_k",
    "h2_d",
    "bola_1",
    "sla_1",
    "gaga_bf1",
    "sasa_uba",
]
PER_LOCUS = 3

# Not MHC Class I: the classifier should place these well below threshold. Sourced from
# well-known sequences rather than randomly generated, so they are realistic negatives.
NEGATIVES = {
    # Human ubiquitin (P0CG48), a small, highly conserved, unrelated fold.
    "ubiquitin": (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ),
    # Human beta-2 microglobulin (P61769) -- the Class I *light* chain: related in biology,
    # but not a Class I alpha chain, so it must not be classified as one.
    "b2m": (
        "MSRSVALAVLALLSLSGLEAIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSF"
        "SKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
    ),
    # Hen egg lysozyme (P00698).
    "lysozyme": (
        "MRSLLILVLCFLPLAALGKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQI"
        "NSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--histo-hmm", type=Path, required=True, help="path to a histo_hmm checkout")
    args = ap.parse_args()

    data_dir = args.histo_hmm / "data" / "cytoplasmic_sequences"
    if not data_dir.is_dir():
        raise SystemExit(f"no {data_dir} -- is that a histo_hmm checkout?")

    rng = random.Random(SEED)
    records = []

    for locus in LOCI:
        path = data_dir / f"{locus}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        raw = json.loads(path.read_text())

        # histo_hmm's data_loader keys each file by the raw (gapped) sequence and strips
        # '-'/'?' to clean it. Do the same, then sample from the sorted set for determinism.
        cleaned = sorted({s.replace("-", "").replace("?", "") for s in raw.keys() if s.strip()})
        if not cleaned:
            raise SystemExit(f"{path} yielded no sequences")

        for i, seq in enumerate(rng.sample(cleaned, min(PER_LOCUS, len(cleaned)))):
            records.append({"name": f"{locus}_{i}", "locus": locus, "kind": "class_i", "sequence": seq})

    for name, seq in NEGATIVES.items():
        records.append({"name": name, "locus": None, "kind": "negative", "sequence": seq})

    DEST.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "note": (
            "Curated sample for the test suite. Class I sequences are drawn deterministically "
            "(seed %d) from histo_hmm's data/cytoplasmic_sequences/, cleaned the way its "
            "data_loader cleans them. Regenerate with tools/make_test_data.py." % SEED
        ),
        "sequences": records,
    }
    DEST.write_text(json.dumps(payload, indent=2) + "\n")

    n_pos = sum(1 for r in records if r["kind"] == "class_i")
    print(f"wrote {len(records)} sequences ({n_pos} class I, {len(NEGATIVES)} negative) -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
