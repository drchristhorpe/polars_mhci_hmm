"""Vendor histo_hmm's trained profile HMMs into `python/polars_mhci_hmm/models/`.

The `.npz` files are copied **byte-identical** from a histo_hmm checkout: this plugin reads
exactly the format histo_hmm writes, so re-syncing retrained models is a copy rather than a
conversion, and there is no chance of a transcoding bug silently changing a model.

The only thing this script *generates* is `manifest.json`, which extends histo_hmm's own
(``n_models`` + ``classes``) with two things the plugin wants:

  lengths   {class_name: L} -- so `polars_mhci_hmm.loci()` can report model lengths without
            importing numpy just to read one integer out of each of 251 archives.
  source    where these models came from, including the commit, so a vendored copy is
            traceable back to the checkout that produced it.

Both additions are backwards compatible: histo_hmm's own `load_models` reads `classes` and
ignores the rest, so a vendored directory still works as a histo_hmm model dir.

Run:  uv run python codegen/vendor_models.py --histo-hmm <path-to-checkout>

Without --histo-hmm, a shallow clone is made into `vendor/histo_hmm` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "python" / "polars_mhci_hmm" / "models"
REPO = "https://github.com/drchristhorpe/histo_hmm"

# The alphabet histo_hmm trains against (histo_hmm/alphabet.py). Vendored models must match,
# otherwise the plugin's hardcoded 20-letter encoding would silently misread them.
ALPHABET_SIZE = 20


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _clone(dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {REPO} -> {dest}")
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(dest)], check=True)
    return dest


def _validate(npz_path: Path, background: np.ndarray) -> int:
    """Check one model archive is shaped the way the Rust kernel assumes. Returns L."""
    data = np.load(npz_path)

    missing = {"match_emit", "insert_emit", "trans", "length"} - set(data.files)
    if missing:
        raise SystemExit(f"{npz_path.name}: missing array(s) {sorted(missing)}")

    length = int(data["length"][0])
    expected = {
        "match_emit": (length + 1, ALPHABET_SIZE),
        "insert_emit": (length + 1, ALPHABET_SIZE),
        "trans": (length + 1, 7),
    }
    for name, shape in expected.items():
        arr = data[name]
        if arr.shape != shape:
            raise SystemExit(f"{npz_path.name}: {name} has shape {arr.shape}, expected {shape}")
        if arr.dtype != np.float64:
            raise SystemExit(f"{npz_path.name}: {name} has dtype {arr.dtype}, expected float64")
        if np.isnan(arr).any():
            raise SystemExit(f"{npz_path.name}: {name} contains NaN")
        # -inf is expected (row 0 of match_emit is unused), +inf is not.
        if np.isposinf(arr).any():
            raise SystemExit(f"{npz_path.name}: {name} contains +inf")

    if background.shape != (ALPHABET_SIZE,):
        raise SystemExit(f"background.npy has shape {background.shape}, expected ({ALPHABET_SIZE},)")

    return length


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--histo-hmm",
        type=Path,
        default=None,
        help="path to a histo_hmm checkout (default: shallow-clone into vendor/histo_hmm)",
    )
    args = ap.parse_args()

    checkout = args.histo_hmm or _clone(ROOT / "vendor" / "histo_hmm")
    src = checkout / "src" / "histo_hmm" / "models"
    if not (src / "manifest.json").exists():
        raise SystemExit(f"no manifest.json under {src} -- is that a histo_hmm checkout?")

    upstream = json.loads((src / "manifest.json").read_text())
    classes = upstream["classes"]

    # Manifest order is load-bearing: it is the tie-break order for `top_loci` (PLAN.md §5.2).
    # histo_hmm writes it sorted; assert rather than assume, and never re-sort here.
    if classes != sorted(classes):
        raise SystemExit("upstream manifest classes are not sorted -- tie-break order would drift")

    background = np.load(src / "background.npy")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    shutil.copy2(src / "background.npy", DEST / "background.npy")

    lengths: dict[str, int] = {}
    for name in classes:
        npz = src / f"{name}.npz"
        if not npz.exists():
            raise SystemExit(f"manifest lists {name} but {npz} is missing")
        lengths[name] = _validate(npz, background)
        shutil.copy2(npz, DEST / f"{name}.npz")

    try:
        commit = _git(checkout, "rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        commit = "unknown"

    manifest = {
        "n_models": len(classes),
        "classes": classes,
        "lengths": lengths,
        "alphabet_size": ALPHABET_SIZE,
        "source": {
            "repo": REPO,
            "commit": commit,
            "vendored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "note": "models copied byte-identical from histo_hmm; see codegen/vendor_models.py",
        },
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    size_mb = sum(f.stat().st_size for f in DEST.iterdir()) / 1e6
    print(f"vendored {len(classes)} models ({size_mb:.2f} MB) from {commit[:8]} -> {DEST}")
    print(f"model lengths: min={min(lengths.values())} max={max(lengths.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
