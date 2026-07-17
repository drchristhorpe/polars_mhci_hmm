# PLAN — `polars-mhci-hmm`: a Polars plugin for MHC Class I HMM classification

A native (Rust) Polars expression plugin that classifies protein sequences as MHC Class I and
predicts their locus, **reproducing the outputs of [`histo_hmm`](https://github.com/drchristhorpe/histo_hmm)**.

---

## 1. Goal & success criterion

Expose a Polars expression namespace so that this works on a `DataFrame`/`LazyFrame` column:

```python
import polars as pl
import polars_mhci_hmm  # registers the `.mhci` namespace

df.with_columns(result=pl.col("sequence").mhci.classify())
```

**Success criterion (the whole point):** for any input sequence and any parameter combination,
the struct the plugin returns equals what `histo_hmm` returns:

```python
df.select(pl.col("sequence").mhci.classify(**kw)).to_series().to_list()
==
[asdict(MHCClassIClassifier(**ctor).classify(s, **kw)) for s in sequences]
```

This is not an aspiration — it is enforced by a differential test against the real `histo_hmm`
package, installed from git as a dev dependency (§7). Anything that does not match `histo_hmm`
is a bug.

Non-goal: inventing a "better" classifier. Where `histo_hmm` is quirky, we are quirky. The one
place we cannot promise bit-identity is floating-point summation order, which is bounded,
measured and documented (§5).

Non-goal: **training**. It needs MAFFT, it is a one-off, and `histo_hmm` already does it. The
trained models are the *input* to this project, not its output.

---

## 2. Why a Rust plugin (and not a Python UDF)

Because classification is genuinely expensive, and the existing implementation is structurally
slow rather than incidentally slow.

Scoring one sequence means a Viterbi pass against **each of 251 profile HMMs**, each O(T×L) with
T ≈ L ≈ 275 — roughly 190 MFLOP per sequence. A sequence long enough to look like a construct
triggers a sliding-window scan that re-scores ~104 overlapping windows: ~26,000 Viterbi passes
for a single input.

Measured on the reference implementation (16-core machine, numba JIT warm):

| case | `histo_hmm` |
|---|---|
| full-length (~275 aa) | **0.352 s/seq** → 2.8 seq/s |
| construct (413 aa, scan path) | **37.4 s/seq** |

`histo_hmm`'s kernel is already numba-JIT'd, so the win is not "compiled beats interpreted" —
it is that the work is single-threaded and driven per-row from Python. A Polars **expression
plugin** (`pyo3-polars`) compiles to a shared library that Polars calls directly on the Arrow
buffers. It runs inside the engine: parallel over rows, no GIL, no per-row Python object churn,
and it composes with the lazy optimiser.

Benchmark vs. `histo_hmm` is part of the deliverable (§7), so the claim gets measured rather
than asserted.

---

## 3. Public API

```python
import polars as pl
import polars_mhci_hmm  # registers the `.mhci` namespace

# The main event: one struct column mirroring histo_hmm's ClassificationResult.
df.with_columns(result=pl.col("sequence").mhci.classify())
df.select(pl.col("sequence").mhci.classify().struct.field("top_loci"))

# Single-model log-odds, mirroring ProfileHMM.log_odds_score.
df.with_columns(score=pl.col("sequence").mhci.score("hla_a"))

# The loci this build knows about -- mirrors polars_seq.codon_tables().
polars_mhci_hmm.loci()
```

`classify(n_top=10, scan_constructs=True, threshold=0.0, model_dir=None)` → `Struct`, mirroring
`ClassificationResult` field-for-field:

| field | dtype | reference source |
|---|---|---|
| `is_class_i` | `Boolean` | `best_score >= threshold` |
| `confidence` | `Float64` | `1/(1+exp(-clamp(best, -50, 50)))` |
| `best_score` | `Float64` | `max(scores.values())` |
| `region_start` | `UInt32` | scan result, else `0` |
| `region_end` | `UInt32` | scan result, else `len(cleaned)` |
| `top_loci` | `List(Struct{locus: String, probability: Float64})` | top `n_top` softmax probabilities |

`ClassificationResult.raw_scores` is deliberately **omitted**: `classify()` never populates it
(it is always `None`). Single-model log-odds are available via `.mhci.score(locus)`.

Conventions:

- **Null input → null output**, following `polars-seq`. `histo_hmm` has no opinion here; a null
  is absence of a sequence, not an empty one.
- **Empty / all-invalid sequence** reproduces the reference's degenerate result exactly:
  `is_class_i=False, confidence=0.0, top_loci=[], best_score=-inf, region=[0, 0]`.

Argument validation lives in Python (§4), so a bad `locus` or a nonexistent `model_dir` raises a
clean `ValueError` at expression-construction time rather than a `ComputeError` from deep inside
the query engine.

---

## 4. Layout

Mirrors `polars-seq`: a compiled kernel, a generated/vendored data artefact, and a thin Python
layer that validates and registers.

```
polars_mhci_hmm/
├── src/
│   ├── lib.rs              # #[pymodule] _internal, mod decls
│   ├── expressions.rs      # #[polars_expr] entry points, kwargs structs
│   ├── classify.rs         # port of classify.py: clean, scan, softmax, confidence
│   ├── hmm.rs              # port of _viterbi_core + log_odds_score
│   ├── models.rs           # model-set loading + process-wide cache
│   └── npy.rs              # minimal .npy/.npz reader
├── python/polars_mhci_hmm/
│   ├── __init__.py         # .mhci namespace, validation, loci()
│   ├── py.typed
│   └── models/             # VENDORED: 251 .npz + background.npy + manifest.json (~0.5 MB)
├── codegen/vendor_models.py  # regenerate models/ from a histo_hmm checkout
├── tests/                  # conftest, parity, polars, golden, edge cases
├── tools/validate.py       # writes artefacts to tmp/
├── tmp/                    # test + validation outputs land here (gitignored)
├── Cargo.toml  pyproject.toml  rust-toolchain.toml  .gitignore
└── PLAN.md  CHANGELOG.md  README.md  LICENSE
```

Build: maturin, `python-source = "python"`, `module-name = "polars_mhci_hmm._internal"`, cdylib
named `_internal`, `abi3-py310` — identical to `polars-seq`.

Rust dependencies: `polars`, `pyo3`, `pyo3-polars`, `serde`, plus `serde_json` (manifest),
`zip` (a `.npz` is a zip of `.npy` members) and `rayon` (row- and window-level parallelism).

### 4.1 Models

Vendored **byte-identical** from `histo_hmm/src/histo_hmm/models/` — the same `.npz` format, so
re-syncing retrained models is a file copy, and `codegen/vendor_models.py` re-runs it.

- 0.5 MB compressed on disk (deflate earns its keep: 44 KB → 4 KB per array), ~26 MB of f64
  once inflated.
- Loaded once into a process-wide cache (`Mutex<HashMap<PathBuf, Arc<ModelSet>>>`), so the
  inflate cost is paid once per process, not once per row.
- Our `manifest.json` extends `histo_hmm`'s with `lengths` and a `source` provenance block
  (repo + commit). It stays readable by `histo_hmm`'s own `load_models`, which only needs
  `classes`.
- A `model_dir=` argument accepts a custom model directory, matching the reference's
  `MHCClassIClassifier(model_dir=...)`.

`.npz` is parsed by a small hand-rolled reader (`npy.rs`) rather than pulling in `ndarray`: the
format is fixed, we vendor the exact files, and it keeps the dependency list as tight as
`polars-seq`'s. It validates dtype (`<f8`/`<i8`), C-order and shape, and errors clearly rather
than guessing.

---

## 5. The faithfulness contract

These are the details a casual port gets wrong. Each one is pinned by a test named after it.

1. **`_clean_sequence`**: uppercase, then keep only the 20 amino acids **and `X`**; everything
   else (`-`, `*`, `B`, `Z`, `?`) is *dropped*, changing the length. `X` survives → encodes to
   `-1` → uniform emission `log(1/20)`. Consequently `region_start`/`region_end` index the
   **cleaned** sequence, not the input.
1b. **The uppercasing is Unicode-aware**, because the reference's is: Python's `str.upper()`
   maps `'ß'` → `"SS"` and `'ı'` → `"I"`, letters that then survive cleaning and get scored. A
   bytewise ASCII uppercase would drop them and silently score a shorter sequence. ASCII input
   (i.e. all real protein data) takes a fast byte path; non-ASCII falls back to
   `str::to_uppercase`. Filtering the result bytewise stays correct because a UTF-8 multi-byte
   sequence never contains an ASCII byte.
2. **Stable tie-breaking in `top_loci`**: `sorted(..., reverse=True)` is stable in Python, so
   equal probabilities keep manifest order (sorted class names). Rust must use a *stable* sort
   by probability descending, iterating models in manifest order.
3. **Scan tie-breaking**: `_scan_sequence` compares with strict `>`, so the *first* window in
   (window_size outer, start inner) order wins. Parallel reduction must break ties by lowest
   sequential index.
4. **Scan fallback**: if no window beats `-inf`, the region stays `(0, seq_len)` — not
   `(0, 200)`.
5. **Scan trigger**: score the full sequence iff `len <= 370` (`_MAX_WINDOW + 50`) or
   `scan_constructs=False`. Otherwise windows are `range(200, min(321, len+1), 10)` with starts
   `range(0, len-w+1, 20)`.
6. **`match_emit[0]` is all `-inf`** (row 0 is unused; match states are 1..L). `-inf + finite =
   -inf` in both numpy and Rust f64, and no NaN arises on any reachable path.
7. **Softmax temperature is 30.0**: `values/30`, shifted by max, then normalised.
8. **Confidence clamps to ±50** before the sigmoid.
9. **Model lengths vary** (L ∈ [273, 282]) — nothing may hardcode 275.

### 5.1 Numerical fidelity — a known, bounded gap

Viterbi is max-plus: no summation reordering, so the model score is reproducible **exactly**.

Two places use `np.sum`, which numpy computes with *pairwise* summation: the null score
(~275 terms) and the softmax denominator (251 terms). A straightforward Rust sum differs from
pairwise by a few ULP — on log-odds of magnitude ~700 that is ~1e-13 absolute. Separately,
`exp` may differ by 1 ULP between numpy's vectorised implementation and Rust's libm.

**Decision:** implement the straightforward sum and assert **tolerances** (1e-9 absolute on
log-odds, 1e-12 on probabilities) rather than reimplementing numpy's internal pairwise
summation, which would over-fit to a numpy implementation detail that is not part of its API.
`is_class_i`, `region_start`/`region_end` and locus *ordering* are asserted **exactly**.

This gap is documented in the README rather than papered over. Asserted tolerances carry ~4
orders of magnitude of headroom over the expected error. If parity tests ever exceed them, the
fallback is to port numpy's pairwise summation.

---

## 6. Performance approach

Preserve semantics exactly; win by removing overhead and adding parallelism, never by changing
the algorithm.

- Row-level parallelism with rayon inside the expression.
- Within a construct scan, parallelise over the ~104 (window_size, start) pairs, reducing to the
  running best `(score, idx, scores)` — memory O(threads × 251) rather than materialising every
  window's scores. The reduction breaks ties by lowest index, so it is deterministic and
  matches the reference's sequential first-wins rule.
- Flat row-major `Vec<f64>` model parameters, contiguous access in the inner loop.
- **No algorithmic shortcuts** — no early termination, no pruning, no score caching across
  overlapping windows. They would change outputs, and outputs are the contract.

Target: ≫10× on full-length sequences, and the 37 s construct case into the low seconds.

---

## 7. Verification

Dev environment: `uv venv --python 3.12`, with
`histo-hmm @ git+https://github.com/drchristhorpe/histo_hmm` as a dev dependency. Verified
working: polars 1.42.1 + numpy 2.4.6 + numba 0.66.0, 251 loci loading. MAFFT is absent but not
needed — it is a training-only dependency.

Test suite, with all artefacts written to `tmp/`:

- **`test_parity.py`** — the test the whole project rests on. Ours vs `histo_hmm` on real
  sequences: full-length, truncated/partial, constructs (the scan path), `X`-bearing,
  lowercase, punctuated, non-MHC negatives, and empty. Asserts `is_class_i` and locus *order*
  exactly, scores within tolerance. Dumps a per-sequence comparison table to `tmp/`.
- **`test_polars.py`** — integration: nulls, empty frames, struct schema and dtypes, lazy
  frames, multiple chunks, `group_by`, and row-order stability under parallelism.
- **`test_golden.py`** — values checked into the repo, so the suite stays meaningful without
  `histo_hmm` installed (parity tests skip if it is absent).
- **`test_edge_cases.py`** — the §5 contract, one test per rule.

Test data: `tests/data/mhci_sequences.json`, a small curated sample drawn from `histo_hmm`'s
`data/cytoplasmic_sequences/`, so the suite is hermetic and needs no network.

`tools/validate.py` (mirroring `polars-seq`'s) writes inspectable artefacts to `tmp/`:

```
tmp/00_SUMMARY.txt              read this one first
tmp/01_parity_full_length.csv   ours vs histo_hmm, per sequence
tmp/02_parity_constructs.csv    ours vs histo_hmm, scan path
tmp/03_golden.csv               the documented edge cases
tmp/04_benchmark.txt            throughput vs histo_hmm
tmp/05_classified.parquet       a classified dataset
```

End-to-end: `maturin develop`, then run the README's example on a real HLA-A sequence and
confirm the top locus is `hla_a` with the reference's probabilities
(`[('hla_a', 0.8784), ('mafa_b', 0.0179), ('mamu_b', 0.0177)]`).

---

## 8. Risks

| risk | mitigation |
|---|---|
| Float parity beyond tolerance | Tolerances have ~4 orders of magnitude of headroom (~1e-13 expected vs 1e-9 asserted); measured, not assumed. Fallback: port numpy's pairwise summation. |
| npz reader bugs | Strict validation of dtype, order and shape; we vendor the exact files; golden tests compare loaded parameters against numpy-loaded values. |
| Wheel size / load cost | 0.5 MB vendored; ~26 MB inflated once into a cached `ModelSet`; milliseconds, paid once per process. |
| Memory during scan | Reduce-to-best rather than materialising all windows. |
| Model re-sync drift | `codegen/vendor_models.py`, with the source commit recorded in the manifest. |

---

## 9. Status

Tracked in [CHANGELOG.md](CHANGELOG.md). Target for the first release: `0.1.0`.
