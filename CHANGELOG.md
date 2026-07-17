# Changelog

All notable changes to `polars-mhci-hmm` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-07-17

The initial release. `polars-mhci-hmm` reimplements
[`histo_hmm`](https://github.com/drchristhorpe/histo_hmm)'s MHC Class I classification as a
native Polars expression plugin, laid out after
[`polars-seq`](https://github.com/drchristhorpe/polars-seq). Classification only — training
stays in `histo_hmm`, which needs MAFFT and only runs once.

### Added

- **Project scaffold** — maturin build (`python-source = "python"`, cdylib `_internal`,
  `abi3-py310`), pinned stable Rust toolchain, MIT licence. Mirrors `polars-seq`'s layout so the
  two projects read the same way.
- **[PLAN.md](PLAN.md)** — goal, success criterion, the faithfulness contract, and the
  numerical-fidelity gap, written up before the code.
- **`.mhci.classify()`** — the main expression. Returns one struct column mirroring
  `histo_hmm`'s `ClassificationResult` field-for-field (`is_class_i`, `confidence`,
  `best_score`, `region_start`, `region_end`, `top_loci`), with `n_top`, `scan_constructs`,
  `threshold` and `model_dir` arguments matching `MHCClassIClassifier`. Null in → null out.
  `raw_scores` is omitted deliberately: the reference never populates it.
- **`.mhci.score(locus)`** — log-odds against a single locus, mirroring
  `ProfileHMM.log_odds_score`.
- **`polars_mhci_hmm.loci()`** — the 251 loci and their model lengths, as a DataFrame.
  `model_dir()` exposes the bundled models' path.
- **Vendored models** — 251 profile HMMs (0.51 MB) copied byte-identical from `histo_hmm`
  @`5cf4329c`, with `codegen/vendor_models.py` to re-sync them. The vendored `manifest.json`
  adds `lengths` and a `source` provenance block while staying readable by `histo_hmm`'s own
  loader. A hand-rolled `.npy`/`.npz` reader (`src/npy.rs`) keeps the dependency list tight;
  models load once per process into a shared cache.
- **Argument validation in Python** — an unknown locus, a bad `model_dir` or a nonsense `n_top`
  raises `ValueError`/`TypeError` at expression-construction time rather than a `ComputeError`
  from inside the query engine. Unknown loci suggest near-misses.
- **Test suite** — parity against a live `histo_hmm` (the test the project rests on), checked-in
  golden values so the suite still means something without it, Polars integration, and one test
  per rule of the faithfulness contract. Artefacts land in `tmp/`.
- **`tools/validate.py`** — writes inspectable parity, assignment and benchmark artefacts to
  `tmp/`. `tools/make_test_data.py` and `tools/make_golden.py` regenerate the fixtures.
- **CI** ([`ci.yml`](.github/workflows/ci.yml)) — the parity suite on Python 3.12 and 3.14 (the
  ends of the range where `histo_hmm` installs); an **abi3-floor** job that installs the built
  wheel on 3.10 with only polars beside it, proving both that the floor works and that the golden
  tests keep the suite meaningful without the reference; a **vendored-models** job that re-vendors
  from the commit the manifest names and fails on any diff; and rustfmt + clippy.
- **Release** ([`release.yml`](.github/workflows/release.yml), [RELEASING.md](RELEASING.md)) —
  push a `v*` tag and it builds one abi3 wheel per platform (Linux x86_64/aarch64, macOS
  arm64/x86_64, Windows x64) plus an sdist, then publishes via PyPI Trusted Publishing. Before
  publishing it checks the tag agrees with `pyproject.toml` and `Cargo.toml`, counts the 251
  models in the sdist, and installs the wheel on a Python it was not built against to classify
  every test sequence against the golden values.

### Performance

- **Transposed model parameters.** numpy stores emissions as `(L+1) × 20`, so a Viterbi row —
  which sweeps model positions with the residue fixed — strode by 20 f64 and touched a fresh
  cache line every step. Storing them `20 × (L+1)` turns each sweep into a contiguous run.
- **Bounds-check-free inner loops.** The match and insert recursions depend only on the previous
  row, so they carry no loop-carried dependency and can vectorise — but only once every operand
  is sliced to a common length, which lets LLVM drop the bounds checks. The delete chain is
  inherently sequential and stays scalar.
- **Buffer swaps instead of copies** between Viterbi rows, and reusable scratch buffers so
  scoring 251 models does not allocate 1,506 vectors.
- **Parallelism across models and scan windows** rather than across rows, so a one-row frame —
  what an interactive session actually runs — still uses every core. `score()` parallelises
  across rows instead, since scoring one locus gives it no per-row fan-out to ride on: ~24,000
  sequences/s against ~1,500 single-threaded.

Measured against `histo_hmm` on a 16-core machine: **~14×** on full-length sequences (160 → 11
ms/seq) and on the construct-scan path (19.1 s → 1.4 s for one 415 aa sequence). The kernel
itself is only modestly faster than numba (0.68 vs 0.80 ms per model); most of the win is cores.

### Fixed (found in review, before first release)

- **Non-ASCII sequences cleaned differently from the reference.** `histo_hmm` uppercases with
  Python's `str.upper()`, which is Unicode-aware: `'ß'` becomes `"SS"` and `'ı'` becomes `"I"` —
  letters that survive cleaning and get scored. A bytewise ASCII uppercase dropped them instead
  and silently scored a shorter sequence (`"MAKß"`: 5 residues vs our 3). ASCII input keeps the
  fast byte path; non-ASCII now falls back to a Unicode uppercase. Pinned by a parity test.
- **`score()` ran single-threaded**, iterating rows sequentially against one model — so unlike
  `classify()` it never used more than one core.
- **A construction error in `top_loci` became a null row** rather than propagating, presenting a
  failed classification as a valid empty one.
- `score()`'s documentation claimed an empty sequence scores 0.0; it returns the all-delete path
  score.
- **`uv sync` failed outright** — the README's own build-from-source instruction. `histo_hmm`
  requires Python ≥ 3.12 while this package supports ≥ 3.10, and uv resolves across the whole
  `requires-python` range, so it rejected the set as unsatisfiable. The dev dependency now
  carries a `python_version >= '3.12'` marker, which is what it always meant.
- **`tmp/.gitkeep` was silently ignored**, so `tmp/` would not have existed in a fresh clone:
  git does not descend into a directory excluded by `tmp/`, which makes the `!tmp/.gitkeep`
  negation unreachable. Now `tmp/*` + `!tmp/.gitkeep`.
- Vendoring is now byte-reproducible: the `vendored_at` timestamp is gone, since the recorded
  commit is the real provenance and a timestamp made every re-vendor look like a change. This is
  what lets CI verify the committed models by re-vendoring them.

<!-- Entries below are appended as work lands; see PLAN.md §5 for the faithfulness contract. -->
