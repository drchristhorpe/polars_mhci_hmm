# polars-mhci-hmm

[![CI](https://github.com/drchristhorpe/polars_mhci_hmm/actions/workflows/ci.yml/badge.svg)](https://github.com/drchristhorpe/polars_mhci_hmm/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/polars-mhci-hmm)](https://pypi.org/project/polars-mhci-hmm/)
[![Python](https://img.shields.io/pypi/pyversions/polars-mhci-hmm)](https://pypi.org/project/polars-mhci-hmm/)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/drchristhorpe/polars_mhci_hmm/blob/master/LICENSE)

Classify MHC Class I protein sequences inside Polars.

A native Polars expression plugin that reproduces
[`histo_hmm`](https://github.com/drchristhorpe/histo_hmm)'s classifier — the same 251 trained
profile HMMs, the same Viterbi recursion, the same outputs — with the scoring kernel in Rust so
it runs inside the query engine, across every core, without the GIL.

```python
import polars as pl
import polars_mhci_hmm  # registers the .mhci namespace

df.with_columns(result=pl.col("sequence").mhci.classify())
```

---

## Install

```bash
uv add polars-mhci-hmm
# or
pip install polars-mhci-hmm
```

The 251 trained models ship inside the wheel (0.5 MB); there is nothing to download or train.

## Usage

`classify()` returns a single struct column, mirroring `histo_hmm`'s `ClassificationResult`:

```python
>>> df = pl.DataFrame({"sequence": [hla_a_seq]})
>>> df.select(pl.col("sequence").mhci.classify(n_top=3)).to_series()[0]
{'is_class_i': True,
 'confidence': 1.0,
 'best_score': 740.8816401083320,
 'region_start': 0,
 'region_end': 273,
 'top_loci': [{'locus': 'hla_a', 'probability': 0.8783894...},
              {'locus': 'mafa_b', 'probability': 0.0179271...},
              {'locus': 'mamu_b', 'probability': 0.0176760...}]}
```

Unpack it with `.struct.field()`, or `unnest` it:

```python
df.with_columns(
    is_class_i=pl.col("sequence").mhci.classify().struct.field("is_class_i"),
    locus=pl.col("sequence").mhci.classify(n_top=1).struct.field("top_loci"),
)

# One row per (sequence, locus) prediction:
(
    df.select("id", pl.col("sequence").mhci.classify(n_top=5).struct.field("top_loci"))
      .explode("top_loci")
      .unnest("top_loci")
)
```

| field | dtype | meaning |
|---|---|---|
| `is_class_i` | `Boolean` | `best_score >= threshold` |
| `confidence` | `Float64` | sigmoid of the best log-odds, clamped to ±50 |
| `best_score` | `Float64` | raw log-odds of the top-scoring locus |
| `region_start` | `UInt32` | start of the detected MHC region |
| `region_end` | `UInt32` | end of the region, exclusive |
| `top_loci` | `List(Struct{locus, probability})` | the best `n_top` loci, best first |

### Arguments

```python
pl.col("sequence").mhci.classify(
    n_top=10,               # how many loci to return
    scan_constructs=True,   # scan long sequences for an embedded MHC region
    threshold=0.0,          # log-odds threshold for the is_class_i call
    model_dir=None,         # a custom directory of models, in histo_hmm's format
)
```

`probability` values are temperature-scaled softmax over **all 251 loci**, so they sum to 1.0
across the full set — not across `top_loci`. Ask for `n_top=251` and they sum to one.

### Single-locus scores

`score()` gives the raw log-odds against one locus, mirroring `ProfileHMM.log_odds_score`:

```python
df.with_columns(
    hla_a=pl.col("sequence").mhci.score("hla_a"),
    h2_k=pl.col("sequence").mhci.score("h2_k"),
)
```

Positive means the sequence is more likely under that locus' model than under the background.
Unlike `classify()`, there is no sliding-window scan — the whole sequence is scored.

### Which loci?

```python
>>> polars_mhci_hmm.loci()
shape: (251, 2)
┌────────┬────────┐
│ locus  ┆ length │
│ ---    ┆ ---    │
│ str    ┆ u32    │
╞════════╪════════╡
│ aole_f ┆ 275    │
│ aotr_g ┆ 275    │
│ …      ┆ …      │
└────────┴────────┘
```

Human (`hla_*`), mouse (`h2_*`), macaque (`mamu_*`, `mafa_*`), cow, pig, chicken, salmon and
more — 251 loci across the vertebrates.

---

## Things worth knowing

### `region_start`/`region_end` index the *cleaned* sequence

Both this plugin and `histo_hmm` clean an input before scoring it: uppercase, then keep only the
20 amino acids **and `X`**. Everything else — gaps, `*`, and the ambiguity codes `B`/`Z`/`J` — is
**dropped**, which shortens the sequence. The region offsets refer to that cleaned string, so if
your input has gaps or punctuation they will not line up with it. Clean your sequences yourself
if you need offsets into the original.

`X` is the exception: it survives cleaning and is scored with a uniform emission, so it shortens
nothing but does change the score.

### Long sequences are scanned, and it is expensive

A sequence longer than ~370 residues is assumed to be a construct with MHC embedded in it, and is
scanned with sliding windows of 200–320 residues. That means scoring ~104 windows against all 251
loci — around 26,000 Viterbi passes for one sequence. It is the right answer for a fusion protein
and a waste for a long non-MHC sequence. Pass `scan_constructs=False` to always score the whole
sequence.

### Nulls and empty sequences

A null sequence classifies to a null struct — the absence of a sequence is not a classification.
A sequence with no scorable residues (`""`, `"---"`, `"123"`) reproduces the reference's
degenerate result: `is_class_i=False`, `confidence=0.0`, `top_loci=[]`, `best_score=-inf`,
`region=[0, 0]`.

### Numerical fidelity

Model scores reproduce `histo_hmm` **exactly**: Viterbi is max-plus, so there is no summation to
reorder.

Two steps do sum — the null score and the softmax denominator — and `histo_hmm` sums them with
numpy's pairwise `np.sum` where this plugin sums sequentially. That, plus an occasional 1-ULP
difference in `exp`, puts the two implementations about **1e-12 apart** on log-odds of magnitude
~700, and about **1e-16** apart on probabilities. The test suite asserts `is_class_i`, the region
and the *ordering* of loci exactly, and the floats to 1e-9 / 1e-12.

Reimplementing numpy's internal pairwise summation would close the last few ULPs, at the cost of
pinning this package to an implementation detail that is not part of numpy's API. That trade
wasn't worth it. See [PLAN.md](https://github.com/drchristhorpe/polars_mhci_hmm/blob/master/PLAN.md) §5.1.

---

## Performance

Each sequence is scored against all 251 profile HMMs — about 190 MFLOP of Viterbi. `histo_hmm`'s
kernel is already numba-JIT compiled, so this is not a compiled-versus-interpreted story: the win
is parallelism, plus a cache-friendly parameter layout, plus not paying Python's per-row cost.

Measured on a 16-core x86_64 machine, classifying real MHC Class I sequences against all 251
loci. The machine was under other load, and repeat runs varied between roughly 14× and 17×, so
treat these as a floor rather than a headline:

| case | `histo_hmm` | `polars-mhci-hmm` | |
|---|---|---|---|
| full-length (~275 aa) | 160 ms/seq (6.3 seq/s) | **11 ms/seq (90 seq/s)** | ~14× |
| construct, 415 aa (scan path) | 19.1 s | **1.4 s** | ~14× |
| `score()` over a column, one locus | — | **~24,000 seq/s** | |

The single-model Viterbi kernel is ~0.68 ms against numba's ~0.80 ms, so only a small part of the
win is the kernel itself — most of it is using more than one core.

Reproduce with `uv run python tools/validate.py`, which writes the numbers and a full parity
comparison into `tmp/`.

The two things that mattered most, in case they are useful elsewhere:

- **Transposing the model parameters.** numpy stores emissions as `(L+1) × 20`. A Viterbi row
  sweeps model positions with the residue fixed, so that layout strides by 20 f64 and touches a
  fresh cache line every step. Stored `20 × (L+1)`, each sweep is one contiguous run.
- **Letting the inner loops vectorise.** The match and insert recursions depend only on the
  previous row, so they have no loop-carried dependency — but LLVM only vectorises them once
  every operand is sliced to a common length and the bounds checks fall away. (The delete chain
  is inherently sequential; it stays scalar.)

---

## Building from source

Needs a Rust toolchain.

```bash
git clone https://github.com/drchristhorpe/polars_mhci_hmm
cd polars_mhci_hmm
uv sync
uv run maturin develop --release
uv run pytest
```

`uv sync` installs `histo_hmm` from git as a dev dependency, which is what the parity tests
compare against. `uv run pytest -m "not slow"` skips the construct-scan parity test, the only
slow one — it is slow because it runs *the reference's* scanner.

`histo_hmm` needs Python ≥ 3.12, while this package supports ≥ 3.10. On 3.10 or 3.11 it is simply
left out of the resolution and the parity tests skip; the golden tests still run, so the suite
stays meaningful. Use 3.12+ if you want parity.

Releases are automated — push a `v*` tag. See [RELEASING.md](https://github.com/drchristhorpe/polars_mhci_hmm/blob/master/RELEASING.md).

To re-sync the models after retraining them in `histo_hmm`:

```bash
uv run python codegen/vendor_models.py --histo-hmm ../histo_hmm
uv run python tools/make_golden.py   # regenerate the checked-in expectations
```

---

## Please cite histo_hmm

This package is a fast reimplementation of `histo_hmm`'s classifier, not an independent piece of
work. The models it ships were trained by `histo_hmm`, the semantics it implements are
`histo_hmm`'s, its behaviour is what the test suite asserts against, and every subtlety
`polars-mhci-hmm` gets right is right because `histo_hmm` worked it out first.

If you use this package, please cite
[`histo_hmm`](https://github.com/drchristhorpe/histo_hmm). `polars-mhci-hmm` deliberately does
not offer itself as an alternative citation.

## See also

- [`histo_hmm`](https://github.com/drchristhorpe/histo_hmm) — the reference implementation, plus
  the training pipeline that produces the models. Train there; classify here.
- [`polars-seq`](https://github.com/drchristhorpe/polars-seq) — DNA/RNA → protein translation in
  Polars, whose layout this project follows.

## Licence

MIT. See [LICENSE](https://github.com/drchristhorpe/polars_mhci_hmm/blob/master/LICENSE).
