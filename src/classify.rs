//! A port of histo_hmm's `classify.py` -- `MHCClassIClassifier` and `ClassificationResult`.
//!
//! Every constant, comparison and tie-break here is chosen to match the reference rather than to
//! be defensible on its own terms: the contract is "same outputs", so where histo_hmm is quirky,
//! we are quirky. The subtleties that a casual port gets wrong are called out inline and pinned
//! by tests named after them (PLAN.md §5).

use std::cmp::Ordering;

use rayon::prelude::*;

use crate::hmm::{encode, log_odds_score, Scratch};
use crate::models::ModelSet;

// Typical MHC Class I alpha-chain lengths, from classify.py.
const MIN_WINDOW: usize = 200;
const MAX_WINDOW: usize = 320;
const WINDOW_STEP: usize = 10;
const SCAN_STRIDE: usize = 20;

/// `MHCClassIClassifier._SOFTMAX_TEMPERATURE`. Converts raw log-odds into a distribution;
/// lower is peakier.
const SOFTMAX_TEMPERATURE: f64 = 30.0;

/// The reference clamps the log-odds into ±50 before the sigmoid, to avoid overflow.
const CONFIDENCE_CLAMP: f64 = 50.0;

/// Sequences longer than this are scanned for an embedded MHC region rather than scored whole
/// (`seq_len <= _MAX_WINDOW + 50`).
const SCAN_THRESHOLD: usize = MAX_WINDOW + 50;

/// The Rust mirror of `ClassificationResult`.
///
/// `raw_scores` is absent because `classify()` never populates it -- it is always `None` on the
/// reference's result. Single-model log-odds are reachable via the `score` expression.
pub struct Classification {
    pub is_class_i: bool,
    pub confidence: f64,
    pub best_score: f64,
    pub region_start: u32,
    pub region_end: u32,
    /// `(model index, probability)`, already truncated to `n_top`, best first.
    pub top_loci: Vec<(usize, f64)>,
}

/// The result the reference returns for a sequence with no scorable residues.
fn empty_result() -> Classification {
    Classification {
        is_class_i: false,
        confidence: 0.0,
        best_score: f64::NEG_INFINITY,
        region_start: 0,
        region_end: 0,
        top_loci: Vec::new(),
    }
}

/// `_clean_sequence`: uppercase, then keep only the 20 amino acids **and `X`**.
///
/// Everything else -- gaps, `*`, and the ambiguity codes `B`/`Z`/`J`/`?` -- is *dropped*, which
/// shortens the sequence. That is why `region_start`/`region_end` index the cleaned sequence and
/// not the caller's input. `X` survives to encode as `-1` and score with a uniform emission.
pub fn clean_sequence(seq: &str) -> Vec<u8> {
    // The overwhelmingly common case: a protein sequence is ASCII, and uppercasing is a byte op.
    if seq.is_ascii() {
        return seq
            .bytes()
            .map(|b| b.to_ascii_uppercase())
            .filter(|b| is_kept(*b))
            .collect();
    }

    // The reference uppercases with Python's `str.upper()`, which is Unicode-aware and can turn
    // one character into several: 'ß' becomes "SS" and 'ı' becomes "I" -- letters that then
    // survive cleaning and get scored. An ASCII-only uppercase would drop them instead and
    // silently score a shorter sequence, so non-ASCII input takes the Unicode path.
    //
    // Filtering the result bytewise stays correct: a UTF-8 multi-byte sequence never contains an
    // ASCII byte, so any character that is still non-ASCII after uppercasing is dropped whole --
    // which is what Python does with it too.
    seq.to_uppercase().bytes().filter(|b| is_kept(*b)).collect()
}

/// `c in AA_TO_IDX or c == "X"` -- the 20 amino acids, plus X.
#[inline]
fn is_kept(b: u8) -> bool {
    b == b'X' || crate::hmm::encodes(b)
}

/// `max(scores.values())`, with numpy's NaN-propagating comparison semantics rather than
/// `f64::max`'s NaN-ignoring ones. Scores are finite or `-inf` in practice.
fn max_of(values: &[f64]) -> f64 {
    let mut it = values.iter().copied();
    let first = match it.next() {
        Some(v) => v,
        None => return f64::NEG_INFINITY,
    };
    it.fold(first, |acc, v| if v > acc { v } else { acc })
}

/// `_score_all_models`: log-odds of one encoded sequence against every model, in manifest order.
fn score_all_models(encoded: &[i8], set: &ModelSet) -> Vec<f64> {
    set.models
        .par_iter()
        .map_init(Scratch::new, |scratch, model| {
            log_odds_score(encoded, model, &set.background, scratch)
        })
        .collect()
}

/// `_normalise_scores`: temperature-scaled softmax over the raw log-odds.
///
/// Shifting by the max before `exp` is the reference's numerical-stability step, reproduced
/// here because it also changes the result in the last bits.
fn normalise_scores(scores: &[f64]) -> Vec<f64> {
    let scaled: Vec<f64> = scores.iter().map(|s| s / SOFTMAX_TEMPERATURE).collect();
    let max = max_of(&scaled);
    let exps: Vec<f64> = scaled.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

/// `_scores_to_confidence`: sigmoid of the best log-odds, clamped to ±50.
///
/// The clamp is the reference's overflow guard, not a modelling choice. `clamp` and the
/// reference's `max(-50, min(50, x))` disagree on NaN, but a score is always finite or `-inf`
/// (and the `-inf` case returns before reaching here), so the difference is unreachable.
fn scores_to_confidence(best: f64) -> f64 {
    let clamped = best.clamp(-CONFIDENCE_CLAMP, CONFIDENCE_CLAMP);
    1.0 / (1.0 + (-clamped).exp())
}

/// The `(start, end)` windows a scan visits, in the reference's iteration order: window size
/// outer, start inner. That order is what makes the first-wins tie-break well defined.
fn scan_windows(seq_len: usize) -> Vec<(usize, usize)> {
    let upper = (MAX_WINDOW + 1).min(seq_len + 1);
    let mut out = Vec::new();
    for window_size in (MIN_WINDOW..upper).step_by(WINDOW_STEP) {
        for start in (0..seq_len - window_size + 1).step_by(SCAN_STRIDE) {
            out.push((start, start + window_size));
        }
    }
    out
}

/// The best window found by a scan.
struct Best {
    /// Position in the reference's iteration order; the tie-break key.
    idx: usize,
    score: f64,
    start: usize,
    end: usize,
    scores: Vec<f64>,
}

/// `_scan_sequence`: slide windows of typical MHC lengths and keep the best-scoring one.
///
/// The reference compares with a strict `>`, so the *first* window in iteration order wins a tie.
/// The parallel reduction below reproduces that by breaking ties on the lower `idx`, which makes
/// it associative and the result independent of how rayon schedules the work.
///
/// Encoding once and slicing is safe because encoding is per-residue: `encode(seq[a..b]) ==
/// encode(seq)[a..b]`.
fn scan_sequence(encoded: &[i8], set: &ModelSet) -> Option<Best> {
    scan_windows(encoded.len())
        .into_par_iter()
        .enumerate()
        .map(|(idx, (start, end))| {
            let scores = score_all_models(&encoded[start..end], set);
            Some(Best {
                idx,
                score: max_of(&scores),
                start,
                end,
                scores,
            })
        })
        .reduce(
            || None,
            |a, b| match (a, b) {
                (None, b) => b,
                (a, None) => a,
                (Some(a), Some(b)) => {
                    let a_wins = match a.score.partial_cmp(&b.score) {
                        Some(Ordering::Greater) => true,
                        Some(Ordering::Less) => false,
                        // Equal, or incomparable -- the latter needs a NaN, which a trained
                        // model cannot produce. Either way the reference's strict `>` keeps
                        // whichever window it reached first, so fall back to iteration order.
                        _ => a.idx < b.idx,
                    };
                    Some(if a_wins { a } else { b })
                }
            },
        )
}

/// `MHCClassIClassifier.classify`.
pub fn classify(
    sequence: &str,
    set: &ModelSet,
    n_top: usize,
    scan_constructs: bool,
    threshold: f64,
) -> Classification {
    let cleaned = clean_sequence(sequence);
    if cleaned.is_empty() {
        return empty_result();
    }

    let seq_len = cleaned.len();
    let encoded = encode(&cleaned);

    let (scores, region_start, region_end) = if seq_len <= SCAN_THRESHOLD || !scan_constructs {
        (score_all_models(&encoded, set), 0, seq_len)
    } else {
        match scan_sequence(&encoded, set) {
            Some(best) => (best.scores, best.start, best.end),
            // Unreachable: the branch guarantees seq_len > 370, so at least one window exists.
            // The reference would leave the region at (0, seq_len) here and then raise on an
            // empty score dict; a crash is not an output worth reproducing, so fall back to the
            // degenerate result instead. See PLAN.md §5.4.
            None => return empty_result(),
        }
    };

    let best_score = max_of(&scores);
    let confidence = scores_to_confidence(best_score);
    let probs = normalise_scores(&scores);

    // `sorted(..., key=..., reverse=True)` is stable in Python: ties keep their original order,
    // which is manifest order. Rust's `sort_by` is stable too, so iterating models in manifest
    // order and sorting by probability descending reproduces the reference's ordering exactly.
    let mut ranked: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    ranked.truncate(n_top);

    Classification {
        is_class_i: best_score >= threshold,
        confidence,
        best_score,
        region_start: region_start as u32,
        region_end: region_end as u32,
        top_loci: ranked,
    }
}

/// Log-odds of a sequence against a single model, mirroring `ProfileHMM.log_odds_score`.
///
/// The sequence is cleaned exactly as `classify` cleans it, so the two agree on what a residue
/// is. An empty cleaned sequence still scores: the null model contributes nothing, and the
/// Viterbi score is that of the all-delete path through the model -- a negative number, not zero.
///
/// The caller supplies the `Scratch` so that scoring a whole column reuses one set of buffers
/// per thread rather than allocating per row.
pub fn score_one(sequence: &str, set: &ModelSet, model_idx: usize, scratch: &mut Scratch) -> f64 {
    let cleaned = clean_sequence(sequence);
    let encoded = encode(&cleaned);
    log_odds_score(&encoded, &set.models[model_idx], &set.background, scratch)
}
