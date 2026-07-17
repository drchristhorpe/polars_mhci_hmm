//! The profile-HMM scoring kernel: a port of histo_hmm's `profile_hmm.py`.
//!
//! `_viterbi_core` there is numba-JIT'd, so this is not "compiled beats interpreted" -- the win
//! is that Polars can run it across rows without the GIL, and that a construct scan can spread
//! its ~26,000 Viterbi passes over every core.
//!
//! The recursion is deliberately a literal transcription, including the order of the `max`
//! comparisons. Viterbi is max-plus -- no summation to reorder -- so this reproduces histo_hmm's
//! model scores **exactly**, not approximately. The only place floating-point drift can enter is
//! the null-score summation in `log_odds_score`; see PLAN.md §5.1.

use crate::models::{Model, ALPHABET_SIZE, DD, DM, II, IM, MD, MI, MM};

const NEG_INF: f64 = f64::NEG_INFINITY;

/// histo_hmm's alphabet (`histo_hmm/alphabet.py`), in index order.
pub const AMINO_ACIDS: &[u8; ALPHABET_SIZE] = b"ACDEFGHIKLMNPQRSTVWY";

/// `AA_TO_IDX`, as a byte-indexed table. Anything unrecognised (notably `X`) maps to `-1`,
/// which the kernel reads as "unknown residue" and scores with a uniform emission.
static AA_TO_IDX: [i8; 256] = {
    let mut t = [-1i8; 256];
    let mut i = 0;
    while i < ALPHABET_SIZE {
        t[AMINO_ACIDS[i] as usize] = i as i8;
        i += 1;
    }
    t
};

/// Encode residues to alphabet indices, `-1` for unknown. Mirrors
/// `np.array([AA_TO_IDX.get(c, -1) for c in sequence])`.
///
/// Encoding is per-character and position-independent, so a window of the encoded sequence is
/// the encoding of that window -- which is why a construct scan encodes once and slices.
pub fn encode(seq: &[u8]) -> Vec<i8> {
    seq.iter().map(|&b| AA_TO_IDX[b as usize]).collect()
}

/// Whether a byte is one of the 20 amino acids -- `c in AA_TO_IDX`, via the same table `encode`
/// uses, so the alphabet has exactly one definition.
#[inline]
pub fn encodes(b: u8) -> bool {
    AA_TO_IDX[b as usize] >= 0
}

/// Reusable row buffers for the Viterbi recursion.
///
/// One sequence is scored against 251 models, and a scan repeats that per window; allocating six
/// vectors per pass would cost more than the arithmetic. A `Scratch` is grown once per thread and
/// reused.
pub struct Scratch {
    prev_m: Vec<f64>,
    prev_i: Vec<f64>,
    prev_d: Vec<f64>,
    curr_m: Vec<f64>,
    curr_i: Vec<f64>,
    curr_d: Vec<f64>,
    /// A row of `log(1/20)`, used in place of a model's emission row for an unknown residue.
    /// Holding it here lets the inner loops read one uniform `&[f64]` either way, instead of
    /// branching on "is this residue known" once per model position.
    uniform: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Self {
        Scratch {
            prev_m: Vec::new(),
            prev_i: Vec::new(),
            prev_d: Vec::new(),
            curr_m: Vec::new(),
            curr_i: Vec::new(),
            curr_d: Vec::new(),
            uniform: Vec::new(),
        }
    }

    fn reset(&mut self, n: usize) {
        for buf in [
            &mut self.prev_m,
            &mut self.prev_i,
            &mut self.prev_d,
            &mut self.curr_m,
            &mut self.curr_i,
            &mut self.curr_d,
        ] {
            buf.clear();
            buf.resize(n, NEG_INF);
        }
        self.uniform.clear();
        self.uniform.resize(n, (1.0 / ALPHABET_SIZE as f64).ln());
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Viterbi log-score of an encoded sequence against one model.
///
/// A transcription of `_viterbi_core`. `-inf` propagates through the additions exactly as it does
/// in numpy (`-inf + finite = -inf`); no `+inf` exists in a trained model, so no addition can
/// produce a NaN, and the comparisons stay total.
pub fn viterbi(encoded: &[i8], model: &Model, scratch: &mut Scratch) -> f64 {
    let l = model.l;
    let n = l + 1;

    scratch.reset(n);
    let Scratch {
        prev_m,
        prev_i,
        prev_d,
        curr_m,
        curr_i,
        curr_d,
        uniform,
    } = scratch;

    // Hoisted out of the loops: each is a contiguous run over model positions.
    let t_mm = model.trans_row(MM);
    let t_mi = model.trans_row(MI);
    let t_md = model.trans_row(MD);
    let t_im = model.trans_row(IM);
    let t_ii = model.trans_row(II);
    let t_dm = model.trans_row(DM);
    let t_dd = model.trans_row(DD);

    prev_m[0] = 0.0;

    // The delete chain at i=0: reachable before any residue is consumed.
    for j in 1..=l {
        let from_m = prev_m[j - 1] + t_md[j - 1];
        let from_d = prev_d[j - 1] + t_dd[j - 1];
        prev_d[j] = if from_m > from_d { from_m } else { from_d };
    }

    for &aa in encoded {
        // An unknown residue (`X`, encoded -1) emits uniformly at every position, so swapping in
        // a uniform row keeps the branch out of the inner loops.
        let (me, ie) = if aa >= 0 {
            let aa = aa as usize;
            (model.match_emit_row(aa), model.insert_emit_row(aa))
        } else {
            (&uniform[..], &uniform[..])
        };

        // M_j and I_j depend only on the *previous* row, so both loops are free of a
        // loop-carried dependency and vectorise. Slicing every operand to one common length
        // first is what lets LLVM prove that and drop the bounds checks: `x[i]` on a slice of
        // known length `l` with `i < l` needs no check. Without this the loops stay scalar.
        curr_m[0] = NEG_INF;
        {
            let cm = &mut curr_m[1..n];
            let (pm, pi, pd) = (&prev_m[..l], &prev_i[..l], &prev_d[..l]);
            let (tmm, tim, tdm) = (&t_mm[..l], &t_im[..l], &t_dm[..l]);
            let me = &me[1..n];

            for i in 0..l {
                let v1 = pm[i] + tmm[i];
                let v2 = pi[i] + tim[i];
                let v3 = pd[i] + tdm[i];
                let mut best = v1;
                if v2 > best {
                    best = v2;
                }
                if v3 > best {
                    best = v3;
                }
                cm[i] = best + me[i];
            }
        }

        {
            let ci = &mut curr_i[..n];
            let (pm, pi) = (&prev_m[..n], &prev_i[..n]);
            let (tmi, tii) = (&t_mi[..n], &t_ii[..n]);
            let ie = &ie[..n];

            for j in 0..n {
                let v1 = pm[j] + tmi[j];
                let v2 = pi[j] + tii[j];
                ci[j] = (if v1 > v2 { v1 } else { v2 }) + ie[j];
            }
        }

        // Deletes depend on this row's matches, so they run after and stay sequential.
        curr_d[0] = NEG_INF;
        for j in 1..=l {
            let from_m = curr_m[j - 1] + t_md[j - 1];
            let from_d = curr_d[j - 1] + t_dd[j - 1];
            curr_d[j] = if from_m > from_d { from_m } else { from_d };
        }

        // The reference copies curr into prev; swapping the buffers is the same thing without
        // the memcpy, because every element of each curr row is written before it is read again.
        std::mem::swap(prev_m, curr_m);
        std::mem::swap(prev_i, curr_i);
        std::mem::swap(prev_d, curr_d);
    }

    let mut result = prev_m[l];
    if prev_i[l] > result {
        result = prev_i[l];
    }
    if prev_d[l] > result {
        result = prev_d[l];
    }
    result
}

/// `log P(seq|model) - log P(seq|null)`, mirroring `ProfileHMM.log_odds_score`.
///
/// The null model scores known residues with the background distribution and unknown ones
/// uniformly. histo_hmm sums the known terms with `np.sum` (pairwise) and only then adds the
/// unknown term; the order here matches, but the summation is sequential -- the one bounded
/// source of drift against the reference (PLAN.md §5.1).
pub fn log_odds_score(
    encoded: &[i8],
    model: &Model,
    background: &[f64],
    scratch: &mut Scratch,
) -> f64 {
    let model_score = viterbi(encoded, model, scratch);

    let mut null_score = 0.0;
    let mut n_unknown = 0usize;
    for &e in encoded {
        if e >= 0 {
            null_score += background[e as usize];
        } else {
            n_unknown += 1;
        }
    }
    if n_unknown > 0 {
        null_score += n_unknown as f64 * (1.0 / ALPHABET_SIZE as f64).ln();
    }

    model_score - null_score
}
