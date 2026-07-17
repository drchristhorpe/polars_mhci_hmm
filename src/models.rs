//! Loading the trained profile HMMs, and keeping them loaded.
//!
//! A model directory is exactly what histo_hmm's `save_models` writes: one `<class>.npz` per
//! locus, a `background.npy`, and a `manifest.json`. We vendor a copy (see
//! `codegen/vendor_models.py`), and `model_dir=` lets a caller point at their own.
//!
//! Two things here are load-bearing:
//!
//! * **Manifest order.** `classes` is the order histo_hmm's `models` dict iterates in, which is
//!   the order Python's stable sort falls back on when two loci score identically. Preserve it
//!   exactly; never re-sort. See PLAN.md §5.2.
//! * **The cache.** Inflating 251 archives yields ~26 MB of f64 and takes tens of milliseconds.
//!   Per row that would dwarf the actual work, so a `ModelSet` is loaded once per directory per
//!   process and shared.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use serde::Deserialize;

use crate::npy;

/// The amino-acid alphabet histo_hmm trains against (`histo_hmm/alphabet.py`).
pub const ALPHABET_SIZE: usize = 20;
/// Transition classes per model position: MM, MI, MD, IM, II, DM, DD.
pub const N_TRANS: usize = 7;

// Transition indices, matching histo_hmm/profile_hmm.py.
pub const MM: usize = 0;
pub const MI: usize = 1;
pub const MD: usize = 2;
pub const IM: usize = 3;
pub const II: usize = 4;
pub const DM: usize = 5;
pub const DD: usize = 6;

/// One trained profile HMM, stored **transposed** relative to numpy.
///
/// numpy stores these as `(L+1) x 20` (and `(L+1) x 7`), so a Viterbi row -- which sweeps `j`
/// with the residue fixed -- would stride by 20 f64 and touch a fresh cache line every step.
/// Transposing to `20 x (L+1)` makes each sweep a contiguous run, which is the difference
/// between reading ~275 cache lines per row and ~35. `read_model` transposes on load; nothing
/// downstream sees numpy's layout.
pub struct Model {
    pub name: String,
    /// Number of match states. Varies across the bundled models (273..=282).
    pub l: usize,
    /// `20 x (L+1)`. Column 0 is unused and is all `-inf`; match states are 1..=L.
    match_emit_t: Vec<f64>,
    /// `20 x (L+1)`.
    insert_emit_t: Vec<f64>,
    /// `7 x (L+1)`, indexed by the `MM`..`DD` constants.
    trans_t: Vec<f64>,
}

impl Model {
    /// Match emissions for one residue across every model position: `me[j]`, contiguous.
    #[inline(always)]
    pub fn match_emit_row(&self, aa: usize) -> &[f64] {
        let n = self.l + 1;
        &self.match_emit_t[aa * n..(aa + 1) * n]
    }

    /// Insert emissions for one residue across every model position: `ie[j]`, contiguous.
    #[inline(always)]
    pub fn insert_emit_row(&self, aa: usize) -> &[f64] {
        let n = self.l + 1;
        &self.insert_emit_t[aa * n..(aa + 1) * n]
    }

    /// One transition class across every model position: `t[j]`, contiguous.
    #[inline(always)]
    pub fn trans_row(&self, t: usize) -> &[f64] {
        let n = self.l + 1;
        &self.trans_t[t * n..(t + 1) * n]
    }
}

/// Transpose a row-major `rows x cols` matrix into `cols x rows`.
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Every model in a directory, plus the shared background distribution.
pub struct ModelSet {
    /// In manifest order. This order is part of the output contract.
    pub models: Vec<Model>,
    /// Background log-frequencies, length 20. histo_hmm's null model.
    pub background: Vec<f64>,
}

impl ModelSet {
    /// Index of a locus by name, or `None`.
    pub fn index_of(&self, locus: &str) -> Option<usize> {
        self.models.iter().position(|m| m.name == locus)
    }
}

#[derive(Deserialize)]
struct Manifest {
    classes: Vec<String>,
}

fn read_manifest(dir: &Path) -> Result<Manifest, String> {
    let path = dir.join("manifest.json");
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parsing {}: {e}", path.display()))
}

fn read_background(dir: &Path) -> Result<Vec<f64>, String> {
    let path = dir.join("background.npy");
    let buf = std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let (shape, data) = npy::parse(&buf)
        .and_then(|n| n.into_f64())
        .map_err(|e| format!("{}: {e}", path.display()))?;

    if shape != [ALPHABET_SIZE] {
        return Err(format!(
            "{}: expected shape [{ALPHABET_SIZE}], found {shape:?}",
            path.display()
        ));
    }
    Ok(data)
}

fn read_model(dir: &Path, name: &str) -> Result<Model, String> {
    let path = dir.join(format!("{name}.npz"));
    let file =
        std::fs::File::open(&path).map_err(|e| format!("opening {}: {e}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("{} is not a .npz: {e}", path.display()))?;

    let ctx = |e: String| format!("{}: {e}", path.display());

    let (len_shape, len_data) = npy::from_npz(&mut archive, "length")
        .and_then(|n| n.into_i64())
        .map_err(ctx)?;
    if len_data.len() != 1 {
        return Err(format!(
            "{}: 'length' should hold one value, found {len_shape:?}",
            path.display()
        ));
    }
    let l_signed = len_data[0];
    if l_signed < 1 {
        return Err(format!(
            "{}: model length {l_signed} is not positive",
            path.display()
        ));
    }
    let l = l_signed as usize;

    let mut get = |member: &str, cols: usize| -> Result<Vec<f64>, String> {
        let (shape, data) = npy::from_npz(&mut archive, member)
            .and_then(|n| n.into_f64())
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if shape != [l + 1, cols] {
            return Err(format!(
                "{}: '{member}' has shape {shape:?}, expected [{}, {cols}] for a length-{l} model",
                path.display(),
                l + 1
            ));
        }
        Ok(data)
    };

    let match_emit = get("match_emit", ALPHABET_SIZE)?;
    let insert_emit = get("insert_emit", ALPHABET_SIZE)?;
    let trans = get("trans", N_TRANS)?;

    // A NaN anywhere would propagate through the max-plus recursion and poison every
    // comparison downstream, silently. -inf is expected and handled; NaN is not.
    for (member, arr) in [
        ("match_emit", &match_emit),
        ("insert_emit", &insert_emit),
        ("trans", &trans),
    ] {
        if arr.iter().any(|v| v.is_nan()) {
            return Err(format!("{}: '{member}' contains NaN", path.display()));
        }
    }

    Ok(Model {
        name: name.to_string(),
        l,
        match_emit_t: transpose(&match_emit, l + 1, ALPHABET_SIZE),
        insert_emit_t: transpose(&insert_emit, l + 1, ALPHABET_SIZE),
        trans_t: transpose(&trans, l + 1, N_TRANS),
    })
}

fn load_uncached(dir: &Path) -> Result<ModelSet, String> {
    let manifest = read_manifest(dir)?;
    if manifest.classes.is_empty() {
        return Err(format!("{}: manifest lists no classes", dir.display()));
    }
    let background = read_background(dir)?;

    // Inflating 251 archives is the one slow part of startup; rayon's `collect` into a Vec
    // preserves input order, so manifest order survives the parallelism.
    let models = manifest
        .classes
        .par_iter()
        .map(|name| read_model(dir, name))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ModelSet { models, background })
}

type Cache = Mutex<HashMap<PathBuf, Arc<ModelSet>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a model directory, or return the already-loaded copy.
///
/// Keyed on the canonical path, so `models/` and `models/../models/` share one entry.
pub fn load(dir: &Path) -> Result<Arc<ModelSet>, String> {
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());

    if let Some(hit) = cache()
        .lock()
        .expect("model cache mutex poisoned")
        .get(&key)
    {
        return Ok(Arc::clone(hit));
    }

    // Loading outside the lock: two threads racing the same cold directory duplicate the work
    // once, which is cheaper than making every reader wait behind whoever got there first.
    let set = Arc::new(load_uncached(dir)?);

    let mut guard = cache().lock().expect("model cache mutex poisoned");
    Ok(Arc::clone(guard.entry(key).or_insert(set)))
}
