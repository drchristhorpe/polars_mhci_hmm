//! A minimal reader for the `.npy` arrays inside NumPy's `.npz` archives.
//!
//! We vendor the exact model files this reads (`codegen/vendor_models.py` copies them
//! byte-identical from histo_hmm and validates their dtype and shape), so this only needs to
//! handle the narrow slice of the format `np.savez_compressed` actually emits: little-endian
//! `<f8`/`<i8`, C-order, versions 1.0-3.0. Anything else is an error rather than a guess --
//! silently misreading a model would produce plausible, wrong scores.
//!
//! Doing this by hand rather than depending on `ndarray-npy` keeps the dependency list as tight
//! as polars-seq's, and the format is small enough to read in one sitting:
//!
//! ```text
//! \x93NUMPY  <major:u8> <minor:u8> <header_len:u16 or u32 LE> <header: python dict, '\n'-padded>  <raw data>
//! ```

use std::io::Read;

/// One parsed `.npy` array. Only the two dtypes our models use are represented.
pub enum Npy {
    F64 { shape: Vec<usize>, data: Vec<f64> },
    I64 { shape: Vec<usize>, data: Vec<i64> },
}

impl Npy {
    /// The `f64` payload, or an error naming what was found instead.
    pub fn into_f64(self) -> Result<(Vec<usize>, Vec<f64>), String> {
        match self {
            Npy::F64 { shape, data } => Ok((shape, data)),
            Npy::I64 { .. } => Err("expected a float64 array, found int64".into()),
        }
    }

    /// The `i64` payload, or an error naming what was found instead.
    pub fn into_i64(self) -> Result<(Vec<usize>, Vec<i64>), String> {
        match self {
            Npy::I64 { shape, data } => Ok((shape, data)),
            Npy::F64 { .. } => Err("expected an int64 array, found float64".into()),
        }
    }
}

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Parse a `.npy` byte buffer.
pub fn parse(buf: &[u8]) -> Result<Npy, String> {
    if buf.len() < 10 || &buf[..6] != MAGIC {
        return Err("not a .npy file (bad magic)".into());
    }

    let major = buf[6];
    // Version 1.0 sizes the header with a u16; 2.0 and 3.0 widened it to u32. The models are
    // small enough to always be 1.0, but reading 2.0/3.0 costs four lines.
    let (header_len, header_start) = match major {
        1 => (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10),
        2 | 3 => {
            if buf.len() < 12 {
                return Err("truncated .npy header".into());
            }
            (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12)
        }
        v => return Err(format!("unsupported .npy version {v}.x")),
    };

    let data_start = header_start + header_len;
    if buf.len() < data_start {
        return Err("truncated .npy header".into());
    }

    let header = std::str::from_utf8(&buf[header_start..data_start])
        .map_err(|_| "non-UTF-8 .npy header".to_string())?;

    let descr = dict_value(header, "descr")?;
    let fortran = dict_value(header, "fortran_order")?;
    let shape = parse_shape(header)?;

    let n: usize = shape.iter().product();

    // Fortran order only differs from C order for 2-D and above; our model arrays are all
    // C-order, but a 1-D array is trivially both, so accept that case rather than fail on it.
    let is_fortran = fortran.trim() == "True";
    if is_fortran && shape.len() > 1 {
        return Err("Fortran-order .npy arrays are not supported".into());
    }

    let body = &buf[data_start..];
    let dtype = descr.trim().trim_matches('\'').trim_matches('"');

    match dtype {
        "<f8" | "=f8" | "f8" => {
            let data = read_scalars::<8>(body, n, dtype)?
                .into_iter()
                .map(f64::from_le_bytes)
                .collect();
            Ok(Npy::F64 { shape, data })
        }
        "<i8" | "=i8" | "i8" => {
            let data = read_scalars::<8>(body, n, dtype)?
                .into_iter()
                .map(i64::from_le_bytes)
                .collect();
            Ok(Npy::I64 { shape, data })
        }
        other => Err(format!(
            "unsupported .npy dtype '{other}'; expected little-endian float64 or int64"
        )),
    }
}

/// Split `n` fixed-width little-endian scalars out of the data section.
fn read_scalars<const W: usize>(body: &[u8], n: usize, dtype: &str) -> Result<Vec<[u8; W]>, String> {
    let need = n
        .checked_mul(W)
        .ok_or_else(|| "implausible .npy element count".to_string())?;
    if body.len() < need {
        return Err(format!(
            "truncated .npy body: {n} '{dtype}' elements need {need} bytes, found {}",
            body.len()
        ));
    }
    Ok(body[..need]
        .chunks_exact(W)
        .map(|c| c.try_into().expect("chunks_exact yields exactly W bytes"))
        .collect())
}

/// Pull `'<key>': <value>` out of the header dict, up to the next `,` at brace depth 0.
///
/// The header is a Python dict literal rather than JSON, so this scans it directly instead of
/// reaching for a parser. The three keys we want all have simple scalar or tuple values.
fn dict_value<'a>(header: &'a str, key: &str) -> Result<&'a str, String> {
    let pat = format!("'{key}'");
    let start = header
        .find(&pat)
        .ok_or_else(|| format!("no '{key}' in .npy header"))?;
    let rest = header[start + pat.len()..]
        .trim_start()
        .strip_prefix(':')
        .ok_or_else(|| format!("malformed '{key}' entry in .npy header"))?;

    let mut depth = 0usize;
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Ok(rest[..i].trim()),
            '}' if depth == 0 => return Ok(rest[..i].trim()),
            _ => {}
        }
    }
    Ok(rest.trim())
}

/// Parse `'shape': (276, 20)`. NumPy writes a 1-tuple as `(20,)` and a scalar as `()`.
fn parse_shape(header: &str) -> Result<Vec<usize>, String> {
    let raw = dict_value(header, "shape")?;
    let inner = raw
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| format!("malformed shape {raw:?} in .npy header"))?;

    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("non-integer shape component {s:?} in .npy header"))
        })
        .collect()
}

/// Read one named member out of a `.npz` (a zip of `.npy` files) and parse it.
pub fn from_npz<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<Npy, String> {
    let member = format!("{name}.npy");
    let mut file = archive
        .by_name(&member)
        .map_err(|_| format!("no '{member}' in archive"))?;
    let mut buf = Vec::with_capacity(file.size() as usize);
    file.read_to_end(&mut buf)
        .map_err(|e| format!("reading '{member}': {e}"))?;
    parse(&buf).map_err(|e| format!("{member}: {e}"))
}
