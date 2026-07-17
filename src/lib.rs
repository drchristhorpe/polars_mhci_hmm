//! `polars_mhci_hmm._internal` -- the native half of the plugin.
//!
//! Polars discovers the expressions in `expressions.rs` by loading this shared library directly;
//! the `#[pymodule]` below exists so that Python can import the module and hand Polars its path.

mod classify;
mod expressions;
mod hmm;
mod models;
mod npy;

use pyo3::prelude::*;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
