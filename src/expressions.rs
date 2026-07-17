//! Polars expression entry points.
//!
//! Arguments arrive already validated by the Python layer (`python/polars_mhci_hmm/__init__.py`),
//! so an unknown locus or a missing model directory raises a clean `ValueError` at expression
//! construction time rather than a `ComputeError` from deep inside the query engine.
//!
//! Rows are walked sequentially here on purpose: the parallelism lives one level down, across the
//! 251 models and (for constructs) the scan windows. That keeps every core busy even on the
//! one-row frame an interactive session starts with, which row-level parallelism alone would
//! leave single-threaded.

use std::path::Path;

use polars::prelude::*;
use polars_arrow::bitmap::Bitmap;
use pyo3_polars::derive::polars_expr;
use rayon::prelude::*;
use serde::Deserialize;

use crate::classify::{classify, score_one, Classification};
use crate::hmm::Scratch;
use crate::models::{self, ModelSet};

#[derive(Deserialize)]
pub struct ClassifyKwargs {
    n_top: usize,
    scan_constructs: bool,
    threshold: f64,
    /// Resolved to a concrete directory by Python, which also checks it exists.
    model_dir: String,
}

#[derive(Deserialize)]
pub struct ScoreKwargs {
    locus: String,
    model_dir: String,
}

fn load(model_dir: &str) -> PolarsResult<std::sync::Arc<ModelSet>> {
    models::load(Path::new(model_dir)).map_err(|e| polars_err!(ComputeError: "{e}"))
}

fn top_locus_dtype() -> DataType {
    DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("locus"), DataType::String),
        Field::new(PlSmallStr::from_static("probability"), DataType::Float64),
    ])
}

/// The struct `classify` returns: `ClassificationResult`, field for field.
fn classify_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let dt = DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("is_class_i"), DataType::Boolean),
        Field::new(PlSmallStr::from_static("confidence"), DataType::Float64),
        Field::new(PlSmallStr::from_static("best_score"), DataType::Float64),
        Field::new(PlSmallStr::from_static("region_start"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("region_end"), DataType::UInt32),
        Field::new(
            PlSmallStr::from_static("top_loci"),
            DataType::List(Box::new(top_locus_dtype())),
        ),
    ]);
    Ok(Field::new(input_fields[0].name().clone(), dt))
}

#[polars_expr(output_type_func=classify_output)]
fn classify_expr(inputs: &[Series], kwargs: ClassifyKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let set = load(&kwargs.model_dir)?;
    let n = ca.len();

    let results: Vec<Option<Classification>> = ca
        .iter()
        .map(|opt| {
            opt.map(|s| {
                classify(
                    s,
                    &set,
                    kwargs.n_top,
                    kwargs.scan_constructs,
                    kwargs.threshold,
                )
            })
        })
        .collect();

    let is_class_i = BooleanChunked::from_iter_options(
        PlSmallStr::from_static("is_class_i"),
        results.iter().map(|r| r.as_ref().map(|c| c.is_class_i)),
    );
    let confidence = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("confidence"),
        results.iter().map(|r| r.as_ref().map(|c| c.confidence)),
    );
    let best_score = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("best_score"),
        results.iter().map(|r| r.as_ref().map(|c| c.best_score)),
    );
    let region_start = UInt32Chunked::from_iter_options(
        PlSmallStr::from_static("region_start"),
        results.iter().map(|r| r.as_ref().map(|c| c.region_start)),
    );
    let region_end = UInt32Chunked::from_iter_options(
        PlSmallStr::from_static("region_end"),
        results.iter().map(|r| r.as_ref().map(|c| c.region_end)),
    );

    let top_loci = build_top_loci(&results, &set)?;

    let fields = vec![
        is_class_i.into_series(),
        confidence.into_series(),
        best_score.into_series(),
        region_start.into_series(),
        region_end.into_series(),
        top_loci,
    ];

    let out = StructChunked::from_series(ca.name().clone(), n, fields.iter())?;

    // A null sequence yields a null struct rather than a struct of nulls: the absence of a
    // sequence is not a classification that happens to say nothing.
    let validity: Bitmap = results.iter().map(|r| r.is_some()).collect();
    let out = out.with_outer_validity(Some(validity));

    Ok(out.into_series())
}

/// Assemble the `List(Struct{locus, probability})` column.
///
/// Built row by row: `top_loci` holds at most `n_top` entries, and the per-row cost is
/// microseconds against the milliseconds each row spends in Viterbi.
fn build_top_loci(results: &[Option<Classification>], set: &ModelSet) -> PolarsResult<Series> {
    // `Option` is the null/non-null axis; a construction failure is an error and must not be
    // collapsed into a null, which would present a failed classification as a valid empty one.
    let rows: Vec<Option<Series>> = results
        .iter()
        .map(|result| {
            let Some(c) = result else { return Ok(None) };

            let loci = StringChunked::from_iter_values(
                PlSmallStr::from_static("locus"),
                c.top_loci.iter().map(|(i, _)| set.models[*i].name.as_str()),
            );
            let probs = Float64Chunked::from_iter_values(
                PlSmallStr::from_static("probability"),
                c.top_loci.iter().map(|(_, p)| *p),
            );
            let fields = [loci.into_series(), probs.into_series()];
            let row = StructChunked::from_series(
                PlSmallStr::from_static("top_loci"),
                c.top_loci.len(),
                fields.iter(),
            )?;
            Ok(Some(row.into_series()))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    // `collect` names the result "", so the field has to be named back -- an unnamed series
    // becomes an unnamed struct field, and `.struct.field("top_loci")` would not find it.
    let ca: ListChunked = rows.into_iter().collect();
    let s = ca
        .into_series()
        .with_name(PlSmallStr::from_static("top_loci"));

    // An all-null or all-empty column carries no inner dtype for polars to infer, which would
    // contradict the schema `classify_output` promised. Casting pins it; it is a no-op otherwise.
    s.cast(&DataType::List(Box::new(top_locus_dtype())))
}

#[polars_expr(output_type=Float64)]
fn score_expr(inputs: &[Series], kwargs: ScoreKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let set = load(&kwargs.model_dir)?;

    // Python validates the locus against the manifest, so this only fires if a caller reaches
    // past it or swaps the model directory underneath the expression.
    let idx = set.index_of(&kwargs.locus).ok_or_else(
        || polars_err!(ComputeError: "unknown locus '{}' in {}", kwargs.locus, kwargs.model_dir),
    )?;

    // Unlike `classify`, this scores a single model, so there is no per-row fan-out over the 251
    // of them to ride on -- without parallelising the rows here, the whole column would run on
    // one core. `map_init` gives each rayon thread its own scratch buffers, and `collect` into a
    // Vec preserves row order.
    let scored: Vec<Option<f64>> = ca
        .par_iter()
        .map_init(Scratch::new, |scratch, opt| {
            opt.map(|s| score_one(s, &set, idx, scratch))
        })
        .collect();

    Ok(Float64Chunked::from_iter_options(ca.name().clone(), scored.into_iter()).into_series())
}
