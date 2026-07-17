"""polars-mhci-hmm -- classify MHC Class I sequences inside Polars.

Importing this module registers a ``.mhci`` namespace on Polars expressions::

    import polars as pl
    import polars_mhci_hmm  # noqa: F401  (import for the side effect)

    df.with_columns(result=pl.col("sequence").mhci.classify())

The semantics are those of ``histo_hmm``'s ``MHCClassIClassifier``, reproduced field-for-field
against the same 251 trained profile HMMs -- including its sequence cleaning, its sliding-window
scan for constructs, and its tie-breaking. See the README for the one bounded difference
(floating-point summation order) and for what ``region_start``/``region_end`` index.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = ["MHCINameSpace", "classify", "loci", "model_dir", "score", "__version__"]
__version__ = "0.1.0"

_PLUGIN_PATH = Path(__file__).parent
_MODEL_DIR = _PLUGIN_PATH / "models"


def model_dir() -> Path:
    """Path to the bundled models, vendored from ``histo_hmm``.

    Pass it to ``classify(model_dir=...)`` explicitly, or point that argument at your own
    directory of ``.npz`` models -- the format is ``histo_hmm``'s, unchanged.
    """
    return _MODEL_DIR


@lru_cache(maxsize=None)
def _manifest(directory: str) -> dict:
    path = Path(directory) / "manifest.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(
            f"no manifest.json in {directory!r}; a model directory holds one .npz per locus, "
            "a background.npy and a manifest.json (as written by histo_hmm's save_models)"
        ) from None
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed manifest.json in {directory!r}: {e}") from None


def _resolve_model_dir(directory: str | Path | None) -> str:
    """Resolve to a concrete directory, failing here rather than inside the query engine."""
    if directory is None:
        if not (_MODEL_DIR / "manifest.json").exists():
            raise RuntimeError(
                f"the bundled models are missing from {_MODEL_DIR}. If you are working from a "
                "source checkout, run:\n  uv run python codegen/vendor_models.py"
            )
        return str(_MODEL_DIR)

    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"model_dir {str(directory)!r} is not a directory")
    _manifest(str(path))  # surfaces a missing or malformed manifest now
    return str(path)


def loci(model_dir: str | Path | None = None) -> pl.DataFrame:
    """Return the MHC Class I loci this build can classify, as a DataFrame.

    Columns are ``locus`` (the name used in ``top_loci`` and accepted by :func:`score`) and
    ``length`` (the profile HMM's match-state count).
    """
    directory = _resolve_model_dir(model_dir)
    manifest = _manifest(directory)
    classes = manifest["classes"]
    lengths = manifest.get("lengths", {})
    return pl.DataFrame(
        {
            "locus": classes,
            "length": [lengths.get(c) for c in classes],
        },
        schema={"locus": pl.String, "length": pl.UInt32},
    )


def _validate_n_top(n_top: int) -> int:
    if isinstance(n_top, bool) or not isinstance(n_top, int):
        raise TypeError(f"n_top must be an int, got {type(n_top).__name__}")
    if n_top < 0:
        raise ValueError(f"n_top must be >= 0, got {n_top}")
    return n_top


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
    return float(threshold)


def _validate_locus(locus: str, directory: str) -> str:
    if not isinstance(locus, str):
        raise TypeError(f"locus must be a str, got {type(locus).__name__}")
    classes = _manifest(directory)["classes"]
    if locus not in classes:
        # Suggest near-misses before dumping 251 names at the user.
        import difflib

        close = difflib.get_close_matches(locus, classes, n=3)
        hint = f"; did you mean {', '.join(repr(c) for c in close)}?" if close else ""
        raise ValueError(
            f"unknown locus {locus!r}{hint} -- call polars_mhci_hmm.loci() to list the "
            f"{len(classes)} available loci"
        )
    return locus


@pl.api.register_expr_namespace("mhci")
class MHCINameSpace:
    """The ``.mhci`` expression namespace."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def classify(
        self,
        n_top: int = 10,
        scan_constructs: bool = True,
        threshold: float = 0.0,
        model_dir: str | Path | None = None,
    ) -> pl.Expr:
        """Classify a protein-sequence column as MHC Class I and predict its locus.

        Returns a struct column mirroring ``histo_hmm``'s ``ClassificationResult``:

        =============== ============================================ =====================
        field           dtype                                        meaning
        =============== ============================================ =====================
        ``is_class_i``  ``Boolean``                                  ``best_score >= threshold``
        ``confidence``  ``Float64``                                  sigmoid of the best log-odds
        ``best_score``  ``Float64``                                  raw log-odds of the top locus
        ``region_start````UInt32``                                   start of the detected region
        ``region_end``  ``UInt32``                                   end of the region (exclusive)
        ``top_loci``    ``List(Struct{locus, probability})``         best ``n_top``, best first
        =============== ============================================ =====================

        Arguments match ``MHCClassIClassifier``:

        n_top
            Number of top loci to return. Probabilities are temperature-scaled softmax values
            over *all* loci, so they sum to 1.0 across the full set, not across ``top_loci``.
        scan_constructs
            If True, a sequence longer than ~370 residues is scanned with a sliding window to
            find the best-matching region, rather than scored whole. This is what handles MHC
            embedded in a fusion construct -- and it is expensive: it scores ~104 windows
            against every locus. Set to False to always score the full sequence.
        threshold
            Log-odds threshold for calling a sequence Class I. The default of 0.0 means "more
            likely under some Class I model than under the background".
        model_dir
            A directory of models in ``histo_hmm``'s format. Defaults to the bundled ones.

        ``region_start``/``region_end`` index the **cleaned** sequence -- uppercased and stripped
        of everything that is not one of the 20 amino acids or ``X`` -- not the input string. If
        your input contains gaps or punctuation, the offsets will not line up with it.

        Null inputs produce null outputs. A sequence with no scorable residues produces the
        reference's degenerate result: ``is_class_i=False``, ``confidence=0.0``, ``top_loci=[]``,
        ``best_score=-inf``, ``region=[0, 0]``.
        """
        directory = _resolve_model_dir(model_dir)
        n_top = _validate_n_top(n_top)
        threshold = _validate_threshold(threshold)
        if not isinstance(scan_constructs, bool):
            raise TypeError(
                f"scan_constructs must be a bool, got {type(scan_constructs).__name__}"
            )

        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="classify_expr",
            args=self._expr,
            kwargs={
                "n_top": n_top,
                "scan_constructs": scan_constructs,
                "threshold": threshold,
                "model_dir": directory,
            },
            is_elementwise=True,
        )

    def score(
        self,
        locus: str,
        model_dir: str | Path | None = None,
    ) -> pl.Expr:
        """Log-odds of each sequence against a single locus' profile HMM.

        Mirrors ``ProfileHMM.log_odds_score``: ``log P(seq|model) - log P(seq|null)``. Positive
        means the sequence is more likely under this locus' model than under the background.

        The sequence is cleaned exactly as :meth:`classify` cleans it, but there is no
        sliding-window scan -- the whole (cleaned) sequence is scored against the one model.

        Null inputs produce null outputs.
        """
        directory = _resolve_model_dir(model_dir)
        locus = _validate_locus(locus, directory)

        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="score_expr",
            args=self._expr,
            kwargs={"locus": locus, "model_dir": directory},
            is_elementwise=True,
        )


def classify(expr: IntoExpr, /, **kwargs) -> pl.Expr:
    """Function form of :meth:`MHCINameSpace.classify`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.mhci.classify(**kwargs)


def score(expr: IntoExpr, /, locus: str, **kwargs) -> pl.Expr:
    """Function form of :meth:`MHCINameSpace.score`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.mhci.score(locus, **kwargs)
