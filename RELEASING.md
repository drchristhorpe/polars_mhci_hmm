# Releasing `polars-mhci-hmm` to PyPI

Publishing is automated: **push a `v*` tag and GitHub Actions builds every wheel, smoke-tests
one, and publishes to PyPI.** There is no API token to create, store, or rotate — it uses PyPI's
Trusted Publishing (OIDC).

The one-time setup below only has to be done once, before the first release.

---

## One-time setup (do this before v0.1.0)

### 1. Register the Trusted Publisher on PyPI

The project does not exist on PyPI yet, so create a **pending** publisher — PyPI will create the
project automatically on first upload.

1. Go to <https://pypi.org/manage/account/publishing/>
2. Under **"Add a new pending publisher"**, fill in exactly:

   | field | value |
   |---|---|
   | PyPI Project Name | `polars-mhci-hmm` |
   | Owner | `drchristhorpe` |
   | Repository name | `polars_mhci_hmm` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. Save.

> The **Environment name must be `pypi`** — it has to match `environment: name: pypi` in
> `.github/workflows/release.yml`, or PyPI will reject the upload.

> Note the repository name is `polars_mhci_hmm` (underscores) while the PyPI project is
> `polars-mhci-hmm` (hyphens). That is fine — they are different namespaces — but the values
> above must match each side exactly.

The name `polars-mhci-hmm` was available at the time of writing. If someone has taken it since,
change `name` in `pyproject.toml` and the values above together.

### 2. Create the `pypi` environment on GitHub

1. Go to <https://github.com/drchristhorpe/polars_mhci_hmm/settings/environments>
2. **New environment** → name it `pypi` → Configure.
3. Optionally add yourself as a required reviewer, which makes every publish need a manual
   click. Worth it: the tag becomes a proposal rather than a trigger.

Nothing else is needed — no secrets.

---

## Releasing

1. **Update the version in both places.** They must agree, and they must agree with the tag; the
   `check-version` job fails the release in ten seconds if they do not.

   ```toml
   # pyproject.toml
   version = "0.1.0"

   # Cargo.toml
   version = "0.1.0"
   ```

2. **Update `CHANGELOG.md`** — move `[Unreleased]` to the new version with today's date.

3. **Check the suite is green**, including parity against the real histo_hmm:

   ```bash
   uv sync
   uv run pytest -q
   uv run python tools/validate.py   # writes tmp/00_SUMMARY.txt; expect 45/45
   ```

4. **Commit, tag, push:**

   ```bash
   git commit -am "Release 0.1.0"
   git tag v0.1.0
   git push && git push origin v0.1.0
   ```

5. Watch <https://github.com/drchristhorpe/polars_mhci_hmm/actions>. The workflow will:
   - check the tag matches `pyproject.toml` and `Cargo.toml`;
   - build wheels for Linux (x86_64, aarch64), macOS (arm64, x86_64) and Windows (x64) — one
     each, because the extension is abi3 for Python ≥ 3.10;
   - build an sdist, and verify all **251 models** are inside it;
   - install the Linux wheel on a Python it was *not* built against and classify every test
     sequence, asserting against the checked-in golden values;
   - publish everything to PyPI.

6. **Verify the published package** in a clean environment:

   ```bash
   uv run --no-project --with polars-mhci-hmm --python 3.11 python -c "
   import polars as pl, polars_mhci_hmm
   print(len(polars_mhci_hmm.loci()), 'loci')
   "
   ```

---

## Notes

- **The wheel carries the models.** They are ~0.5 MB of `.npz` pulled in by an explicit `include`
  in `pyproject.toml`, not by being importable Python — which is why both the sdist job and the
  smoke test count them rather than trusting the packaging.
- **PyPI is append-only.** A version cannot be edited, re-uploaded, or replaced; the only
  recourse is a new version number. This includes the README — PyPI freezes the description into
  the uploaded distribution at release time and never re-reads the repository, so a README fix
  needs a release to become visible on the PyPI page.
- **`workflow_dispatch` builds but does not publish.** The `publish` job is gated on
  `startsWith(github.ref, 'refs/tags/v')`, so a manual run is a safe way to exercise the wheel
  matrix without releasing anything.
- **Re-running a failed publish is safe.** Trusted Publishing is idempotent per file: if some
  wheels uploaded and the job then failed, re-running skips what is already there.
