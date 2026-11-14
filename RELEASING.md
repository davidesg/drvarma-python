# Releasing drvarma to PyPI

Publishing is done by `.github/workflows/publish.yml` when a `v*` tag is pushed.
Nothing is uploaded by hand: the wheels with the compiled engine only come out
of that workflow's matrix, and a manual upload leaves users without them.

## What gets published

* **Binary wheels**, cp310–cp314, with the GSL bundled (cibuildwheel):
  Linux x86_64 and aarch64, macOS arm64 and x86_64 (`macosx_11_0`, built on
  macos-15 and macos-15-intel), Windows AMD64.
  `tools/check_macos_minos.py` stops the build if a macOS binary asks for a
  newer system than its tag.
* **Pure-Python wheel** `py3-none-any` (built with `DRVARMA_NO_ENGINE=1`), the
  fallback for any other platform.
* **sdist**, which ships `csrc/` and builds the engine where GSL is present.

All three carry the bug register (`tools/sync_material.py` copies it into
`src/drvarma/material/`; sima serves it as `sima://engine-defects`). The
workflow runs that step itself.

Upload is by PyPI **Trusted Publishing** (OIDC, environment `pypi`). The
publisher is registered against the file name `publish.yml`: renaming the
workflow breaks the upload.

## Steps

### 1. Release PR

On a branch `release-drvarma-X.Y.Z`:

* `pyproject.toml` → `[project] version = "X.Y.Z"`. It is the only place:
  `drvarma.__version__` is read from the installed metadata.
* `CHANGELOG.md` → a section `## X.Y.Z — unreleased` saying what changes for
  the user.

### 2. Validate without publishing

Run the workflow by hand on the branch. It builds and smoke-tests everything;
the publish job only runs on a tag, so it is skipped:

```sh
gh workflow run publish.yml -R davidesg/drvarma-python --ref release-drvarma-X.Y.Z
```

When drvarma is released together with other ATSW packages, also run
art-python's `preflight-atsw.yml` with `origen=repos` and the release branches.

### 3. The date, in the same PR

When the release is approved, a last commit on the PR replaces `unreleased`
with the date (`## X.Y.Z — YYYY-MM-DD`). Then merge, and wait for CI on master.

### 4. Tag the merge commit

A lightweight tag `vX.Y.Z` on the merge commit of the release PR:

```sh
gh api repos/davidesg/drvarma-python/git/refs \
  -f ref=refs/tags/vX.Y.Z -f sha=<merge commit sha>
```

The tag starts `publish.yml`, which builds again and publishes.

### 5. Check

* If the publish job fails before uploading with a Sigstore/Rekor error (502),
  wait about ten minutes and rerun it: `gh run rerun <id> --failed`.
  `skip-existing: true` makes the rerun safe if part was already uploaded.
* Check PyPI has every file: `https://pypi.org/project/drvarma/X.Y.Z/#files`.
* A clean install picks the binary wheel:
  ```sh
  python3 -m venv /tmp/t && /tmp/t/bin/pip install drvarma==X.Y.Z
  /tmp/t/bin/python -c "import drvarma; print(drvarma.__version__)"
  ```

## Releasing together with the rest of ATSW

Packages are tagged in dependency order, each one only once its dependencies
are on PyPI, because the smoke tests install from PyPI:
fue → drvarma → drvec → sima → art / pyfug / drtran → atsw.

After the last tag, run `preflight-atsw.yml` with `origen=pypi` and the suites.
For a few minutes after a publish, some runners do not yet see the new version
on the index: rerun the failed jobs.

## Notes

* **Project URLs**: `[project.urls]` points at this repo,
  `github.com/davidesg/drvarma-python` (the C engine lives in
  `davidesg/drvarma`).
* **Optional extras**: `drvarma[c-engine]` (GSL build), `drvarma[plots]`
  (matplotlib + pyfug), `drvarma[forecast-report]` (jinja2), `drvarma[test]`.
* **Local build**, only to inspect the artifacts (never to upload):
  `python3 tools/sync_material.py && DRVARMA_NO_ENGINE=1 python -m build`,
  then `twine check dist/*` (twine ≥ 6.1 for the Metadata 2.4 fields).
