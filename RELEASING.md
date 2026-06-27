# Releasing drvarma to PyPI

The package is configured for a clean release: `pyproject.toml` carries the full
metadata (SPDX `GPL-2.0-or-later`, classifiers, URLs, extras) and `setup.py`
builds the C engine as an **optional** accelerator, so the published artifacts are
pure-Python and install anywhere.

Two artifacts are published:

* **sdist** `drvarma-<v>.tar.gz` — ships `csrc/`, so `pip install drvarma` compiles
  the CFFI engine when GSL dev headers are present and otherwise installs
  pure-Python.
* **wheel** `drvarma-<v>-py3-none-any.whl` — pure-Python (built with
  `DRVARMA_NO_ENGINE=1`), so it installs with no C toolchain.

> Platform wheels *with* the compiled engine (manylinux/macOS/Windows) are a
> later CI job (cibuildwheel); the sdist already covers "build the engine if GSL
> is here".

## 0. One-time prerequisites

A modern `build` + `twine` (twine ≥ 6.1 / packaging ≥ 24.2 — older ones reject the
Metadata 2.4 `License-Expression`/`License-File` fields). Use an isolated env:

```sh
python3 -m venv ~/.venvs/release
~/.venvs/release/bin/pip install -U pip build twine
alias rbuild='~/.venvs/release/bin/python -m build'
alias rtwine='~/.venvs/release/bin/python -m twine'
```

(Set up a PyPI / TestPyPI API token in `~/.pypirc` or pass it at upload time.)

## 1. Bump the version

Edit **both** (keep them in sync):

* `pyproject.toml` → `[project] version = "X.Y.Z"`
* `src/drvarma/__init__.py` → `__version__ = "X.Y.Z"`

Follow semver. `0.1.0` is the first public release.

## 2. Build

```sh
rm -rf dist build
DRVARMA_NO_ENGINE=1 rbuild          # sdist + pure-Python wheel
```

## 3. Check

```sh
rtwine check dist/*                 # both must report PASSED
```

Sanity-install the wheel in a throwaway env and run the CLI:

```sh
python3 -m venv /tmp/t && /tmp/t/bin/pip install dist/drvarma-*.whl
/tmp/t/bin/python -c "import drvarma; print(drvarma.__version__)"
/tmp/t/bin/drvarma --help
```

## 4. TestPyPI (rehearsal)

```sh
rtwine upload --repository testpypi dist/*
python3 -m venv /tmp/tp && \
  /tmp/tp/bin/pip install -i https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ drvarma
```

## 5. PyPI (real)

```sh
rtwine upload dist/*
```

## 6. Tag the release

```sh
git tag -a vX.Y.Z -m "drvarma X.Y.Z"
git push --tags
```

## Notes

* **Name availability** — confirm `drvarma` is free/owned on PyPI before the first
  upload (an upload to a name you don't own will fail). If taken, change
  `[project] name` (e.g. `drvarma-py`) and the import stays `drvarma`.
* **Project URLs** — `[project.urls]` currently points at the C-engine repo
  (`github.com/davidesg/drvarma`); update if the Python port gets its own repo.
* **Optional extras** at install time: `drvarma[c-engine]` (GSL build),
  `drvarma[plots]` (matplotlib + pyfug), `drvarma[forecast-report]` (jinja2),
  `drvarma[test]`.
