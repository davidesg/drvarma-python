# Homologation with the C engine

**The C engine and this port are the same engine.** They must give the same
results; the C is the oracle, a change starts there, and the port moves with
it. The rules are in atsw-gui `docs/MOTORES.md`. This page says how the port
records and checks it.

## The record

`tests/data/homologation.json`:
- **the C's commit** the port was last homologated against
  (`davidesg/atsw-gui`, `engines/drvarma`);
- **the cases**: for each, the C's command line, this port's call, and the
  log-likelihood and parameter count both give;
- **the vendored C sources** (`csrc/`), each with its sha256, the C file it
  comes from and whether it is identical to it at that commit.

## The checks

`tests/test_c_oracle.py`:

| check | runs |
|---|---|
| the port gives every recorded number | always |
| the C gives every recorded number | when its binary is at hand |
| the C at hand is at the recorded commit | when its binary is at hand |
| the fixtures are the C battery's files, byte for byte | with a checkout of atsw-gui |
| `csrc/` is the recorded one | always |

The C is found through `ATSW_GUI` (the checkout, built; `DRVARMA_C_BIN`
names another binary). No folder is guessed: a checkout that happens to be
nearby may be at any commit. The binary says its commit, and a test checks
it is the recorded one:

```sh
make -C $ATSW_GUI/engines/drvarma
ATSW_GUI=/path/to/atsw-gui pytest tests/test_c_oracle.py
```

The other homologation tests pin the C's numbers one by one, with the
section of the C's battery they come from: `test_homologation_c_0_4.py` (the
default start, `split`), `test_homologation_c_det.py` (the mask of the
deterministic terms), `test_homologation_c_route_v.py` (route V) and
`test_homologation_c_echelon.py` (the echelon form). The C's battery pins
the port's numbers in turn (`engines/drvarma/tests/escalera/pruebas.sh`).

## When the C changes

1. Build the C at the new commit and run the port's suite with `ATSW_GUI`
   and `DRVARMA_C_BIN` set.
2. Bring the change to the port until the C's numbers are the port's.
3. Update `tests/data/homologation.json`: the commit, the date, and any
   case whose number moved on purpose, saying so in the changelog.
4. Only then release (`RELEASING.md`).

Changing a file under `csrc/` makes `test_the_vendored_c_is_the_recorded_one`
fail until the record is updated: the same steps.

## What is not homologated against the current C

- **The deprecated multivariate `.inp` path.** Its tests
  (`test_estimate_cffi.py`, `test_estimate_py.py`, `test_deseason.py`,
  `test_diagnostics.py`, `test_forecast.py`, `test_irf.py`,
  `test_recursive.py`, `test_report.py`, `test_residual_report.py`,
  `test_volatility.py`, `test_pure_python_out.py`, `test_io_transform.py`,
  `test_reliability.py`) compare against the frozen 0.4.1 program
  (`drvarma_v.04.1`) and its saved outputs, and are skipped where that
  checkout is missing. The C's `.inp` path has moved since (fdhess standard
  errors, the MA wall report): pointing them at it means regenerating their
  references. The path is removed in drvarma 6.0.
- **`csrc/`.** Only `elfvarma.c` is identical to the C's. `drvmlest.c`,
  `qnewtopt.c`, `nlatools.c` and `main.h` differ (BUG-0002), and
  `multshea.c` in its header. The record pins what they are; bringing them
  to the C's is pending.
- **What exists in one engine only**: `docs/ARCHITECTURE.md` §8.
