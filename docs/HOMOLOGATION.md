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

## On the MA wall, the iterations too

A VARMA with an MA can have no interior maximum: the search runs along the
invertibility wall and stops there. The point where it stops is then not a
property of the model but of the path, and two engines reach the same one
only if they take the same steps. So for those cases the record also keeps
the number of iterations (`nit`), and both sides are checked against it.

What that needed in the port (BUG-0030): the C's cast (no reciprocal of an
MA(1) beyond the unit circle: the C refuses the point), the C's order of the
parameters, and the C's arithmetic in the ols seeds. A difference in the
sixteenth digit is enough to part the paths there; one in the cast is not
rounding at all.

**With the compiled likelihood only.** The wheel carries the C's own `elf`,
and with it the port takes the C's steps one by one. The pure-Python
fallback (no extension built) evaluates the same likelihood with another
arithmetic: equal to rounding, which is enough wherever there is a maximum
and not on the wall. Measured on the four wall cases of the record: one ends
where the C does, three do not (by 0.0003 to 0.4 in logL). So those cases
are checked with the compiled likelihood and skipped without it. The
fallback is a fallback: about a hundred times slower, and not what an
analysis should run on.

**On the platform of the record only.** The record was made on Linux
x86-64, C and port. On the wall the point follows the last digit of the
arithmetic, and that digit is the compiler's and the C library's: the same
fit (the flour prices' echelon form, in sima's suite) stops at −658.9552 on
Linux, −659.2055 on Windows and −659.7980 on macOS. So the cases with
iterations are checked on Linux x86-64 and skipped elsewhere. Whether the
port and the C agree with each other on macOS or Windows has not been
measured: the C has only been run on Linux for this record.

## When the C changes

1. Build the C at the new commit and run the port's suite with `ATSW_GUI`
   and `DRVARMA_C_BIN` set.
2. Bring the change to the port until the C's numbers are the port's.
3. Update `tests/data/homologation.json`: the commit, the date, and any
   case whose number moved on purpose, saying so in the changelog.
4. Only then release (`RELEASING.md`).

Changing a file under `csrc/` makes `test_the_vendored_c_is_the_recorded_one`
fail until the record is updated: the same steps.

## The `.inp` path (deprecated)

`csrc/` is the C's: every vendored file is identical to atsw-gui's at the
recorded commit (`identical_to_c` in the record), only `drvarma_api.c` and
`.h`, the binding, are the package's own (BUG-0002, 2026-11-28).
`tests/test_inp_path_as_the_c.py` holds six cases of the C's battery
(`tests/banco/ref`, `tests/data/inp_c/cases.json`): with the compiled engine
the port's parameter table is the C's line for line, the standard errors
fdhess's computed by the C itself (`estimate_w(hessian="fd")`). The
pure-Python estimator matches on two of them and not yet on four (BUG-0033).

## What is not homologated against the current C

- **The pure-Python `.inp` path** on the four cases of BUG-0033.
- **The older `.inp` tests** (`test_estimate_cffi.py`, `test_estimate_py.py`,
  `test_deseason.py`, `test_diagnostics.py`, `test_forecast.py`,
  `test_irf.py`, `test_recursive.py`, `test_report.py`,
  `test_residual_report.py`, `test_volatility.py`, `test_pure_python_out.py`,
  `test_io_transform.py`, `test_reliability.py`) compare against the frozen
  0.4.1 program (`drvarma_v.04.1`) and its saved outputs, and are skipped
  where that checkout is missing. The path is removed in drvarma 6.0.
- **What exists in one engine only**: `docs/ARCHITECTURE.md` §8.
