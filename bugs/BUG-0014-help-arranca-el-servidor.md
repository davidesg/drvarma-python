---
id: BUG-0014
title: sima --help starts the stdio server instead of printing help
status: fixed
severity: low
component: mcp
found_in: 0.1.7
fixed_in: sima-tseries 0.1.0, drtran 0.2.5, art-tseries 0.2.3
reported: 2026-09-27
reporter: External review of atsw 1.6.1 (2026-09-27), filed 2026-09-28
tags: [cli, help]
references: []
---

## Summary

`sima --help` (and `mtram --help`, `art-mcp --help`) start the stdio server
instead of printing a usage line, so the command seems to hang.

## Impact

Minor, but it is the first thing a user tries.

## Reproduction

Review §2.6. Still so on 2026-09-28: `mcp_server.main()` calls `mcp.run()`
with no argument parsing.

## Root cause

No argument handling before `mcp.run()`.

## Fix

Handle `-h/--help` (and `--version`) before starting the server, in the three
servers.

## Validation

`sima --help` exits 0 with a usage line.

## Fix (2026-10-04)

The fix is in the three servers, not in drvarma: `_cli_flags()` in each
`mcp_server.py`, called by `main()` before `mcp.run()`. `-h`/`--help` prints a
usage line that says the command speaks MCP over stdio and is started by a
client; `--version` prints the distribution's version (`sima-tseries`,
`drtran`, `art-tseries`). Both exit 0. Any other argument is left alone, as
before.

Validation: `tests/test_cli_help.py` in sima-python, drtran-python and
art-python (help, version, and no flags returns).

