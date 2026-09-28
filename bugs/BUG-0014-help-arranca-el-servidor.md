---
id: BUG-0014
title: sima --help starts the stdio server instead of printing help
status: open
severity: low
component: mcp
found_in: 0.1.7
fixed_in:
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
