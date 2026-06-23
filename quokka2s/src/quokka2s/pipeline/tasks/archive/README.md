# tasks/archive — retired task scripts (inert reference)

These task modules are **not part of the live pipeline** and are **not imported**
by anything (no `__init__.py` here on purpose, so this is a plain folder, not a
Python package). They are kept as reference / backup per the project's
"deprecate by wrapping, don't delete" habit.

Archived 2026-06-24. Their internal `from ..base import ...` relative imports
are dormant (they only resolved from `tasks/`); to reactivate one, `git mv` it
back up into `tasks/`, re-add its import to `tasks/__init__.py` + a
`register_task(...)` line in `run_pipeline.py`, and fix any import depth.

Recoverable history: the live versions are in git up to commit 2d3fb84.
