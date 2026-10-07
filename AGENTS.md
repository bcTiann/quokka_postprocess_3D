# Project guidance

## Understand the current workflow

- Start with `docs/code_walkthrough.md` and `docs/repository_layout.md`.
- Use `docs/emission_method.md` and the current source for scientific rules.
- `process_snapshot.py` calculates and saves numerical products.
- `plot_emission_results.py` reads those products; it must not read snapshots or query emission tables.
- All renderers read prepared numerical arrays. Normalization, physical unit conversions, pixel sums, display masks/ranges and contour analysis belong to numerical preparation before saving, including auxiliary figures. Plot readers/getters only select stored values. Matplotlib scaling, tick/label formatting and layout remain rendering.

## Write readable code

- Use descriptive names for data, units and actions.
- Write one assignment per line and expand complex calls with named arguments.
- Write concise English comments and docstrings. Explain inputs, outputs, shapes, units and source data; include a small example when useful.
- Split complex operations into meaningful stages. Keep helpers when their names explain a real action; remove pure forwarding and redundant packaging.
- Prefer standard NumPy/SciPy tools to handwritten equivalents when behavior matches.
- Update callers together when simplifying an interface. Avoid unused compatibility layers and speculative frameworks.

## Refactor in small passes

- Inspect callers and data ownership before editing. Identify the existing behavior, intended structural improvement and smallest useful verification.
- Address one cleanup theme at a time: unused code, duplicate logic, control flow, module boundaries or interfaces.
- Preserve scientific rules, units, per-line missingness, numerical products and figure appearance unless the user authorizes a change.
- Preserve reduction and parallel merge order when refactoring numerical accumulation.
- Ask the user before changing scientific assumptions or output meaning.

## Verify and report

- Use existing relevant checks. Add tests only for a meaningful uncovered behavior.
- For numerical refactors, compare named arrays and missing values on an appropriate real slab or region, and check conservation.
- For rendering refactors, compare figures from the same saved products, including paper and titled versions when affected.
- For performance changes, measure representative fixed inputs before and after; clearer code alone does not establish a speed improvement.
- Review the final diff for unintended changes and update affected usage or code-reading documentation.
- Keep personal validation outputs, local tests and conversation records out of public Git commits. Put local comparison artifacts under ignored `output/`.
