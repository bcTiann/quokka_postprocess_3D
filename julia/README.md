# Experimental Julia process with uv and YAML

The `codex/julia-experimental` branch processes one Quokka snapshot with
Julia/Mera and uses the existing Python numerical preparation and plotting.
Use an independent checkout and choose a snapshot for validation. The original
simulation data stays at its existing location.

From the repository root, daily use is:

```bash
uv run process-julia --dry-run
uv run process-julia
```

Both commands read [configs/emission_julia.yaml](../configs/emission_julia.yaml).
Edit the snapshot, job name and output there; `--config PATH` selects another
file. The launcher supplies the current uv Python, Julia project, package depot,
thread settings and plotting cache. No environment activation, per-run exports,
temporary configuration files or Slurm allocation is needed to start it.

The locked environments require **Python 3.11** and **Julia 1.13.2**. Mera is
pinned to commit `db277419e05a45e8d4921fb5f3a0a4d15792cf7f`, including the Quokka
frontend. Keep `uv.lock`, `julia/Project.toml` and `julia/Manifest.toml` together.
The uv project manages Python; Julia's Pkg manages its locked dependencies.
See [the implementation guide](../docs/julia_process.md) for the scientific
definitions, native layout restrictions and saved-array schema.

## Install an independent checkout

Locally, choose a normal project parent directory. On Setonix, start under
`$MYSOFTWARE` so the checkout, `.venv`, runtime and output stay outside home:

```bash
cd "$MYSOFTWARE"  # Setonix only; choose your own parent locally.
git clone --branch codex/julia-experimental \
  https://github.com/bcTiann/quokka_postprocess_3D.git \
  quokka_postprocess_3D-julia
cd quokka_postprocess_3D-julia
```

An existing repository can instead supply an independent worktree:

```bash
git fetch origin
git worktree add --track -b codex/julia-experimental \
  ../quokka_postprocess_3D-julia origin/codex/julia-experimental
```

Run worktree creation once. If that branch already exists, use
`git worktree list` to find its checkout, or attach the unoccupied local branch
with `git worktree add PATH codex/julia-experimental`. Each checkout needs its
own `.venv` because this project is installed in editable mode.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then use either the local or Setonix setup below. A C/C++ compiler and Git are
needed to build the existing Python spectrum extension and dependencies.

### Local macOS or Linux

Install [Juliaup](https://github.com/JuliaLang/juliaup#installation) if needed:

```bash
curl -fsSL https://install.julialang.org | sh
```

Open a new terminal after installation, return to the checkout, then select
the pinned Julia for this directory without changing other projects:

```bash
juliaup add 1.13.2
juliaup override set 1.13.2
uv sync --locked --python 3.11
uv run process-julia --setup
```

The YAML's `julia_executable: julia` finds Julia on PATH. If Juliaup is not used,
set `julia_executable` to the installed Julia 1.13.2 binary's absolute path.

### Setonix

Install Linux dependencies on Setonix; a macOS `.venv`, Julia binary or compiled
package cache cannot be transferred. Source code, lock files and exported
numeric tables are portable. All commands below start in the experimental
checkout and run in the current shell.

Use an existing uv installation, or install it once outside home:

```bash
curl -LsSf https://astral.sh/uv/0.11.31/install.sh |
  env UV_INSTALL_DIR="$MYSOFTWARE/uv/bin" sh
```

Open a new terminal so the installer's PATH setup takes effect. Return to the
checkout and install the locked Python environment. This branch's uv project
uses `runtime/uv-cache/`, so the project cache also stays outside home without
an export. An already configured `UV_CACHE_DIR` takes precedence.

```bash
module load python/3.11.6
uv sync --locked --python "$(command -v python)"
```

If the site module changes, use `module spider python` to find an available
Python 3.11 module. Keep its compiler/runtime available when rebuilding the
environment. The launcher reuses the resulting uv Python automatically.

Use an installed Julia 1.13.2 by putting its absolute executable path in YAML.
Alternatively, install the official Linux x86_64 binary inside this checkout:

```bash
uname -m
mkdir -p runtime/julia/downloads
cd runtime/julia/downloads
curl --fail --location --output julia-1.13.2-linux-x86_64.tar.gz \
  https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.2-linux-x86_64.tar.gz
curl --fail --location --output julia-1.13.2.sha256 \
  https://julialang-s3.julialang.org/bin/checksums/julia-1.13.2.sha256
sha256sum --check --ignore-missing julia-1.13.2.sha256
tar -xzf julia-1.13.2-linux-x86_64.tar.gz -C ..
cd ../../..
```

Check that the architecture is `x86_64` and the checksum reports the Linux
archive as `OK` before extracting. Downloads and checksums are from
[Julia's official releases](https://julialang.org/downloads/manual-downloads/).
In `configs/emission_julia.yaml`, set:

```yaml
julia_executable: ../runtime/julia/julia-1.13.2/bin/julia
```

Then initialize the project:

```bash
uv run process-julia --setup
```

`--setup` checks the exact Julia version, instantiates and precompiles the
checked-in manifest, and exports the repository's current fixed emission/dust
tables if the configured assets are absent. Existing assets are checked against
their source hashes and reused; conflicting assets are preserved and reported.
This does not run Cloudy or DESPOTIC or change abundances. Setup needs network
access; later processing reuses the installed environment.

By default all launcher-managed files stay under ignored `runtime/julia/`:
`depot/` for Julia packages, `assets/` for `tables.npz` and `physics.json`, and
`matplotlib/` for the plotting cache. uv uses `runtime/uv-cache/`. No separate
shell environment file is required. Setonix installation and execution still
need verification on the target machine; local tests do not establish cluster
performance.

## Select inputs in YAML

Edit [configs/emission_julia.yaml](../configs/emission_julia.yaml), for example:

```yaml
dataset: /absolute/path/to/run/plt0000000
jobname: my-run-julia-validation
output: ../output/julia-experimental/my-run/plt0000000-full-v1

x_index_range: null
y_index_range: null
threads: 1
figure_formats: [png]
figure_styles: [titled]
```

`dataset` names one complete `plt` plus digits directory. A YAML dataset may
be relative to the YAML file; an explicit `--dataset` override must be absolute.
`jobname` is chosen manually, never inferred from the path. `output` is the
exact destination and must be new, outside the snapshot and asset directories.
The `output/julia-experimental/` subtree is ignored by Git. Choose a new output
suffix after a failed or completed attempt; automatic reuse and restart are
not implemented. Changing dataset/jobname does not update `output`
automatically; change all three fields when selecting a new case.

All relative YAML paths start at the YAML file's directory. `~` is expanded;
shell variables such as `$MYSOFTWARE` are not expanded inside YAML. A bare
`julia_executable: julia` uses PATH; a path with a slash is YAML-relative.
CLI path overrides are relative to the current shell directory, except that
`--dataset` requires an absolute path. CLI values override YAML, then built-in
defaults supply omitted settings. The resolved launch settings, YAML path and
YAML hash are recorded in `report.json` under `run_configuration`.

Keep `x_index_range` and `y_index_range` at `null` for the full x/y box. An axis
can independently use `[START, STOP]`, with 0-based included start and excluded
stop, while every sightline retains the complete z depth. The launcher defaults
to one Julia thread and a single BLAS thread; choose `threads` explicitly when
comparing resource budgets. Other settings in the commented YAML control the
retained-reader budget, emission batch size and figure selection.

## Run a region, then the full snapshot

Review the resolved configuration without accessing Julia, snapshots or assets:

```bash
uv run process-julia --dry-run
```

Override x/y and output for a first small-region check:

```bash
uv run process-julia --x 2 4 --y 2 4 \
  --output output/julia-experimental/my-run/plt0000000-smoke-v1
```

These ranges select four full z columns and need at least five native cells
on each x/y axis for their surrounding halo. Choose valid interior indices for
the selected snapshot. CLI axes override only their respective YAML axes.
Small selections may still read large intersecting FAB blocks.

For the full snapshot, keep both YAML ranges `null` and run:

```bash
uv run process-julia
```

For another saved configuration or explicit input selection:

```bash
uv run process-julia --config configs/my_julia_run.yaml
uv run process-julia --dataset /absolute/path/to/other/plt0000000 \
  --jobname other-run \
  --output output/julia-experimental/other-run/plt0000000-v1
```

Every explicit `--dataset` requires an explicit `--jobname`. Python remains
required with `--no-plot`: that option saves raw and prepared products and skips
only rendering. `--format png pdf --style titled untitled` requests all 252
standard figure files. Use `uv run process-julia --help` for accepted overrides.

The command runs on the node where it is invoked and makes no scheduler
request. Pawsey describes login nodes as the entry point for preparing and
submitting work; choose an allocation for sustained workloads according to the
[site guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925858/Connecting+to+a+Supercomputer).
Slurm is optional for this entry point. Within an existing batch script, use
the same checkout, Python module and command; set YAML `threads` to the assigned
CPU count, or pass `--threads "$SLURM_CPUS_PER_TASK"`. No separate Julia command
or environment file is needed.

## Check completion and redraw

Leave long runs to finish and inspect their log and the output directory's
`status.json` and `report.json` afterwards. Redirect normal console output to a
chosen log if desired:

```bash
uv run process-julia > julia-process.log 2>&1
```

A full result has `processing_complete: true` and `full_snapshot: true` in
`report.json`. If plots were requested, check `plot_status: completed`.
With `--no-plot`, plot status is `not_requested`; numerical preparation still
finishes. `raw_processing_complete` alone is an intermediate state.

Results contain `products.npz`, the reports, `prepared/`, and requested figure
directories: `figures/` for untitled and `figures_titled/` for titled. There are
63 figures per selected format/style. Titled plots use the saved job name,
snapshot and actual simulation time in Myr; untitled plots use the paper layout.

Render again from saved products:

```bash
uv run python julia/plot_results.py /absolute/path/to/result/prepared \
  --format png pdf --style titled untitled
```

For plotting on another machine, install this branch's Python environment and
copy the whole `prepared/` directory, including `preparation.json` and
`emission_report.json`. Plot checks saved-file hashes and reads saved job/time
metadata. It needs no Julia installation, original snapshot, numeric table
assets or raw `products.npz`.

Older saved results can acquire job/time title metadata without recalculating
emission:

```bash
uv run python julia/prepare_results.py /absolute/path/to/old-result \
  --jobname my-run-julia-validation
```

A conflicting saved job name is rejected. Missing physical time is displayed
as unavailable; it is not inferred from the snapshot number.

## Supported inputs and updates

The current reader supports uniform level-0 Cartesian 3D snapshots with cubic
cells, stored temperature, positive metadata unit factors, and FAB boxes
forming aligned complete z layers. Non-cubic domains are allowed. AMR,
staggered z-layer layouts, temperature reconstruction, run-directory discovery
and numerical checkpoint restart are not implemented.

`reader_cache_mib: 1024` is a retained-array budget, not a total memory cap.
Mera temporaries, Julia, emission/product arrays and Python need additional
memory. `batch_cells: 131072` must fit one complete selected x/y plane. Review
the reader's plan/error before increasing these for larger inputs.

For a different physical state, compare Python and Julia on the same snapshot,
region and fixed tables. Check missing/zero masks, counts, conservation, images,
spectra and moments. A completed internal report is not an independent
cross-implementation validation.

Update only while no run uses the checkout:

```bash
git status --short
git pull --ff-only origin codex/julia-experimental
uv sync --locked
uv run process-julia --setup
git rev-parse HEAD
```

Preserve local changes before pulling. `configs/emission_julia.local.yaml` is
ignored by Git; copy the example there and select it with `--config` for personal
machine paths and run settings. Retain it with the validation results. Use reviewed
merges to bring `main` into the experimental branch; processing detects source
or asset changes during a run. Preserve the commit, effective settings, log and
reports for each comparison. Run-specific validation outputs stay outside
public commits.
