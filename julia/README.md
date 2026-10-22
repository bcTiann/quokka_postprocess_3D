# Experimental Julia process: installation and use

The `codex/julia-experimental` branch adds a Julia/Mera processor for one Quokka
snapshot and reuses this repository's Python numerical preparation and plotting.
The Python commands on `main` remain available. Use a separate checkout,
environment and result directory for the Julia experiment; the original
simulation data can stay at its existing location.

The environment is locked to **Julia 1.13.2** and Mera commit
`db277419e05a45e8d4921fb5f3a0a4d15792cf7f`, which includes the Quokka frontend.
Keep both `Project.toml` and `Manifest.toml`; use `Pkg.instantiate()`, not
`Pkg.update()`, when installing this version. Python 3.11 and the repository's
`uv.lock` are also required, including with `--no-plot`: that option skips
rendering, while Python still prepares the saved numerical figure arrays.

This guide provides local and Setonix deployment commands. Setonix installation
and execution must be checked on the target machine; local validation does not
establish cluster performance. For the physics, reader algorithm, array schema
and complete option list, see [the implementation guide](../docs/julia_process.md).

## 1. Obtain an independent checkout

If the usual repository already exists on Setonix, create a worktree beside it:

```bash
cd "$MYSOFTWARE/quokka_postprocess_3D"
git fetch origin
git worktree add --track -b codex/julia-experimental \
  "$MYSOFTWARE/quokka_postprocess_3D-julia" \
  origin/codex/julia-experimental
cd "$MYSOFTWARE/quokka_postprocess_3D-julia"
```

Run that creation command once. If the local branch already exists, inspect
`git worktree list`; use its existing experimental worktree, or attach the
unoccupied branch with `git worktree add PATH codex/julia-experimental`.
Keep the original checkout on its existing branch. Worktrees share Git history,
but each needs its own `.venv` because Python installs this checkout in editable
mode.

Alternatively, create a separate clone. On Setonix, run this from `$MYSOFTWARE`;
locally, choose a normal project parent directory:

```bash
git clone --branch codex/julia-experimental \
  https://github.com/bcTiann/quokka_postprocess_3D.git \
  quokka_postprocess_3D-julia
cd quokka_postprocess_3D-julia
git branch --show-current
git rev-parse HEAD
```

Follow **either section 2 or section 3**, then export the assets in section 4.
All processing commands below run from this experimental repository root.

## 2. Local macOS or Linux installation

Install [Juliaup](https://github.com/JuliaLang/juliaup#installation) if it is not
already available. Open a new terminal after the installer configures PATH,
then return to this checkout:

```bash
curl -fsSL https://install.julialang.org | sh
```

Select the exact Julia version without changing the default for other projects:

```bash
juliaup add 1.13.2
export QPP_JULIA="$(julia +1.13.2 --startup-file=no -e 'print(joinpath(Sys.BINDIR, "julia"))')"
"$QPP_JULIA" --version
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then install both locked environments. Building the Python Cython extension
requires a C/C++ compiler (Xcode Command Line Tools on macOS, or the normal
compiler/build tools on Linux).

```bash
export QPP_REPO="$PWD"
uv sync --locked --python 3.11
export JULIA_NUM_PRECOMPILE_TASKS=4
export OPENBLAS_NUM_THREADS=1
"$QPP_JULIA" --startup-file=no --project=julia \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
.venv/bin/python -c 'import quokka2s, numpy, matplotlib'
"$QPP_JULIA" --startup-file=no --project=julia julia/bin/process.jl --help

export QPP_ASSETS="$HOME/quokka-julia-assets-v1"
export QPP_RESULTS="$HOME/quokka-julia-results"
mkdir -p "$QPP_RESULTS"
```

Juliaup chooses the binary for the local architecture. The local default Julia
package cache is `~/.julia`; the Setonix instructions below move it outside home.
The `QPP_*` exports are shell settings, so repeat them when opening a new terminal.

## 3. Setonix installation under `$MYSOFTWARE`

Use the Setonix **CPU** environment. Download and install Linux dependencies on
Setonix; do not copy a macOS `.venv`, Julia binary, package cache or compiled
libraries. The source checkout, lock files and numeric asset files are portable.

Create a small environment file outside Git. Source it in future shells and
batch jobs; it sets the paths used by the rest of this guide:

```bash
mkdir -p "$MYSOFTWARE/quokka-julia-runtime"
cat > "$MYSOFTWARE/quokka-julia-runtime/environment.sh" <<'EOF'
export QPP_REPO="$MYSOFTWARE/quokka_postprocess_3D-julia"
export QPP_RUNTIME="$MYSOFTWARE/quokka-julia-runtime"
export QPP_JULIA="$QPP_RUNTIME/julia-1.13.2/bin/julia"
export QPP_ASSETS="$MYSOFTWARE/quokka-julia-assets-v1"
export QPP_RESULTS="$MYSOFTWARE/quokka-julia-results"
export JULIA_DEPOT_PATH="$QPP_RUNTIME/depot"
export JULIA_NUM_PRECOMPILE_TASKS=4
export UV_CACHE_DIR="$QPP_RUNTIME/uv-cache"
export UV_PYTHON_INSTALL_DIR="$QPP_RUNTIME/python"
export PATH="$MYSOFTWARE/uv/bin:$PATH"
export MPLCONFIGDIR="$QPP_RUNTIME/matplotlib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
EOF
source "$MYSOFTWARE/quokka-julia-runtime/environment.sh"
mkdir -p "$QPP_RUNTIME/downloads" "$QPP_RESULTS/logs" "$MPLCONFIGDIR"
```

Install the official **Linux x86_64** binary on Setonix CPU nodes. Check that
`uname -m` prints `x86_64`. These URLs come from Julia's
[official manual downloads](https://julialang.org/downloads/manual-downloads/),
including its [1.13.2 SHA256 list](https://julialang-s3.julialang.org/bin/checksums/julia-1.13.2.sha256):

```bash
uname -m
cd "$QPP_RUNTIME/downloads"
curl --fail --location --output julia-1.13.2-linux-x86_64.tar.gz \
  https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.2-linux-x86_64.tar.gz
curl --fail --location --output julia-1.13.2.sha256 \
  https://julialang-s3.julialang.org/bin/checksums/julia-1.13.2.sha256
sha256sum --check --ignore-missing julia-1.13.2.sha256
```

Continue only after the checksum reports the Linux archive as `OK`. If Julia
1.13.2 is already installed at this exact path, skip the download/extraction.

```bash
tar -xzf julia-1.13.2-linux-x86_64.tar.gz -C "$QPP_RUNTIME"
"$QPP_JULIA" --version
```

Use an existing uv installation, or install it outside home:

```bash
curl -LsSf https://astral.sh/uv/0.11.31/install.sh |
  env UV_INSTALL_DIR="$MYSOFTWARE/uv/bin" sh
export PATH="$MYSOFTWARE/uv/bin:$PATH"
```

Install Python in this checkout, following the repository's existing
[Setonix Python environment instructions](../docs/setonix.md#2-configure-the-environment):

```bash
cd "$QPP_REPO"
module load python/3.11.6
uv sync --locked --python "$(command -v python)"
"$QPP_JULIA" --startup-file=no --project=julia \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
.venv/bin/python -c 'import quokka2s, numpy, matplotlib'
"$QPP_JULIA" --startup-file=no --project=julia julia/bin/process.jl --help
```

If that site Python module has changed, inspect `module spider python` and use
an available Python 3.11 module consistently for installation and the batch job.
A GCC/Clang-compatible compiler must be available for the Python extension.
Dependency installation needs network access and can take several minutes;
finish it before submitting processing jobs. Keep the same depot and environment
paths in the job so the compute node can find the installed packages.

## 4. Export the existing fixed tables once

From the experimental checkout, with `QPP_ASSETS` set by section 2 or 3:

```bash
.venv/bin/python julia/export_tables.py --output "$QPP_ASSETS"
```

The destination must be new. The exporter reads the repository's tracked
`inputs/tables/despotic/interpolated.npz`,
`inputs/tables/cloudy/emission.npz` and fixed Draine dust file. It writes
`tables.npz` and `physics.json`, retaining source SHA256 hashes and the adopted
abundances. It does not run Cloudy or DESPOTIC, rebuild the tables, or modify them.
`--despotic PATH --cloudy PATH --dust PATH` can select explicit existing sources.

When transferring an already exported asset directory, copy **both** files.
The Julia loader checks `tables.npz` against the hash in `physics.json`; the
processing report also records their identity. Preserve each asset version and
use a new directory after an intentional table change. Do not overwrite assets
while a calculation is running.

## 5. Select a snapshot and run a small region first

Choose a snapshot in a different physical state for the next validation.
Set one **absolute snapshot path**, an explicit manually chosen job name, and
a fresh output path. The input must be a `plt` plus digits snapshot directory,
not its parent run directory. For example, replace the path below:

```bash
export QPP_DATASET=/absolute/path/to/run/plt0000000
export QPP_JOBNAME=my-run-julia-validation
export QPP_SNAPSHOT="$(basename "$QPP_DATASET")"
export QPP_SMOKE="$QPP_RESULTS/$QPP_JOBNAME/$QPP_SNAPSHOT-smoke-v1"
```

Locally, run:

```bash
"$QPP_JULIA" --startup-file=no --project=julia --threads=4 julia/bin/process.jl \
  --dataset "$QPP_DATASET" --jobname "$QPP_JOBNAME" \
  --assets "$QPP_ASSETS" --output "$QPP_SMOKE" \
  --python "$PWD/.venv/bin/python" \
  --x 2 4 --y 2 4 --format png --style titled
```

On Setonix, use the batch script in section 6 with `--x 2 4 --y 2 4` appended
for the first smoke run. Numerical processing belongs in a compute allocation.
These ranges select four complete z columns; start indices are included and
stop indices excluded. These interior example ranges require at least five
cells along both x and y and avoid wrapping the periodic halo across the box.
Choose valid indices for the selected snapshot. Omitting
one axis independently selects that whole axis. A small region can still read
large intersecting FAB blocks, so its I/O need not scale with its cell count.

`--jobname` is required and is never inferred from the path. `--output` names
the exact result directory; Julia does not automatically construct the Python
job/snapshot hierarchy or read `configs/emission.yaml`. Titled plots identify
the saved job name, snapshot name and actual simulation time in Myr. Untitled
plots keep the manuscript layout.

For a full local snapshot, choose a different output path and omit both ranges:

```bash
"$QPP_JULIA" --startup-file=no --project=julia --threads=4 julia/bin/process.jl \
  --dataset "$QPP_DATASET" --jobname "$QPP_JOBNAME" \
  --assets "$QPP_ASSETS" \
  --output "$QPP_RESULTS/$QPP_JOBNAME/$QPP_SNAPSHOT-full-v1" \
  --python "$PWD/.venv/bin/python" \
  --format png pdf --style titled untitled
```

The output directory must not exist and must be outside the input snapshot
and assets. A failed calculation is preserved; choose a new suffix to retry.

## 6. Run one snapshot in a Setonix CPU batch job

Save this as `$QPP_RESULTS/run_snapshot.sbatch`. Replace `YOUR_PROJECT`,
`QPP_DATASET` and `QPP_JOBNAME`. Start with a smoke run by appending the x/y
options noted above, then submit a full run after checking its outputs.

The example requests one Julia process with four threads, 16 GiB and 30 minutes.
These are initial resources for validation, not measured Setonix requirements;
increase memory/time for larger snapshots and compare thread counts later.
The single-task launch follows Pawsey's
[CPU batch-job guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51927426/Example+Slurm+Batch+Scripts+for+Setonix+on+CPU+Compute+Nodes).

```bash
#!/bin/bash --login
#SBATCH --account=YOUR_PROJECT
#SBATCH --partition=work
#SBATCH --job-name=quokka-julia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
source "$MYSOFTWARE/quokka-julia-runtime/environment.sh"
module load python/3.11.6
cd "$QPP_REPO"

QPP_DATASET=/absolute/path/to/run/plt0000000
QPP_JOBNAME=my-run-julia-validation
QPP_SNAPSHOT="$(basename "$QPP_DATASET")"
QPP_OUTPUT="$QPP_RESULTS/$QPP_JOBNAME/$QPP_SNAPSHOT-slurm-$SLURM_JOB_ID"

# Save the checkout identity and launch settings in the Slurm log.
git rev-parse HEAD
git status --short
"$QPP_JULIA" --version
"$PWD/.venv/bin/python" --version
printf 'dataset=%s\njobname=%s\noutput=%s\nthreads=%s\n' \
  "$QPP_DATASET" "$QPP_JOBNAME" "$QPP_OUTPUT" "$SLURM_CPUS_PER_TASK"

srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --distribution=block:block:block \
  "$QPP_JULIA" --startup-file=no --project=julia \
  --threads="$SLURM_CPUS_PER_TASK" julia/bin/process.jl \
  --dataset "$QPP_DATASET" --jobname "$QPP_JOBNAME" \
  --assets "$QPP_ASSETS" --output "$QPP_OUTPUT" \
  --python "$PWD/.venv/bin/python" \
  --format png pdf --style titled untitled
```

Submit from a shell with the environment file loaded:

```bash
mkdir -p "$QPP_RESULTS/logs"
sbatch --output="$QPP_RESULTS/logs/julia-%j.log" \
  --error="$QPP_RESULTS/logs/julia-%j.err" \
  "$QPP_RESULTS/run_snapshot.sbatch"
```

The Slurm job ID makes each output directory new. Save the submitted script
with the results when comparing runs. Leave a long run alone and inspect the
log and reports after it finishes; continuous monitoring is unnecessary.
`sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS` provides scheduler
status, including the `srun` job step. Its memory accounting differs from the
Julia-only peak RSS in `report.json`, especially when Python subprocesses run.

## 7. Check completion and redraw from saved results

Open `status.json` and `report.json` in the exact output directory printed in
the log. A successful full calculation has `processing_complete: true` and
`full_snapshot: true` in the report. If plots were requested, also check
`plot_status: completed`. With `--no-plot`, numerical preparation still finishes
and the plot status is `not_requested`. `raw_processing_complete` alone is an
intermediate state, not the completed prepared result.

The output contains `products.npz`, the reports and `prepared/`. Standard plots
are written under `figures/` (untitled) and `figures_titled/`. There are 63 figures
per style and format, or 252 files with both formats and both styles.

To render later on Setonix or locally:

```bash
.venv/bin/python julia/plot_results.py /absolute/path/to/result/prepared \
  --output /absolute/path/to/result/figures \
  --format png pdf --style titled untitled
```

For local plotting, install the branch's Python environment and copy the whole
`prepared/` directory, including `preparation.json` and `emission_report.json`.
No Julia runtime, original snapshot, table assets or raw `products.npz` is needed
for plot-only use. The command verifies saved-file hashes and uses saved job/time
metadata; it never reopens the snapshot. Do not edit prepared arrays or metadata
by hand.

Results prepared before job/time metadata was added can be upgraded from the
saved raw result without recalculating emission:

```bash
.venv/bin/python julia/prepare_results.py /absolute/path/to/old-result \
  --jobname my-run-julia-validation
```

A conflicting saved job name is rejected. If an older diagnostic report did
not store physical time, the title says `time = unavailable`; it is not inferred
from the snapshot number. New snapshot processing saves time automatically.

## 8. Limits, validation and updates

The current reader supports uniform, level-0, three-dimensional Cartesian
snapshots with cubic cells, stored temperature, positive metadata unit factors,
and FAB boxes forming aligned complete z layers. Non-cubic domains are allowed.
AMR, staggered z-layer layouts, temperature reconstruction, whole-run discovery
and numerical checkpoint restart are not implemented. Mera's broader capabilities
do not automatically extend this processor's supported inputs.

The default retained-reader budget is 1,024 MiB (`--reader-cache-mib`); it is
not a total memory cap. Runtime, Mera temporaries, emission/product arrays and
Python need additional memory. `--batch-cells` defaults to 131,072 and must fit
at least one selected x/y plane. A larger snapshot may require both values to
be increased after reviewing the reader's plan/error.

Before adopting a new physical case, compare it with the Python pipeline using
the same snapshot, region and fixed assets. Inspect missing/zero masks, counts,
mass and luminosity conservation, images, spectral profiles and moments. A
completed report checks internal consistency; it is not by itself an independent
cross-implementation validation. Local speed measurements are not a Setonix
benchmark.

Update the experimental deployment only while no job uses that checkout:

```bash
cd "$QPP_REPO"
git status --short
git pull --ff-only origin codex/julia-experimental
uv sync --locked
"$QPP_JULIA" --startup-file=no --project=julia \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
git rev-parse HEAD
```

Use `cd` to the local experimental checkout if `QPP_REPO` is not set. Preserve
local changes before pulling. The experimental branch should receive reviewed
`main` changes through normal merges; deployment machines should pull the tested
experimental branch, rather than merging `main` into an active calculation.
The processor checks source identity at the beginning and end, so code or asset
changes during a run cause failure. Keep the commit, settings, logs and reports
with each validation. The external result paths in this guide keep personal
validation data outside public Git commits.
