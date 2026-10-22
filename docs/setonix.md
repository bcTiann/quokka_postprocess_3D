# Setonix: process and plot with uv

Keep the repository, environment and generated results under `$MYSOFTWARE`.
Run from the repository root. Bundled tables need no separate DESPOTIC installation.

## 1. Clone

```bash
cd "$MYSOFTWARE"
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

To update: `git switch main`, then `git pull --ff-only origin main`.

## 2. Configure the environment

Start with a new SSH login. The following setup is needed only once. Keep the
repository and `.venv` under `$MYSOFTWARE`, as in step 1.

For the default Setonix Bash login, save the cache locations in `~/.profile`:

```bash
cat >> ~/.profile <<'EOF'
export UV_CACHE_DIR="$MYSOFTWARE/uv/cache"
export MPLCONFIGDIR="$MYSOFTWARE/uv/matplotlib"
EOF
source ~/.profile
```

If your Bash login uses `~/.bash_profile` or `~/.bash_login`, add those two lines
there instead. Skip this step if these locations are already saved. These
settings are loaded automatically on later logins; do not repeat them per run.

Install uv using its normal installer, which saves its PATH setup for future
terminals. Only the binary location changes to avoid the home quota:

```bash
curl -LsSf https://astral.sh/uv/0.11.31/install.sh |
  env UV_INSTALL_DIR="$MYSOFTWARE/uv/bin" sh
```

Open a new terminal, return to the repository, then create the environment:

```bash
cd "$MYSOFTWARE/quokka_postprocess_3D"
module load python/3.11.6
uv sync --python "$(command -v python)"
```

uv creates `.venv` and installs the checkout in editable mode. No activation,
`PYTHONPATH` or manual `pip install` is needed. Building yt and the Cython
spectrum kernel needs Git and C/C++ build tools. uv installs the pinned Cython
and SciPy build dependencies automatically. Keep a GCC/Clang-compatible C
compiler available when rebuilding the spectrum extension after a source update;
`uv sync` tracks `src/quokka2s/products/_gaussian_fractions.pyx` for rebuilds.
After setup, later logins only need `cd` to this repository
and the `uv run` commands below. They reuse the existing environment and check
its dependencies automatically, without activation or repeated exports.

Only the small shell-startup and installer-configuration files remain in home;
home must still allow those small writes. The binaries, environment and caches
are stored outside home.

## 3. Set the inputs

Edit `jobname` and `dataset` in
[`configs/emission.yaml`](../configs/emission.yaml). Keep `results_dir: ../output`:

```yaml
jobname: job41101509-sigma13-run1
dataset: /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2/job41101509-sigma13/run1/plt0008703
results_dir: ../output
```

`../output` is relative to the YAML's `configs/` directory, so results and figures
are saved in `$MYSOFTWARE/quokka_postprocess_3D/output/`. No separate scratch
output directory or symbolic link is needed. The original simulation stays at
its existing path and is read-only.

Keep the supplied table paths. `jobname` only names the output group: use a
distinct name when switching simulation runs. `dataset` selects one complete
snapshot or its containing run directory.
YAML does not expand shell variables; relative paths start at the YAML file.
Coverage is snapshot-dependent; see
[table-input checks](usage.md#check-table-input-ranges).

## 4. Process

```bash
uv run process --x 64 66 --y 64 66
```

This setup test selects x/y indices 64 and 65, with complete z columns. Ranges
include the start and exclude the stop. CLI axes override only the corresponding
YAML ranges. For the full box, set both YAML ranges to `null` and omit these options.
Each slab prints x/y/z bounds and completed batches. Long reads and calculations
report their current stage about every ten seconds; active batches show whether
they are looking up emission or integrating spectra. No extra option is needed.
Results are saved under
`output/jobname/plt0008703/region_x64-66_y64-66/processed/`. A full box has no region
folder. Process draws figures from the saved products by default. Complete
matching results are reused after configuration, input/source identity and
product-integrity checks; incomplete, conflicting or legacy `processed/`
directories are reported without being overwritten. Legacy results remain
drawable with plot. Choose a fresh results root for a separate calculation;
see the [reuse rules](usage.md#completed-result-reuse).

To process the run's direct `plt` plus digits directories, in numeric order:

```bash
uv run process --dataset /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2/job41101509-sigma13/run1/
```

Each snapshot is saved and drawn before the next starts. The inventory is fixed
at startup and discovery does not recurse. For numerical products only, append
`--no-plot`; draw the whole output group later with `uv run plot --all`.
Ordinary per-snapshot failures are summarized while other snapshots continue;
any failure produces a nonzero exit status. Ctrl-C stops the batch. There is no
slab checkpoint for restarting an interrupted numerical calculation.

## 5. Plot

Process already draws figures by default. To redraw a saved region, use the
same selection:

```bash
uv run plot --x 64 66 --y 64 66
```

Without `--all`, YAML `dataset` must select one snapshot. If it names a run
directory, use `plot --all` for the output group, or supply
`--dataset /path/to/run1/plt0008703` to redraw one snapshot. Plot does not expand
raw run directories.

To draw every saved snapshot after a numerical-only batch, repeat any x/y
overrides used during processing:

```bash
uv run plot --all --x 64 66 --y 64 66
```

This reads the current `results_dir/jobname/` only; the original run directory
and tables need not exist. `--all` cannot be combined with explicit `--dataset`.

Plot prints its progress and reads only saved products, without opening the
simulation or querying tables. By default it saves only titled PNG figures
beside `processed/`, under `figures_titled/png/`. Set `figure_formats` and
`figure_styles` in the shared YAML, or choose output with command-line flags
accepted by both plot and process:

```bash
uv run plot --x 64 66 --y 64 66 --pdf
uv run plot --x 64 66 --y 64 66 --png --pdf --titled --untitled
```

`--pdf` selects PDF only; combining `--png --pdf` selects both formats.
`--untitled` selects the manuscript version in `figures/`; combining
`--titled --untitled` selects both styles. Explicit flags override their YAML
category. Each spectral figure has ±50 and ±200 km/s views, including separate
intrinsic and attenuated all-lines comparisons normalized by each line's
saved full-window peak. These plots use the existing products; no processing
rerun is needed. See the [plot settings](usage.md#plot-settings) for examples.

To change snapshots without editing `dataset`, both commands accept an absolute
`--dataset` path, for example:

```bash
uv run process --dataset /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2/job41101509-sigma13/run1/plt0001450 --x 64 66 --y 64 66
uv run plot --dataset /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2/job41101509-sigma13/run1/plt0001450 --x 64 66 --y 64 66
```

`~/...` also works for CLI `--dataset`; relative CLI paths are rejected. Other
settings still come from YAML, selected with optional `--config PATH`. Process
saves the effective configuration in its report. For local plotting, copy the
complete job/snapshot/region results structure and set the local `results_dir`;
plot uses only the dataset folder name, so the original snapshot and tables
are not needed. See the [README](../README.md) for local setup and the
[usage guide](usage.md) for additional options.

## 6. Commit code and results

`output/` is tracked by Git, including numerical arrays and figures. On Setonix,
commit and push the intended code, configuration and output files normally:

```bash
git add src configs/emission.yaml output
git commit -m "Update code and results"
git push origin main
```

A local `git clone` includes the same output directory structure and saved files.
For an existing local checkout, use `git pull` to receive committed updates.
