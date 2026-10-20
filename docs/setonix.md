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

Keep the supplied table paths. `jobname` only names the output group: change it
manually when switching to another job. `dataset` selects one complete snapshot.
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
folder. Existing `processed/` directories are not overwritten; choose a fresh
results root for a separate run.

## 5. Plot

After process finishes, use the same selection:

```bash
uv run plot --x 64 66 --y 64 66
```

Plot prints its progress and reads only saved products, without opening the
simulation or querying tables. Figures are saved beside `processed/` under
`figures/` and `figures_titled/`, each with `png/` and `pdf/` directories.
Set `titled_figures: false` to save only the manuscript version.

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
