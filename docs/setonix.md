# Setonix: process and plot with uv

Keep the repository, environment and generated results under `$MYSOFTWARE`.
Run from the repository root. Bundled tables need no separate DESPOTIC installation.

Pawsey's [login-node guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51926034/How%2Bto%2Blog%2Binto%2BSetonix)
reserves the shared login nodes for access and preparation. Use them for SSH,
Git updates, file editing and lightweight header inspection. Run processing,
plotting and other computational workloads inside a Slurm compute-node
allocation. The processing commands below assume that allocation is active.

## 1. Clone

```bash
cd "$MYSOFTWARE"
git clone --branch codex/cython-optimized https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

To update this branch: `git switch codex/cython-optimized`, then
`git pull --ff-only origin codex/cython-optimized`.

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
`PYTHONPATH` or manual `pip install` is needed. Project calculations use
NumPy/SciPy and small Cython numerical extensions. `uv sync` builds the project
extensions automatically using a C compiler. The pinned yt source dependency
can still need Git and C/C++ build tools when no cached build is available;
uv installs its build dependencies automatically. Editing project Python files
does not rebuild processing kernels.
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
snapshot. A containing run directory is rejected.
YAML does not expand shell variables; relative paths start at the YAML file.
Coverage is snapshot-dependent; see
[table-input checks](usage.md#check-table-input-ranges).

## 4. Process one explicit snapshot

From the repository root in your processing environment:

```bash
uv run python process_snapshot.py \
  --dataset /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2/job41101509-sigma13/run1/plt0008703 \
  --jobname job41101509-sigma13-run1 --x 64 66 --y 64 66
```

This reads four full-z columns from intersecting original FABs and required
global gradient neighbours. It never creates a z-contiguous disk copy. Short
edge blocks and dimensions such as 1024 x 1024 x 2028 use their metadata sizes.
Remove x/y arguments, with null YAML ranges, to process the complete snapshot.
A parent run directory is an error; the script processes only the named plt.

Only numerical products are calculated, checked and saved. Results use
`output/jobname/snapshot_name/processed/`, with a region directory for explicit
x/y ranges. Default memory planning and RSS guards stay within a 30 GB limit
or stop with an actionable error; reduce batch/worker settings when needed.
Original simulation files and lookup tables remain read-only.

The sole Gaussian route accumulates weighted luminosity at four stored template
corners and forms 400-bin spectra using NumPy convolution. Sigma nodes stay
logarithmically spaced with linear sigma weights. Valid widths outside table
coverage use stable direct integration and are logged. Only template preparation
is excluded from process time; original reads, all numerical work, checks and
saving remain included. Numerical extensions are built during environment
synchronization, before snapshot processing; Python grids and physical settings
can change without recompilation.

## 5. Plot one saved result

Run the independent script with the same snapshot and region selection:

```bash
uv run python plot_emission_results.py \
  --dataset /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2/job41101509-sigma13/run1/plt0008703 \
  --jobname job41101509-sigma13-run1 --x 64 66 --y 64 66
```

Plot opens only saved numerical products. It does not access the original plt
or emission tables. The default writes titled PNGs to `figures_titled/png/`.
Add `--pdf`, or `--png --pdf --titled --untitled`, to select other saved styles.
There is no automatic plot stage or run-directory enumeration in process.
Both scripts accept `--config` for a separate settings file. Repeat any CLI
selection overrides in the later plot command; overrides do not edit YAML.

## 6. Commit code and results

The repository can track selected numerical arrays and figures under `output/`.
Review the working tree and stage only intended source, configuration and
published product paths with `git add PATH`. Keep personal validation outputs,
local tests and comparison records in ignored directories; do not stage the
whole `output/` tree.

```bash
git status --short
git diff --stat
# After staging the selected paths:
git diff --cached --stat
git commit -m "Update code and results"
git push origin codex/cython-optimized
```

A local `git clone` includes the same output directory structure and saved files.
For an existing local checkout, use `git pull` to receive committed updates.
