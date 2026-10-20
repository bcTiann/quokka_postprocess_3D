# Setonix: process and plot with uv

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
`PYTHONPATH` or manual `pip install` is needed. The first yt build needs Git and
C/C++ build tools. After setup, later logins only need `cd` to this repository
and the `uv run` commands below. They reuse the existing environment and check
its dependencies automatically; no activation, repeated exports or Python
module loading is needed for this validated environment.

Only the small shell-startup and installer-configuration files remain in home;
home must still allow those small writes. The binaries, environment and caches
are stored outside home.

## 3. Set the inputs

Edit [`configs/emission.yaml`](../configs/emission.yaml) once, using the common
simulation root and your own writable output root:

```yaml
source_root: /scratch/pawsey0807/chongchong/Projects-data/2026-quokka-outflow/hpc/run2.v2
results_dir: /scratch/YOUR_PROJECT/YOUR_USERNAME/quokka_results
```

Keep the supplied table paths. These job/run directories live below that root:

```text
job41101509-sigma13/run1
job41114622-sigma13-zmet0.2/run1
job41114623-sigma50/run1
job44195690-sigma2.5-2pc/run1
```

Select one exact `JOB/RUN/PLT` path per command; no job alias or wildcard is used.
Original data stays read-only. YAML does not expand `$MYSCRATCH` or other shell
variables; relative paths start at the YAML file. Coverage is snapshot-dependent; see
[table-input checks](usage.md#check-table-input-ranges).

## 4. Process

```bash
uv run process job41101509-sigma13/run1/plt0008703 --x 64 66 --y 64 66
```

This setup test selects x/y indices 64 and 65, with complete z columns. Ranges
include the start and exclude the stop. Omit `--x` and `--y` for the full box.
Each slab prints x/y/z bounds. Results retain the complete identity:
`results_dir/JOB/RUN/PLT/region_x64-66_y64-66/processed/`. A full box has no region
folder. Existing `processed/` directories are not overwritten; choose a fresh
results root for a separate run.

## 5. Plot

After process finishes, use the same selection:

```bash
uv run plot job41101509-sigma13/run1/plt0008703 --x 64 66 --y 64 66
```

Plot prints its progress and reads only saved products, without opening the
simulation or querying tables. Figures are saved beside `processed/` under
`figures/` and `figures_titled/`, each with `png/` and `pdf/` directories.
Set `titled_figures: false` to save only the manuscript version.

Both commands accept `--config /path/to/emission.yaml`. The original YAML-only
`uv run process` and `uv run plot` still use `dataset`, `jobname` and the YAML
x/y ranges. Supplying `JOB/RUN/PLT` replaces that selection; omitted CLI axes
then mean full axes. Process saves the effective configuration in its report.
For local plotting, copy the complete job/run/snapshot/region structure and set the local
`results_dir`; the original snapshot is not needed. See the [README](../README.md)
for local setup and [usage guide](usage.md) for additional options.
