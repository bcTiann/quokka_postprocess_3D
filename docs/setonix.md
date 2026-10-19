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

Install uv once and keep its binaries, environment and caches outside home:

```bash
curl -LsSf https://astral.sh/uv/0.11.31/install.sh |
  env UV_UNMANAGED_INSTALL="$MYSOFTWARE/uv/bin" sh
export PATH="$MYSOFTWARE/uv/bin:$PATH"
export UV_CACHE_DIR="$MYSOFTWARE/uv/cache"
export MPLCONFIGDIR="$MYSOFTWARE/uv/matplotlib"
module load python/3.11.6
uv sync --locked --extra process --python "$(command -v python)"
```

uv creates `.venv` and installs the checkout in editable mode. No activation,
`PYTHONPATH` or manual `pip install` is needed. The first yt build needs Git and
C/C++ build tools. Repeat sync after `uv.lock` or project metadata changes.

Optionally save the three environment exports in your shell startup file;
in a new terminal, load the site Python module and reuse these setup paths.

## 3. Set the inputs

Edit [`configs/emission.yaml`](../configs/emission.yaml), using your actual
absolute paths:

```yaml
jobname: sigma13.v2
dataset: /scratch/YOUR_PROJECT/YOUR_USERNAME/simulations/sigma13.v2/plt0008703
results_dir: /scratch/YOUR_PROJECT/YOUR_USERNAME/quokka_results
x_index_range: null
y_index_range: null
```

Keep the supplied table paths. `dataset` must point to the complete snapshot
including `Header`, `metadata.yaml` and its data directories. The simulation
is read-only; results go in your own writable directory. YAML does not expand
`$MYSCRATCH` or other shell variables. Relative paths start at the YAML file.

For a small setup test, set both x/y ranges to `[64, 66]`; z remains complete.
Use `null` for the full box. Coverage is snapshot-dependent; see
[table-input checks](usage.md#check-table-input-ranges).

## 4. Process

```bash
uv run --no-sync process
```

Process prints its progress and saves to
`results_dir/jobname/snapshot_name/processed/`. Explicit x/y ranges add a region
folder automatically. An existing `processed/` directory is not overwritten;
choose a fresh results root for a separate run.

## 5. Plot

After process finishes, keep the same YAML and run:

```bash
uv run --no-sync plot
```

Plot prints its progress and reads only saved products, without opening the
simulation or querying tables. Figures are saved beside `processed/` under
`figures/` and `figures_titled/`, each with `png/` and `pdf/` directories.
Set `titled_figures: false` to save only the manuscript version.

Both commands accept `--config /path/to/emission.yaml`. For local plotting,
copy the complete job/snapshot/region results structure and set the local
`results_dir`; the original snapshot is not needed. See the [README](../README.md)
for local setup and [usage guide](usage.md) for additional options.
