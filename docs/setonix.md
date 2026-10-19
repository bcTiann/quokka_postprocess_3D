# Setonix processing and plotting

Run `process` on Setonix, then choose where to plot: on Setonix or on your
laptop. The emission tables and dust table are included in Git; no table
rebuilding is needed. Run Setonix commands from the repository root after
setup.

The bundled emission tables target the `plt0655228` reference snapshot.
Before using another snapshot, check that its physical settings and table-query
coordinates are compatible with these tables. See the [usage guide](usage.md).

## 1. Clone

On Setonix:

```bash
cd "$MYSOFTWARE"
git clone --branch codex/cloudy-emission-backup https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

Keep the repository and Python environment under `$MYSOFTWARE`. Store the
snapshot and numerical results under `$MYSCRATCH`, following
[Pawsey's filesystem guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925876).
These instructions follow the `codex/cloudy-emission-backup` branch. For an
existing clone, select that branch before updating:

```bash
git switch codex/cloudy-emission-backup
git pull --ff-only
```

## 2. Configure the environment

With [uv](https://docs.astral.sh/uv/getting-started/installation/) available
on `PATH`, create the environment under `$MYSOFTWARE` from the repository root.
Use the module's Python 3.11 interpreter. This environment setup has been
checked on Setonix with Python 3.11.6. Each snapshot must also satisfy the
emission tables' coordinate limits:

```bash
module avail python
module load python/3.11.6

mkdir -p "$MYSCRATCH/quokka2s_uv_cache" "$MYSCRATCH/quokka2s_installation_temp"
export UV_CACHE_DIR="$MYSCRATCH/quokka2s_uv_cache"
export TMPDIR="$MYSCRATCH/quokka2s_installation_temp"

uv venv --python "$(command -v python)"
source .venv/bin/activate
uv pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

`UV_CACHE_DIR` keeps reusable downloads and builds on scratch; `TMPDIR` holds
installation working files. These settings concern package installation, not
simulation inputs or processed results. The locked yt dependency is built from
a Git commit, so its first source build needs Git and C/C++ build tools in the
module environment. Cached builds can be reused.

The project itself is not installed: commands read the current `src/` through
`PYTHONPATH`. In each later terminal session or batch-job shell, return to the
repository root and activate that same environment:

```bash
module load python/3.11.6
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Repeat the scratch-cache exports and `uv pip install -r requirements.txt` only
when installing an updated dependency lock. Source changes are used directly
on the next run. `pyproject.toml` maintains the direct dependencies; the lock
also includes their transitive dependencies. See
[dependency setup and maintenance](usage.md#source-environment-and-dependencies)
for saved-products-only installation and intentional lock updates.

## 3. Set the inputs

Edit the first three settings in
[`configs/emission.yaml`](../configs/emission.yaml). Use `jobname` for the
simulation setup, `dataset` for one snapshot, and `results_dir` for the root
containing saved jobs and snapshots. The bundled tables are used by default.

| Setting | Meaning |
|---|---|
| `jobname` | Simulation setup, for example `sigma13.v2` or `sigma50`. |
| `dataset` | Path to the complete simulation snapshot directory. |
| `results_dir` | Root directory for all jobs' numerical results and figures. |
| `query_chunk` | Maximum number of cells per query batch. |
| `chunk_workers` | Number of batches processed concurrently. |
| `spectral_workers` | Number of spectral-integration threads per batch. |

Use actual absolute paths for your project and username. For example:

```yaml
jobname: sigma13.v2
dataset: /scratch/YOUR_PROJECT/YOUR_USERNAME/simulations/sigma13.v2/plt0655228
results_dir: /scratch/YOUR_PROJECT/YOUR_USERNAME/quokka_postprocess_3D/output
query_chunk: 1000000
chunk_workers: 2
spectral_workers: 3
```

Place the snapshot at the path set by `dataset`, or point it to the snapshot's
existing location on scratch. The directory must contain `Header`,
`metadata.yaml`, and all data subdirectories. The two emission tables are
already in the clone and need no YAML entries. Advanced configurations can
override their paths with `despotic_table` and `cloudy_table`. Process reads
the original snapshot; put results in your own writable output directory.

Example directory layout:

```text
$MYSOFTWARE/quokka_postprocess_3D/
├── .venv/
├── configs/
│   └── emission.yaml
└── inputs/tables/
    ├── despotic/interpolated.npz
    └── cloudy/emission.npz

$MYSCRATCH/simulations/sigma13.v2/plt0655228/
├── Header
├── metadata.yaml
└── Level_0/

$MYSCRATCH/quokka_postprocess_3D/output/
├── sigma13.v2/
│   ├── plt0655228/
│   │   ├── processed/       # Numerical results
│   │   ├── figures/         # Paper figures
│   │   └── figures_titled/  # Figures with titles
│   └── plt0916720/
└── sigma50/
    └── plt0655228/
```

Use actual absolute paths in YAML; `$MYSOFTWARE` and `$MYSCRATCH` above only
show the storage locations.

This configuration processes the whole snapshot. For optional settings such
as an x–y region, see [region settings](usage.md#select-an-x-y-region). Setting
`x_index_range: [64, 66]` and `y_index_range: [64, 66]` adds
`region_x64-66_y64-66/` under the snapshot folder automatically. Keep
`results_dir` unchanged. An omitted or `null` axis uses `all` in that folder name;
with both axes omitted or `null`, outputs stay directly under the snapshot's
result directory.

## 4. Process

With the Setonix Python environment activated:

```bash
python src/quokka2s/process_snapshot.py
```

The snapshot name is taken from the supplied `dataset` path. Process saves to
`results_dir/jobname/snapshot_name/processed/` when both x/y ranges are omitted
or `null`, or inside the automatically named region folder when a range is set.
Check its completion status using the paths in your YAML:

```bash
cat /scratch/YOUR_PROJECT/YOUR_USERNAME/quokka_postprocess_3D/output/sigma13.v2/plt0655228/processed/status.json
```

A completed full-box run has `status: completed`, `processing_complete: true`,
and `full_snapshot: true`. Process refuses an existing `processed/` directory;
choose another `results_dir` to keep a separate run.

## 5. Plot

Choose either location below.

### Option A: Plot on Setonix

Keep the same shared YAML and run:

```bash
python src/quokka2s/plot_emission_results.py
```

Use the same activated environment as process. Plot reads this job and
snapshot's selected full-box or region `processed/` directory and writes the
sibling `figures/` and `figures_titled/` directories. Each contains separate `png/` and `pdf/`
subdirectories. Set `titled_figures: false` to omit the titled version.

### Option B: Download the results and plot locally

On your laptop, follow the [README](../README.md#1-clone) to clone the same
repository version and configure the local environment. From the local
repository root, download the entire `processed/` directory. Replace
`YOUR_USERNAME` with your Pawsey username and `/path/to/processed` with your
Setonix process output directory. Preserve the job, snapshot and any region
folder names. This example downloads a full-box result:

```bash
mkdir -p output/sigma13.v2/plt0655228/processed

rsync -rvh --progress --partial \
  "YOUR_USERNAME@data-mover.pawsey.org.au:/path/to/processed/" \
  output/sigma13.v2/plt0655228/processed/
```

In your local clone, edit
[`configs/emission.yaml`](../configs/emission.yaml) to select the downloaded
job and snapshot:

```yaml
jobname: sigma13.v2
dataset: /scratch/YOUR_PROJECT/YOUR_USERNAME/simulations/sigma13.v2/plt0655228
results_dir: ../output
```

Plot uses only the snapshot name `plt0655228` from `dataset`; the original
Setonix path does not need to exist on your laptop. `results_dir` points to the
local root containing `sigma13.v2/plt0655228/processed/`.
For a region, copy into the corresponding folder, such as
`output/sigma13.v2/plt0655228/region_x64-66_y64-66/processed/`, and retain
the same x/y index ranges in the local YAML. Plot selects the region using these
settings alone; it does not read the snapshot.

Then run:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python src/quokka2s/plot_emission_results.py
```

Wait for the download to finish successfully before plotting. If you used a
different process output directory, update the download source accordingly.
Only the processed results need to be downloaded; do not copy the Setonix
Python environment. Plot reads saved results without rerunning process.

Figures are saved under `output/sigma13.v2/plt0655228/figures/` and
`figures_titled/`, each with separate `png/` and `pdf/` directories.
For a region, both figure directories are inside its region folder.
Transfer results to local or long-term storage after processing: scratch is
temporary storage and is subject to Pawsey's purge policy.
