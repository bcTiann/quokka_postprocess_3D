# Setonix processing and plotting

Run `process` on Setonix, then choose where to plot: on Setonix or on your
laptop. The emission tables and dust table are included in Git; no table
rebuilding is needed. Run Setonix commands from the repository root after
setup.

The current DESPOTIC default is the validated expanded 55 × 60 × 64
`interpolated.npz`, used in the completed full `sigma13/plt0008703` run.
The previous table is kept unchanged as `interpolated_old.npz`. Cloudy and
the dust table are unchanged. A normal clone includes all of these files;
DESPOTIC itself is not needed to process snapshots using the prebuilt table.

Before another snapshot, check physical settings and table-query coverage.
DESPOTIC still has 184 unavailable nodes, and the initial sigma13 `plt0000000`
region fails the existing Cloudy density limit. Neither is repaired by changing
the Python environment. See the [usage guide](usage.md).

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

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once if
it is not already available:

```bash
curl -LsSf https://astral.sh/uv/0.11.31/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

From the repository root, use the site's Python 3.11.6 interpreter:

```bash
module load python/3.11.6
uv sync --locked --extra process --python "$(command -v python)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

uv creates `.venv` and installs the versions in `uv.lock`. No manual environment
activation or installation of `quokka2s` is needed. Python imports the current
checkout's `src/` through `PYTHONPATH`. The processing lock preserves the
previously validated dependency versions and yt commit; the uv project commands
still need validation on Setonix.

Complete `uv sync` before submitting a compute job. Its first yt source build
needs internet access, Git and C/C++ build tools. Processing and plotting use
`uv run --no-sync`, so they do not install or resolve dependencies on compute
nodes. After pulling a changed lockfile, repeat the same sync command.

In later terminal sessions, return to the repository root and run:

```bash
module load python/3.11.6
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Then use the process and plot commands below. For plotting-only environments,
optional table-building dependencies and lock maintenance, see
[the usage guide](usage.md#source-environment-and-dependencies).

## 3. Set the inputs

Edit the first three settings in
[`configs/emission.yaml`](../configs/emission.yaml). Use `jobname` for the
simulation setup, `dataset` for one snapshot, and `results_dir` for the root
containing saved jobs and snapshots. Keep the supplied table entries:

```yaml
despotic_table: ../inputs/tables/despotic/interpolated.npz
cloudy_table: ../inputs/tables/cloudy/emission.npz
dust_opacity_table: ../vendor/draine/kext_albedo_WD_MW_3.1_60_D03.all
```

These paths start at `configs/emission.yaml`. If you copy the YAML elsewhere,
make these paths absolute or remove the three entries to use the bundled
defaults. Selecting `interpolated_old.npz` reproduces the previous table choice.

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
`metadata.yaml`, and all data subdirectories. The emission tables are
already in the clone. Process reads
the original snapshot; put results in your own writable output directory.

Example directory layout:

```text
$MYSOFTWARE/quokka_postprocess_3D/
├── .venv/
├── configs/
│   └── emission.yaml
└── inputs/tables/
    ├── despotic/interpolated.npz       # Expanded default
    ├── despotic/interpolated_old.npz   # Previous table
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

For a first setup check, use `plt0008703` or another snapshot with known
coverage, select both `[64, 66]` ranges and choose a fresh results root. Keep
the original simulation path read-only and put all results in your own scratch
directory. After process and plot succeed, change both ranges to `null` for
full-box processing. A successful 2 × 2 region is not a full-box coverage test.
YAML paths do not expand `$MYSCRATCH`, `$MYSOFTWARE` or other shell variables;
write actual absolute paths where needed.

## 4. Process

Run computational work on a CPU compute node allocated through Slurm, following
[Pawsey's job scheduling guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51926920).
Login nodes are for setup and job submission. This workflow does not need a GPU
or MPI: one Python process owns the reading and batch/spectral threads.

Save this example as `run_process.sbatch` in the repository root. Replace
`YOUR_PROJECT` with your allocation account. It runs process then plot using
the same YAML; `set -e` prevents plotting if process fails. Keep that YAML
unchanged until the job finishes.

```bash
#!/bin/bash --login
#SBATCH --account=YOUR_PROJECT
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=quokka-process

set -e
module load python/3.11.6
cd "$MYSOFTWARE/quokka_postprocess_3D"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# The YAML owns batch/spectral concurrency; keep numerical libraries at one.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

srun --ntasks=1 --cpus-per-task=8 uv run --no-sync python src/quokka2s/process_snapshot.py
srun --ntasks=1 --cpus-per-task=8 uv run --no-sync python src/quokka2s/plot_emission_results.py
```

The eight requested CPU cores allow the configured two concurrent batches,
up to three spectral threads each, and reading/orchestration. This is a starting
resource request, not evidence that more workers are faster. The six-hour
wall limit is a scheduler limit, not a runtime prediction; adjust it and the
memory request for your snapshot. The program's 3 GiB field-cache cap is not
a limit on total process memory.

Submit from the repository root, putting the log on scratch. Copy the job ID
printed by `sbatch` to inspect that job with `squeue -j JOB_ID`:

```bash
mkdir -p "$MYSCRATCH/quokka2s_logs"
sbatch --output="$MYSCRATCH/quokka2s_logs/%x-%j.log" run_process.sbatch
```

If you already have a suitable interactive CPU allocation with the same
environment and thread settings, the direct command remains:

```bash
uv run --no-sync python src/quokka2s/process_snapshot.py
```

The snapshot name is taken from the supplied `dataset` path. Process saves to
`results_dir/jobname/snapshot_name/processed/` when both x/y ranges are omitted
or `null`, or inside the automatically named region folder when a range is set.
Check its completion status using the paths in your YAML:

```bash
cat /scratch/YOUR_PROJECT/YOUR_USERNAME/quokka_postprocess_3D/output/sigma13.v2/plt0655228/processed/status.json
```

A completed full-box run has `status: completed`, `processing_complete: true`,
and `full_snapshot: true`. A completed requested region instead has
`full_snapshot: false`; it is not an unfinished process. Process refuses an
existing `processed/` directory; choose another `results_dir` to keep a separate
run. An interrupted process is not resumed by rerunning into that directory.

Changing the table does not refresh old saved products. Use a fresh results
root for new-table comparisons. After completion, review `input_tables`,
`missing_emissivity_cells`, `missing_emissivity_mass_g`, clipping and conservation
in `emission_report.json`; a zero exit status alone is not a coverage assessment.

## 5. Plot

Choose either location below.

### Option A: Plot on Setonix

Keep the same shared YAML and run:

```bash
uv run --no-sync python src/quokka2s/plot_emission_results.py
```

Use the same uv environment as process. Plot reads this job and
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
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
uv run --no-sync python src/quokka2s/plot_emission_results.py
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

## Optional installation troubleshooting

uv chooses its cache and temporary storage automatically. If installation
actually fails because the default storage is full or unavailable, choose a
writable cache with `UV_CACHE_DIR`; change `TMPDIR` only for a temporary-storage
error. These are storage workarounds, not required setup steps. The Python
environment stays in `.venv`. See [uv cache settings](https://docs.astral.sh/uv/concepts/cache/).
