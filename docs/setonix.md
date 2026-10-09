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
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

Keep the repository and Python environment under `$MYSOFTWARE`. Store the
snapshot and numerical results under `$MYSCRATCH`, following
[Pawsey's filesystem guidance](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925876).
If this clone already exists, update it with `git pull --ff-only origin main`.

## 2. Configure the environment

From the repository root on Setonix, create the Python environment under
`$MYSOFTWARE`:

```bash
module avail python
module load python/3.11.6
python -m venv .venv
source .venv/bin/activate

mkdir -p "$MYSCRATCH/quokka2s_installation_temp"
export TMPDIR="$MYSCRATCH/quokka2s_installation_temp"

python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -e .
```

`quokka2s_installation_temp/` holds temporary installation files on scratch.
[`--no-cache-dir`](https://pip.pypa.io/en/stable/topics/caching/#disabling-caching)
disables pip's persistent download/build cache. These settings concern package
installation, not simulation inputs or processed results.

## 3. Set the inputs

Edit [`configs/emission_process.yaml`](../configs/emission_process.yaml).
Keep the two bundled table paths unchanged. Set `dataset` and `output_dir`
to locations under your scratch directory.

| Setting | Meaning |
|---|---|
| `dataset` | Path to the complete simulation snapshot directory. |
| `despotic_table` | Path to the DESPOTIC table, already included in the clone. |
| `cloudy_table` | Path to the Cloudy table, already included in the clone. |
| `output_dir` | Directory where process saves its numerical results. Use a new directory for each run. |
| `query_chunk` | Maximum number of cells per query batch. |
| `chunk_workers` | Number of batches processed concurrently. |
| `spectral_workers` | Number of spectral-integration threads per batch. |

Replace `/path/to/plt0655228` with your complete snapshot directory and
`/path/to/processed` with a new directory for the numerical results:

```yaml
dataset: /path/to/plt0655228
despotic_table: ../inputs/tables/despotic/interpolated.npz
cloudy_table: ../inputs/tables/cloudy/emission.npz
output_dir: /path/to/processed
query_chunk: 1000000
chunk_workers: 2
spectral_workers: 3
```

Place the snapshot at the path set by `dataset`, or point it to the snapshot's
existing location on scratch. The directory must contain `Header`,
`metadata.yaml`, and all data subdirectories. The two emission tables are
already in the clone; `../inputs/` resolves from `configs/` to the repository's
`inputs/` directory.

Example directory layout:

```text
$MYSOFTWARE/quokka_postprocess_3D/
├── .venv/
├── configs/
│   ├── emission_process.yaml
│   └── emission_plot.yaml
└── inputs/tables/
    ├── despotic/interpolated.npz
    └── cloudy/emission.npz

$MYSCRATCH/quokka_postprocess_3D/
├── inputs/snapshots/plt0655228/
│   ├── Header
│   ├── metadata.yaml
│   └── Level_0/
└── output/plt0655228/
    ├── processed/       # Numerical results
    ├── figures/         # Paper figures
    └── figures_titled/  # Figures with titles
```

Use actual absolute paths in YAML; `$MYSOFTWARE` and `$MYSCRATCH` above only
show the storage locations. YAML does not expand environment variables.

This configuration processes the whole snapshot. For optional settings such
as an x–y region, see [processing settings](usage.md#processing-settings).

## 4. Process

With the Setonix Python environment activated:

```bash
python -m quokka2s.process_snapshot
```

Results are saved at `output_dir`. Check its completion status, replacing
`/path/to/processed` with the same path:

```bash
cat /path/to/processed/status.json
```

A completed full-box run has `status: completed`, `processing_complete: true`,
and `full_snapshot: true`.

## 5. Plot

Choose either location below.

### Option A: Plot on Setonix

Edit [`configs/emission_plot.yaml`](../configs/emission_plot.yaml), using your
actual paths. `products` must match the process configuration's `output_dir`;
the other two paths choose where the figures are saved:

```yaml
products: /path/to/processed
output_dir: /path/to/figures
titled_output_dir: /path/to/figures_titled
```

Then run:

```bash
python -m quokka2s.plot_emission_results
```

Use the same activated environment as process. Each figure directory contains
separate `png/` and `pdf/` subdirectories.

### Option B: Download the results and plot locally

On your laptop, follow the [README](../README.md#1-clone) to clone the same
repository version and configure the local environment. From the local
repository root, download the entire `processed/` directory. Replace
`YOUR_USERNAME` with your Pawsey username and `/path/to/processed` with your
Setonix process output directory:

```bash
mkdir -p output/plt0655228_setonix/processed

rsync -rvh --progress --partial \
  "YOUR_USERNAME@data-mover.pawsey.org.au:/path/to/processed/" \
  output/plt0655228_setonix/processed/
```

In your local clone, edit
[`configs/emission_plot.yaml`](../configs/emission_plot.yaml) to read the
downloaded results:

```yaml
products: ../output/plt0655228_setonix/processed
output_dir: ../output/plt0655228_setonix/figures
titled_output_dir: ../output/plt0655228_setonix/figures_titled
```

Then run:

```bash
conda activate quokka2s
python -m quokka2s.plot_emission_results
```

Wait for the download to finish successfully before plotting. If you used a
different process output directory, update the download source accordingly.
Only the processed results need to be downloaded; do not copy the Setonix
Python environment. Plot reads saved results without rerunning process.

Figures are saved under `output/plt0655228_setonix/figures/` and
`figures_titled/`, each with separate `png/` and `pdf/` directories.
Transfer results to local or long-term storage after processing: scratch is
temporary storage and is subject to Pawsey's purge policy.
