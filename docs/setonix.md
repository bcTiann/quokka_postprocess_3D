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
Keep machine-specific settings in ignored `runtime/` files so updates do not
conflict with edits to the supplied configurations.

## 2. Configure the environment

From the repository root on Setonix, create the Python environment under
`$MYSOFTWARE`. Installation temporary files go under `$MYSCRATCH`:

```bash
module avail python
module load python/3.11.6
python -m venv .venv
source .venv/bin/activate

mkdir -p "$MYSCRATCH/quokka2s_tmp"
export TMPDIR="$MYSCRATCH/quokka2s_tmp"

python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir -e .
```

Python 3.11.6 was used for the Setonix setup. If the available modules change,
select an available Python 3.11 module and use it for both installation and
running the pipeline. Both pip commands are needed: `requirements.txt` supplies
the pinned yt version with the QUOKKA reader, and `-e .` installs this
repository's package.
See [Pawsey's Python installation guide](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925902/Installing+Python+Packages).

For later sessions, return to the repository, load the same Python module, and
run `source .venv/bin/activate`. Installation only needs to be done once.

## 3. Set the inputs

Use these locations on Setonix:

| Input | Location | Provided by |
|---|---|---|
| Simulation snapshot | `$MYSCRATCH/quokka_postprocess_3D/inputs/snapshots/plt0655228/` | The user |
| DESPOTIC table | `$MYSOFTWARE/quokka_postprocess_3D/inputs/tables/despotic/interpolated.npz` | Included in Git |
| Cloudy table | `$MYSOFTWARE/quokka_postprocess_3D/inputs/tables/cloudy/emission.npz` | Included in Git |

The snapshot directory must contain `Header`, `metadata.yaml`, and all data
subdirectories. The processed results also go under `$MYSCRATCH`.

### Upload the snapshot, if needed

If the snapshot is already on Setonix, use its existing path in the process
configuration below. Otherwise, upload it from your laptop.

On Setonix, create the destination:

```bash
mkdir -p "$MYSCRATCH/quokka_postprocess_3D/inputs/snapshots/plt0655228"
```

On your laptop, set your username, Setonix scratch path, and local snapshot
directory:

```bash
setonix_user=YOUR_USERNAME
setonix_scratch=/scratch/YOUR_PROJECT/YOUR_USERNAME
local_snapshot=/absolute/path/to/plt0655228

rsync -rvhL --progress --partial --chmod=Dg+s \
  --exclude='.DS_Store' \
  "${local_snapshot}/" \
  "${setonix_user}@data-mover.pawsey.org.au:${setonix_scratch}/quokka_postprocess_3D/inputs/snapshots/plt0655228/"
```

The trailing slash copies the snapshot's contents; `-L` follows a local
snapshot symlink. If your SSH setup needs a specific key, add
`-e 'ssh -i /path/to/private_key'` to rsync. Transfers use
[Pawsey's data-mover service](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925882/Transferring+Files+in+out+Pawsey+Filesystems).

Wait for rsync to finish without errors. If interrupted, repeat the same
command. The uploaded directory must contain `Header`, `metadata.yaml`, and
all data subdirectories; an existing `Header` alone does not prove completion.

### Configure the input and output paths

On Setonix, from the repository root:

```bash
mkdir -p runtime

cat > runtime/emission_process_setonix.yaml <<EOF
dataset: $MYSCRATCH/quokka_postprocess_3D/inputs/snapshots/plt0655228
despotic_table: $MYSOFTWARE/quokka_postprocess_3D/inputs/tables/despotic/interpolated.npz
cloudy_table: $MYSOFTWARE/quokka_postprocess_3D/inputs/tables/cloudy/emission.npz
output_dir: $MYSCRATCH/quokka_postprocess_3D/output/plt0655228/processed
query_chunk: 1000000
chunk_workers: 2
spectral_workers: 3
EOF
```

The unquoted `EOF` lets the shell write actual absolute paths into the YAML.
The YAML reader itself does not expand `$VARIABLE` expressions. Relative YAML
paths are resolved from the configuration file's directory.

This configuration processes the whole snapshot. Use a **new output directory
for each run**, including retries after interruption. See
[processing settings](usage.md#processing-settings).

## 4. Process

With the Setonix Python environment activated:

```bash
python -m quokka2s.process_snapshot \
  --config runtime/emission_process_setonix.yaml
```

Results are saved under
`$MYSCRATCH/quokka_postprocess_3D/output/plt0655228/processed/`. Check completion:

```bash
cat "$MYSCRATCH/quokka_postprocess_3D/output/plt0655228/processed/status.json"
```

A completed full-box run has `status: completed`, `processing_complete: true`,
and `full_snapshot: true`.

## 5. Plot

Choose either location below.

### Option A: Plot on Setonix

Create a plot configuration pointing to the saved scratch results:

```bash
cat > runtime/emission_plot_setonix.yaml <<EOF
products: $MYSCRATCH/quokka_postprocess_3D/output/plt0655228/processed
output_dir: $MYSCRATCH/quokka_postprocess_3D/output/plt0655228/figures
titled_output_dir: $MYSCRATCH/quokka_postprocess_3D/output/plt0655228/figures_titled
EOF

python -m quokka2s.plot_emission_results \
  --config runtime/emission_plot_setonix.yaml
```

Use the same activated environment as process. Figures are saved under
`$MYSCRATCH/quokka_postprocess_3D/output/plt0655228/figures/` and
`figures_titled/`, each with `png/` and `pdf/` subdirectories.

### Option B: Download the results and plot locally

On your laptop, follow the [README](../README.md#1-clone) to clone the same
repository version and configure the local environment. From the local
repository root, download the entire `processed/` directory:

```bash
setonix_user=YOUR_USERNAME
setonix_scratch=/scratch/YOUR_PROJECT/YOUR_USERNAME
mkdir -p output/plt0655228_setonix/processed

rsync -rvh --progress --partial \
  "${setonix_user}@data-mover.pawsey.org.au:${setonix_scratch}/quokka_postprocess_3D/output/plt0655228/processed/" \
  output/plt0655228_setonix/processed/

mkdir -p runtime
cat > runtime/emission_plot_local.yaml <<'EOF'
products: ../output/plt0655228_setonix/processed
output_dir: ../output/plt0655228_setonix/figures
titled_output_dir: ../output/plt0655228_setonix/figures_titled
EOF

conda activate quokka2s
python -m quokka2s.plot_emission_results \
  --config runtime/emission_plot_local.yaml
```

Wait for the download to finish successfully before plotting. If you used a
different process output directory, update the download source accordingly.
Only the processed results need to be downloaded; do not copy the Setonix
Python environment. Plot reads saved results without rerunning process.

Figures are saved under `output/plt0655228_setonix/figures/` and
`figures_titled/`, each with separate `png/` and `pdf/` directories.
Transfer results to local or long-term storage after processing: scratch is
temporary storage and is subject to Pawsey's purge policy.
