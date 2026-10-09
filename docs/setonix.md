# Setonix processing and local plotting

Run `process` on Setonix, download the numerical results, then run `plot` on
your laptop. The two emission tables and the dust table are included in Git;
this workflow does not rebuild tables or run Cloudy or DESPOTIC themselves.

The bundled emission tables target the `plt0655228` reference snapshot.
Before using another snapshot, check that its physical settings and table-query
coordinates are compatible with these tables. See the [usage guide](usage.md).

## 1. Log in and choose storage locations

On your laptop, replace `YOUR_USERNAME` with your Pawsey username:

```bash
ssh YOUR_USERNAME@setonix.pawsey.org.au
```

On Setonix:

```bash
printenv MYSOFTWARE MYSCRATCH
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

## 2. Install the Python environment

From the repository root on Setonix:

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
select an available Python 3.11 module and use the same module in the job script
below. Both pip commands are needed: `requirements.txt` supplies the pinned yt
version with the QUOKKA reader, and `-e .` installs this repository's package.
See [Pawsey's Python installation guide](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925902/Installing+Python+Packages).

For later sessions, return to the repository, load the same Python module, and
run `source .venv/bin/activate`. Installation only needs to be done once.

## 3. Upload the complete snapshot

On Setonix, create the destination:

```bash
mkdir -p "$MYSCRATCH/quokka_postprocess_3D/inputs/snapshots/plt0655228"
```

On your laptop, set your username, the scratch path printed in step 1, and
the local snapshot directory:

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

## 4. Create the process configuration

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

Omitting x/y ranges and `max_slabs` processes the whole snapshot. These worker
settings allow up to six spectral-integration threads in one Python process.
Use a **new output directory for each run**, including retries after interruption.
Emission processing starts again from the beginning; `status.json` is not a
checkpoint. See [processing settings](usage.md#processing-settings).

## 5. Submit a CPU job

Use a CPU allocation for this pipeline. `work` is Pawsey's CPU production
partition; the account must have access to it. `normal` is a QoS name, rather
than the CPU partition used in this example. Check your account associations:

```bash
sacctmgr -P show associations where user="$USER" cluster=setonix \
  format=Account%40,Partition%20,QOS%40
sinfo -h -o '%P'
```

If CPU access is unavailable, ask your project administrator or Pawsey support
to confirm the account and partition. The example below assumes that access
is available; it does not grant access. Production processing belongs on
compute nodes, following [Pawsey's scheduling guide](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925964/Job+Scheduling).

From the repository root on Setonix, replace `YOUR_CPU_ACCOUNT` with your
eligible CPU account. Change `cpu_partition` if your allocation uses another
CPU partition:

```bash
compute_account=YOUR_CPU_ACCOUNT
cpu_partition=work
repo_dir="$MYSOFTWARE/quokka_postprocess_3D"
log_dir="$MYSCRATCH/quokka_postprocess_3D/logs"
mkdir -p "$log_dir"

cat > runtime/process_setonix.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=quokka2s-process
#SBATCH --account=$compute_account
#SBATCH --partition=$cpu_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=$log_dir/process-%j.log

set -e
cd "$repo_dir"
module load python/3.11.6
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --export=ALL --cpu-bind=cores \\
  python -u -m quokka2s.process_snapshot \\
  --config runtime/emission_process_setonix.yaml
EOF

sbatch runtime/process_setonix.sbatch
```

The shell fills the account and paths before submission. Slurm runs one Python
process with eight CPUs. The memory and time requests are starting values for
the reference snapshot, not measured optimal requests for every dataset.
The three thread variables limit extra NumPy/library threads; the configured
batch and spectral threads remain active. The environment is loaded inside
the job rather than relying on an activated login shell.

Use `squeue -u "$USER"` to inspect the job. Its output is written to
`$MYSCRATCH/quokka_postprocess_3D/logs/process-JOB_ID.log`, where `JOB_ID` is
the ID printed by `sbatch`.

## 6. Confirm completion

On Setonix:

```bash
cat "$MYSCRATCH/quokka_postprocess_3D/output/plt0655228/processed/status.json"
ls -lh "$MYSCRATCH/quokka_postprocess_3D/output/plt0655228/processed/"
```

A completed full-box run has `status: completed`, `processing_complete: true`,
and `full_snapshot: true`. Download the entire directory, including:

| File | Contents |
|---|---|
| `images.npz` | Intrinsic and dust-attenuated luminosity images |
| `spectra.npz` | Integrated spectra and line velocity moments |
| `phase_velocity.npz` | Gas-phase velocity distributions and moments |
| `emission_report.json` | Settings, counts, luminosities and calculation checks |
| `status.json` | Completion status and elapsed time |

## 7. Download results and plot locally

On your laptop, clone the same repository version and install its environment
using the [README](../README.md). From the local repository root, set the same
remote username and scratch path used for uploading:

```bash
setonix_user=YOUR_USERNAME
setonix_scratch=/scratch/YOUR_PROJECT/YOUR_USERNAME
mkdir -p output/plt0655228_setonix/processed

rsync -rvh --progress --partial \
  "${setonix_user}@data-mover.pawsey.org.au:${setonix_scratch}/quokka_postprocess_3D/output/plt0655228/processed/" \
  output/plt0655228_setonix/processed/

mkdir -p runtime
cat > runtime/emission_plot_setonix.yaml <<'EOF'
products: ../output/plt0655228_setonix/processed
output_dir: ../output/plt0655228_setonix/figures
titled_output_dir: ../output/plt0655228_setonix/figures_titled
EOF

conda activate quokka2s
python -m quokka2s.plot_emission_results \
  --config runtime/emission_plot_setonix.yaml
```

Wait for the download to finish successfully before plotting. If you used a
different process output directory, update the download source accordingly.
Plot reads the saved results; no snapshot, table queries, or new process run
is needed. Use the laptop's own environment, rather than copying Setonix's
`.venv` between machines.

Figures are saved under `output/plt0655228_setonix/figures/` and
`figures_titled/`, each with separate `png/` and `pdf/` directories.
Transfer results to local or long-term storage after processing: scratch is
temporary storage and is subject to Pawsey's purge policy.
