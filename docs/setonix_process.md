# Process on Setonix, plot on the laptop

Setonix runs `quokka2s.process_snapshot` with the existing emission tables.
The laptop runs `quokka2s.plot_emission_results` on the saved products.
The processing calculation is the same on both machines.

## 1. Get the code and install Python dependencies

Clone the current branch on Setonix:

```bash
git clone --branch codex/cloudy-emission-backup \
  https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

Use a Python 3.11 environment on Setonix. If you use Pawsey's modules,
`module spider python` lists the available versions; load the appropriate
CPU Python module before creating the environment. Do not copy a macOS
environment to Setonix.

With Python 3.11 active, create and install the environment once:

```bash
python -m venv "${MYSOFTWARE}/venvs/quokka2s-py311"
source "${MYSOFTWARE}/venvs/quokka2s-py311/bin/activate"
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` pins the validated numerical-library versions and the
exact yt source commit. DESPOTIC itself, a Cloudy executable and RADMC-3D
are not required for processing these prebuilt tables. The dust table is
already included in `vendor/draine/` in Git.

Keep the checkout: the editable installation and dust file use it directly.
On later logins, load the same Python module and activate this environment
before submitting the job.

## 2. Put the snapshot and tables in place

Transfer these three inputs separately from Git:

```text
inputs/
  snapshots/plt0655228/                 complete snapshot directory
  tables/despotic/interpolated.npz      filled DESPOTIC lookup table
  tables/cloudy/emission.npz            Cloudy lookup table
```

Copy the entire snapshot directory, including `Header`, `metadata.yaml`
and its data subdirectories. Processing does not read the raw DESPOTIC
table, table-builder checkpoints or old intermediate caches.

Place large inputs and run outputs on Setonix scratch storage. The YAML
can instead point directly to their existing scratch paths, so a second
copy under `inputs/` is unnecessary.

## 3. Choose paths in the process YAML

Edit [configs/emission_process.yaml](../configs/emission_process.yaml):

```yaml
dataset: ../inputs/snapshots/plt0655228
despotic_table: ../inputs/tables/despotic/interpolated.npz
cloudy_table: ../inputs/tables/cloudy/emission.npz
query_chunk: 1000000
chunk_workers: 2
spectral_workers: 3
output_dir: ../output/plt0655228/processed
```

Relative paths are resolved from the YAML's directory. Absolute paths also
work; write them explicitly rather than using `$MYSCRATCH` or `~` inside
YAML. Choose an output directory that does not already exist.

The supplied configuration uses two concurrent query batches and up to
three spectral threads per batch. This is one Python process with up to
six spectral threads, not eight separate Python workers.

## 4. Submit a CPU job

From the repository root, with the Python environment active:

```bash
mkdir -p output/slurm
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION \
  tools/setonix/run_process.sbatch configs/emission_process.yaml
```

Replace the two capitalized values with the account and CPU partition you
use on Setonix. The script deliberately leaves these account-specific
settings to the submission command.

The [script](../tools/setonix/run_process.sbatch) requests one node, one
task, eight CPUs, 16 GiB of memory and one hour. These are starting resource
settings, not a measured Setonix runtime estimate. It inherits the active
Python environment and writes progress to `output/slurm/quokka2s-process-JOBID.log`.
The log directory must exist before submission.

For the first Setonix run, you can make a copy of the YAML with
`max_slabs: 1` and a different `output_dir`, then pass that file to the
same script. It reads 4,194,304 cells for this snapshot and intentionally
saves a diagnostic subset. Remove `max_slabs` for the complete snapshot.

View the job and log using the ID printed by `sbatch`:

```bash
squeue -j JOBID
tail -f output/slurm/quokka2s-process-JOBID.log
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

The full run must finish with exit code zero and `status: completed` in
its `status.json`. A one-slab diagnostic has status `partial diagnostic`
and `full_snapshot: false` instead. Existing luminosity checks run inside
processing before the results are saved.

An interrupted emission run starts from the beginning in a new output
directory. The table-builder resume mechanism is unrelated to this command.

## 5. Download the products and plot

Download the completed output directory containing:

```text
images.npz
spectra.npz
phase_velocity.npz
emission_report.json
status.json
```

On the laptop, set `products` in
[configs/emission_plot.yaml](../configs/emission_plot.yaml) to that directory:

```bash
python -m quokka2s.plot_emission_results --config configs/emission_plot.yaml
```

Plotting does not reopen the snapshot or lookup tables. For an intentionally
partial diagnostic only, add `allow_partial: true` to the plot YAML.

## Official references

- [Pawsey: CPU batch scripts](https://pawsey.atlassian.net/wiki/spaces/US/pages/51927426)
- [Pawsey: software environment and storage paths](https://pawsey.atlassian.net/wiki/spaces/US/pages/51929054)
- [Pawsey: current Setonix known issues](https://pawsey.atlassian.net/wiki/spaces/US/pages/51929082)
- [Slurm: sbatch options and environment inheritance](https://slurm.schedmd.com/sbatch.html)

The script uses `--ntasks=1` without `--ntasks-per-node`, and uses the
shared-node distribution setting recommended in Pawsey's known-issues page.
