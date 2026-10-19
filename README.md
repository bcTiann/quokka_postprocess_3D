# quokka2s — QUOKKA emission post-processing

Process one QUOKKA snapshot, then draw emission-line images and spectra from
the saved results. On Setonix, use the short [Setonix guide](docs/setonix.md).

## 1. Clone

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

## 2. Configure the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once on
your local computer, then install the locked project environment:

```bash
curl -LsSf https://astral.sh/uv/0.11.31/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked --extra process
```

uv creates `.venv` and installs this checkout in editable mode. No activation,
`PYTHONPATH` or manual `pip install` is needed. The locked environment uses
Python 3.11; the first yt build needs Git and C/C++ build tools. Repeat sync
after an update changes `uv.lock` or the project metadata.

On Setonix, follow its [environment setup](docs/setonix.md#2-configure-the-environment)
instead: uv and its caches belong under `$MYSOFTWARE` to avoid the home quota.

## 3. Set the inputs

Edit [configs/emission.yaml](configs/emission.yaml):

```yaml
jobname: sigma13.v2
dataset: ../inputs/snapshots/plt0655228
results_dir: ../output
```

`dataset` is the complete snapshot directory. `jobname` identifies the
simulation, and `results_dir` is your writable output root. Relative paths
start at the YAML file; absolute paths also work. Shell variables such as
`$MYSCRATCH` are not expanded inside YAML.

The YAML selects the bundled expanded DESPOTIC, Cloudy and dust tables; no
table building is required. Coverage is snapshot-dependent; see
[table-input checks](docs/usage.md#check-table-input-ranges).

For a small setup test, select a region while retaining the complete z columns:

```yaml
x_index_range: [64, 66]
y_index_range: [64, 66]
```

Set both to `null` for the full box. Keep `results_dir` unchanged: regions
automatically receive their own output folder.

## 4. Process

From the repository root:

```bash
uv run --no-sync process
```

Process prints its progress, reads the snapshot and tables without modifying
them, and saves to `results_dir/jobname/snapshot_name/processed/`. Choose a
fresh results root for a separate run; process refuses an existing `processed/`
directory. Explicit x/y ranges add an automatically named region folder.

## 5. Plot

After process finishes, keep the same YAML and run:

```bash
uv run --no-sync plot
```

Plot prints its progress and reads only saved numerical products, without
opening the snapshot or emission tables. It saves `figures/` and
`figures_titled/` beside `processed/`, each with `png/` and `pdf/` directories.
Set `titled_figures: false` to save only the manuscript version.

Both commands accept `--config /path/to/emission.yaml` for another configuration.

[Detailed usage](docs/usage.md) · [Methods](docs/emission_method.md) ·
[Code guide](docs/code_walkthrough.md) · [Repository layout](docs/repository_layout.md)
