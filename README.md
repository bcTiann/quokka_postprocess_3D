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
```

Open a new terminal after installation so the installer's PATH setup takes
effect, return to the repository, then run:

```bash
uv sync
```

uv creates `.venv` and installs this checkout in editable mode. No activation,
`PYTHONPATH` or manual `pip install` is needed. The locked environment uses
Python 3.11; the first yt build needs Git and C/C++ build tools. Both process
and plot dependencies are installed by default. `uv run` checks the environment
automatically and reuses the locked versions; it does not reinstall everything
on each run.

On Setonix, follow its [environment setup](docs/setonix.md#2-configure-the-environment)
instead: uv and its caches belong under `$MYSOFTWARE` to avoid the home quota.

## 3. Set the inputs

In [configs/emission.yaml](configs/emission.yaml), set the common snapshot root
and your writable output root once:

```yaml
source_root: /path/to/simulations
results_dir: ../output
```

Choose an individual snapshot with its complete `JOB/RUN/PLT` path below that
root, for example `job41101509-sigma13/run1/plt0008703`. Job numbers and run names
remain explicit; there is no sigma-to-job alias. Relative YAML paths start at
the YAML file. Shell variables such as `$MYSCRATCH` are not expanded there.

The YAML selects the bundled expanded DESPOTIC, Cloudy and dust tables; no
table building is required. Coverage is snapshot-dependent; see
[table-input checks](docs/usage.md#check-table-input-ranges).

## 4. Process

From the repository root:

```bash
uv run process job41101509-sigma13/run1/plt0008703 --x 64 66 --y 64 66
```

This small region selects x/y indices 64 and 65, with complete z columns.
Ranges include the start and exclude the stop. Omit `--x` and `--y` to select
the full box when supplying `JOB/RUN/PLT`.

Process prints each slab's x/y/z bounds, reads the snapshot and tables without
modifying them, and saves under
`results_dir/JOB/RUN/PLT/region_x64-66_y64-66/processed/`. A full box has no region
folder. Process refuses an existing `processed/` directory; use a fresh results
root for a separate run.

## 5. Plot

After process finishes, use the same selection:

```bash
uv run plot job41101509-sigma13/run1/plt0008703 --x 64 66 --y 64 66
```

Plot prints its progress and reads only saved numerical products, without
opening the snapshot or emission tables. It saves `figures/` and
`figures_titled/` beside `processed/`, each with `png/` and `pdf/` directories.
Set `titled_figures: false` to save only the manuscript version.

Both commands accept `--config /path/to/emission.yaml`. The original YAML-only
commands, `uv run process` and `uv run plot`, still use `dataset`, `jobname` and
the YAML x/y ranges. With `JOB/RUN/PLT`, the command selects the snapshot and
region instead; tables, workers and output root still come from YAML. Process
records the effective settings in `emission_report.json`.

[Detailed usage](docs/usage.md) · [Methods](docs/emission_method.md) ·
[Code guide](docs/code_walkthrough.md) · [Repository layout](docs/repository_layout.md)
