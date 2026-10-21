# quokka2s — QUOKKA emission post-processing

Process one QUOKKA snapshot, then draw emission-line images and spectra from
the saved results. On Setonix, use the short [Setonix guide](docs/setonix.md).

Setonix results are tracked in [output/](output/). A normal clone includes the
saved numerical products and PNG/PDF figures in their original directory structure.

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
Python 3.11. Building yt and the Cython spectrum kernel needs Git and C/C++
build tools. uv installs the pinned build dependencies automatically; no manual
Cython installation or compilation command is needed. Both process and plot
dependencies are installed by default. `uv run` checks the environment
automatically and reuses the locked versions; it does not reinstall everything
on each run.

After changing `src/quokka2s/products/_gaussian_fractions.pyx`, run `uv sync`
to rebuild the compiled spectrum kernel. The build cache also tracks this source
when `uv run` checks the environment. Generated C and compiled binaries stay local.

On Setonix, follow its [environment setup](docs/setonix.md#2-configure-the-environment)
instead: keep the repository, environment, caches and generated results under
`$MYSOFTWARE` to avoid the home quota.

## 3. Set the inputs

Edit [configs/emission.yaml](configs/emission.yaml):

```yaml
jobname: sigma13.v2
dataset: ../inputs/snapshots/plt0655228
results_dir: ../output
```

`dataset` is the complete snapshot directory. `jobname` names its output group;
change it yourself when switching jobs, since no job identity is inferred from
the input path. Keep `results_dir: ../output` to save results in the repository's
`output/` directory. On Setonix this is `$MYSOFTWARE/quokka_postprocess_3D/output/`;
no separate scratch output directory or link is needed. Relative YAML paths
start at the YAML file. Shell variables are not expanded there.

The YAML selects the bundled expanded DESPOTIC, Cloudy and dust tables; no
table building is required. Coverage is snapshot-dependent; see
[table-input checks](docs/usage.md#check-table-input-ranges).

## 4. Process

From the repository root:

```bash
uv run process
```

This uses the YAML settings. For a small setup test, run
`uv run process --x 64 66 --y 64 66`: x/y indices 64 and 65, with complete z
columns. Ranges include the start and exclude the stop. Each supplied CLI axis
overrides its YAML range; omitted axes keep their YAML settings.

Process prints each slab's x/y/z bounds and completed batches. During long
reads or calculations, timestamped status messages appear about every ten
seconds, including each active batch's current stage and elapsed time.
It reads the snapshot and tables without modifying them, and saves under
`results_dir/jobname/snapshot_name/processed/`. Explicit x/y ranges add, for
example, `region_x64-66_y64-66/` before `processed/`. A full box has no region
folder. Process refuses an existing `processed/` directory; use a fresh results
root for a separate run.

## 5. Plot

After process finishes, use the same selection:

```bash
uv run plot
```

For the small region above, use `uv run plot --x 64 66 --y 64 66` instead.
Plot prints its progress and reads only saved numerical products, without
opening the snapshot or emission tables. The default is **PNG with titles**,
saved in `figures_titled/png/` beside `processed/`. Select formats and styles
with command-line flags:

```bash
uv run plot --pdf
uv run plot --png --pdf --titled --untitled
```

`--pdf` selects only PDF; combining `--png --pdf` saves both formats.
`--untitled` selects only the manuscript version in `figures/`; combining
`--titled --untitled` saves both styles. Each selected style uses its own
`png/` and/or `pdf/` directories. The equivalent YAML defaults are
`figure_formats: [png]` and `figure_styles: [titled]`. Explicit flags override
the YAML for that category; an omitted category keeps its YAML setting.

Every spectrum is drawn in separate ±50 and ±200 km/s views. The additional
all-lines figures compare all ten cold+hot total line profiles, each divided
by its own peak across the saved ±200 km/s window. Intrinsic and dust-attenuated
profiles have separate comparison figures. Both views use the same saved
curves and normalization; no processing rerun is needed. Filenames include
`_v50` or `_v200`, for example `spectrum_co10_v200.png` and
`spectrum_all_lines_attenuated_v200.png`. Images are saved once per selected
style. Existing files in unselected formats or styles are left in place.

Both commands accept `--config /path/to/emission.yaml`. To select another
snapshot without editing `dataset`, supply its absolute path and the same
region to both commands:

```bash
uv run process --dataset /path/to/simulations/plt0008703 --x 64 66 --y 64 66
uv run plot --dataset /path/to/simulations/plt0008703 --x 64 66 --y 64 66
```

CLI `--dataset` accepts an absolute path (including `~/...`); YAML keeps its
existing relative-path support. `jobname`, tables, workers and output root still
come from YAML. Plot uses only the dataset folder name to locate saved products;
the original simulation is not needed. Process records its effective settings
in `emission_report.json`.

[Detailed usage](docs/usage.md) · [Methods](docs/emission_method.md) ·
[Code guide](docs/code_walkthrough.md) · [Repository layout](docs/repository_layout.md)
