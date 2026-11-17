# quokka2s — QUOKKA emission post-processing

Process one explicitly selected QUOKKA snapshot by reading original native
columns into bounded RAM. A separate script draws images and spectra from the
saved numerical arrays. On Setonix, use the short
[Setonix guide](docs/setonix.md).

This branch enables the compiled numerical kernels. The supplied configuration
uses 16 batches of at most 131072 cells, sigma coverage through 8192 km/s, and
a 30 GB whole-process memory guard. Runtime physics and grids remain Python inputs.

Setonix results are tracked in [output/](output/). A normal clone includes the
saved numerical products and PNG/PDF figures in their original directory structure.

## 1. Clone

```bash
git clone --branch codex/cython-optimized https://github.com/bcTiann/quokka_postprocess_3D.git
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
Python 3.11. Project calculations use NumPy/SciPy with small Cython numerical
loops. `uv sync` builds these automatically and installs their pinned build
dependencies; a C compiler is required. The pinned yt source dependency can
also require Git and C/C++ build tools when no cached build is available.
Both process and plot
dependencies are installed by default. `uv run` checks the environment
automatically and reuses the locked versions; it does not reinstall everything
on each run.

The editable installation reads changes to the processing Python files directly.
Run `uv sync` after changing dependencies or package settings.
Changing grids, physical parameters or configuration in Python requires no
native rebuild. Changes to `.pyx` files trigger rebuilding on the next sync/run.

On Setonix, follow its [environment setup](docs/setonix.md#2-configure-the-environment)
instead: keep the repository, environment, caches and generated results under
`$MYSOFTWARE` to avoid the home quota.

## 3. Process one snapshot

Run the script from the repository root and supply one snapshot's absolute path:

```bash
uv run python process_snapshot.py --dataset /absolute/path/to/run1/plt0008703
```

The path must name the snapshot itself, containing `Header`. A parent run
folder is rejected; this script never discovers or processes other snapshots.
`--jobname my-run` optionally changes the output group from the YAML default.
[configs/emission.yaml](configs/emission.yaml) holds output locations, table
paths and execution settings. Its `dataset: null` requires the command-line
path; alternatively set it to one absolute snapshot path. Other relative YAML
paths start at that YAML file. Shell variables are not expanded in YAML.

For a small check, select native x/y indices; the stop is excluded:

```bash
uv run python process_snapshot.py \
  --dataset /absolute/path/to/run1/plt0008703 \
  --jobname my-run --x 64 66 --y 64 66
```

This processes only x=64,65 and y=64,65, retaining each column's full original z
depth. Metadata supplies grid dimensions; no 256/512/2048-sized grid is assumed.
The reader supports rectangular level-zero grids, including shorter edge FABs
and dimensions such as 1024 x 1024 x 2028. It reads intersecting original FAB
fields and required global gradient neighbours into bounded RAM; it does not
write a z-contiguous disk copy. A small region can still require complete
intersecting FAB-field reads, but never a whole-snapshot conversion.

`process_snapshot.py` only calculates, checks and saves numerical products.
It never invokes plotting. Original snapshots and lookup tables stay read-only.
Results go to `results_dir/jobname/snapshot_name/processed/`; explicit x/y
ranges add a matching `region_x64-66_y64-66/` directory before `processed/`.
Complete matching results can be reused after integrity and identity checks.
Conflicting or incomplete outputs are preserved; choose a new results root to
calculate a separate version.

## 4. Plot the saved products

Run the independent plotting script with the same selection:

```bash
uv run python plot_emission_results.py \
  --dataset /absolute/path/to/run1/plt0008703 \
  --jobname my-run --x 64 66 --y 64 66
```

Plot reads only saved arrays. The original snapshot need not still exist; its
folder name selects the result. Omit x/y arguments to select full-box products.
The default saves titled PNG figures in `figures_titled/png/`.
Select formats and title styles independently:

```bash
uv run python plot_emission_results.py --dataset /absolute/path/to/plt0008703 --pdf
uv run python plot_emission_results.py --dataset /absolute/path/to/plt0008703 --png --pdf --titled --untitled
```

Both scripts accept `--config /path/to/emission.yaml`. Repeat command-line
selection overrides in the plot command; they do not edit YAML or persist.
There are no process/plot console aliases registered in `pyproject.toml`.

## Current numerical route

Each selected RAM tile supplies density, temperature, velocity, shielding and
foreground columns, and velocity gradient. Physical lookups calculate each
line's emissivity and foreground dust attenuation. Each original Gaussian
splits luminosity across four existing fractional-mean/sigma template corners.
After all selected cells, NumPy convolution turns those accumulated luminosities
into 400 spectral bins on [-200,200] km/s. Sigma nodes remain logarithmically
spaced; interpolation weights use linear sigma distances.

This is the sole production Gaussian method. Valid sigma values outside table
coverage use stable direct integration with the same 5-sigma cutoff and are
reported in the log. Compiled loops handle physical-table interpolation,
Gaussian coordinate preparation, independent window probabilities and full
moments; their grids, tables and numerical parameters are Python inputs. Only Gaussian
template preparation is outside process timing; metadata, field reads, physical
calculations, all accumulation, checks and saving remain timed. The reader uses
a conservative memory preflight and a 30 GB process RSS guard; tune query and
worker sizes for the available memory. Benchmark complete processing on the
intended snapshot and host when choosing worker settings.

[Detailed usage](docs/usage.md) · [Methods](docs/emission_method.md) ·
[Code guide](docs/code_walkthrough.md) · [Repository layout](docs/repository_layout.md)
