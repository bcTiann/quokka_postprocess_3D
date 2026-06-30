# quokka2s — QUOKKA post-processing pipeline

Post-processing for [QUOKKA](https://github.com/quokka-astro/quokka) radiation-MHD
snapshots: turns a simulation `plt*` output into **synthetic line emission and
multi-phase ISM diagnostics**, so simulations can be compared against real
observations. Line emissivities use a pre-built [DESPOTIC](https://despotic.readthedocs.io)
chemistry/cooling table plus CHIANTI atomic data (via `fiasco`).

### What it produces

- **Synthetic line emission** — CO J=1–0, [C II] 158 µm, Hα, and H I 21 cm:
  spatially integrated 1D spectra (intrinsic + instrument-convolved) and
  phase/line-profile overlays for all three lines of sight.
- **Multi-phase ISM diagnostics** — mass- and luminosity-weighted histograms over
  the five ISM phases (CNM/UNM/WNM/WIM/HIM), N_H–ρ phase planes, and per-phase
  velocity dispersions σ_x/σ_y/σ_z.
- **Field slices** — temperature, density, column density, dV/dr, … as multi-panel
  figures.

### Repository layout

```
quokka_postprocess_3D/            ← repo root
├── pyproject.toml  requirements.txt
├── README.md
├── scripts/                       run_dataset_series.sh + standalone scripts
└── src/
    └── quokka2s/                  the package (import name: quokka2s)
        ├── pipeline/
        │   ├── prep/      config.py, physics_fields.py, …
        │   ├── services/  spectrum_service.py
        │   └── tasks/     run_pipeline.py  +  Build_*/Plot_* tasks   ← entry point
        ├── tables/        DESPOTIC table builder + lookup
        ├── utils/
        └── analysis.py  data_handling.py  plotting.py  despotic_tables.py
```

---

# Reproducing from scratch

The whole setup — fresh conda environment to final figures. This procedure was
tested end-to-end in a clean clone on 2026-06-30 with Python 3.11.15.

## 0. What you need

- **`conda`/`mamba`** and a **C compiler** (yt builds Cython extensions from source —
  on macOS install the Xcode Command Line Tools, on Linux `gcc`).
- **At least 20 GB free disk** for the 8 GB snapshot, roughly 6 GB of
  full-resolution (`down=1`) field caches, figures/task results, and the
  **~2.3 GB** CHIANTI atomic database (step 3).
- From the author: the dataset directory (e.g. `plt0655228/`) and the prebuilt
  table `output_tables_3D_GOW_LVG/despotic_table.npz`. **You do not rebuild the
  table** — place both inputs as shown in step 4.

## 1. Create the environment

```bash
conda create -n test-env python=3.11
conda activate test-env
```

## 2. Install dependencies + the package

`requirements.txt` pins the known-good versions **and** installs yt from git. The
QUOKKA frontend needed by these `plt*` files is taken from yt's `main` branch;
the stable yt release tested during this project had only a partial reader:

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D

pip install -r requirements.txt      # yt-from-git + numpy/scipy/astropy/h5py/fiasco/…
pip install -e .                      # the quokka2s package itself
```

> `despotic` is **not** installed — it's only needed to *rebuild* tables. If you
> ever want to: `pip install -e ".[tables]"` (or uncomment it in `requirements.txt`).

Verify the install:

```bash
python -c "import quokka2s, yt; print('yt', yt.__version__); print('quokka2s', quokka2s.__file__)"
```

## 3. CHIANTI atomic database (automatic — nothing extra to install)

The [C II] 158 µm emissivity uses CHIANTI atomic data through `fiasco`. The `fiasco`
*package* was already installed in step 2, but the CHIANTI **database** itself is a
~2.3 GB science dataset, not a pip package — so it can't live in `requirements.txt`.
`fiasco` fetches and builds it into `~/.fiasco/` the **first time it's used**, which
happens automatically on your first pipeline run. There is no separate install step.

It only downloads once and is shared across all conda environments (it lives in your
home directory). If you'd rather pull it now than have it happen mid-run, trigger it:

```bash
python -c "import astropy.units as u; import fiasco; print(fiasco.Ion('C 2', 1e4*u.K))"   # accept the download prompt
```

The temperature must carry an Astropy unit; `fiasco` 0.6 rejects unitless lists
such as `[1e4]` before it checks or downloads the database.

This work used **CHIANTI 10.1**; if `fiasco` offers a choice, pick 10.1 to match.

## 4. Put the snapshot and DESPOTIC table in place

Do this **before** running `MODE=compute`. The runner derives the repository root
from its own location, so the standard local setup does not require editing
`config.py` or setting absolute paths. Put the two non-Git inputs here:

```text
quokka_postprocess_3D/
├── plt0655228/                         # complete QUOKKA plotfile directory
│   ├── metadata.yaml
│   └── ...
├── output_tables_3D_GOW_LVG/
│   └── despotic_table.npz              # precomputed 3D GOW/LVG lookup table
├── scripts/
└── src/
```

Both input paths are ignored by Git and are not downloaded by `git clone`. Copy
them from local storage, for example:

```bash
cd ~/quokka_postprocess_3D
cp -a /path/to/plt0655228 ./
mkdir -p output_tables_3D_GOW_LVG
cp /path/to/despotic_table.npz output_tables_3D_GOW_LVG/
```

For a large snapshot, a symlink is also valid:

```bash
ln -s /absolute/path/to/plt0655228 ./plt0655228
```

Check both inputs before starting the expensive run:

```bash
test -d plt0655228 && echo "snapshot: OK"
test -f output_tables_3D_GOW_LVG/despotic_table.npz && echo "table: OK"
python -c "import numpy as np; p='output_tables_3D_GOW_LVG/despotic_table.npz'; z=np.load(p, allow_pickle=True); print(p, len(z.files), 'arrays')"
```

To analyze a different plotfile stored under the repository root, pass its
directory name to the runner, for example
`scripts/run_dataset_series.sh plt0857000`. For inputs stored elsewhere, either
symlink them into the layout above or use the direct module with
`YT_DATASET=/absolute/path/to/plt...` and
`DESPOTIC_TABLE_LVG=/absolute/path/to/despotic_table.npz`.

## 5. Run the pipeline

Activate the environment and run from the clone root. The driver resolves
`python` from the active conda environment and runs each task group in its own
process so memory is released between groups. The canonical setup is full
resolution, `L_ext = 15 kpc`, and the GOW/LVG table:

```bash
conda activate test-env
cd ~/quokka_postprocess_3D

# heavy physics → caches the 7 derived fields + per-task results
MODE=compute LEXT_KPC=15 scripts/run_dataset_series.sh

# render all figures from the caches (fast)
MODE=plot    LEXT_KPC=15 scripts/run_dataset_series.sh
```

The first line printed by the runner includes the resolved Python executable.
It should point into the active environment, not another hard-coded conda env.
Detailed logs are written to `logs/dataset_series/`; each task group should end
with `RC=0`. Do not start a second runner against the same dataset/output while
one is active.

Or call the module directly:

```bash
# only re-plot from cached results
LEXT_KPC=15 python -m quokka2s.pipeline.tasks.run_pipeline --mode plot

# a single task
LEXT_KPC=15 python -m quokka2s.pipeline.tasks.run_pipeline \
    --mode plot --task Plot_VelocityPhase

# wipe both intermediate stores and exit
python -m quokka2s.pipeline.tasks.run_pipeline --clean-intermediates
```

- **`--mode compute`** runs the `Build_*` tasks (derive + cache fields/results);
  **`--mode plot`** runs the `Plot_*` tasks (render from cache); **`--mode all`**
  (default) does both.
- The first `--mode compute` run is the expensive one — it derives and caches seven
  fields (`column_density_H`, `dVdr_lvg`, `temperature_despotic`, and the four
  `*_luminosity` fields). After that, `--mode plot` re-renders in about a minute
  on the tested workstation.
- Prefer `run_dataset_series.sh` for a full run. Calling the module directly is
  useful for a selected task, but a whole direct run keeps more state in one
  Python process and is less suitable for a memory-limited workstation.

## 6. Verify

| Path | Contents |
|---|---|
| `output/<dataset>_down<N>_Lext<L>kpc<_tag>/` | the figures (PNG) |
| `output/<…>/task_intermediates/` | per-task results (HDF5) — for quantitative diffs |
| `<dataset_parent>/intermediates/<dataset>/fields/` | cached derived fields (HDF5) |
| `logs/dataset_series/` | master and per-task-group logs |

With the standard layout, the field cache is therefore
`intermediates/plt0655228/fields/`. The 2026-06-30 clean test produced 30 PNGs,
including `PhaseSpectrumOverlay_*_los{x,y,z}_*.png`, `PhaseSigmaV_*.png`, the
integrated spectra, `phase_combined.png`, and ten multi-field slices. The `*.h5`
under `task_intermediates/` hold the underlying numbers for a precise comparison.

```bash
# no output means no recorded traceback/error/non-zero task return code
rg 'Traceback|ERROR|RC=[1-9]' logs/dataset_series
```

---

## Running on an HPC cluster

All machine-specific paths are env-overridable (defaults are repo-relative), so the
same code runs unchanged — you just point a few env vars at the cluster's
filesystems. Layout/install don't change: src-layout runs identically, and either
`pip install -e .` (into a writable conda/venv) **or** `export PYTHONPATH=$PROJECT/src`
works (the package is pure-Python). Note `requirements.txt` deps must be installed
either way.

Env knobs:

| Var | Points at | Default |
|---|---|---|
| `QUOKKA_ROOT` | repo root (base for the defaults below) | derived from `config.py` location |
| `YT_DATASET` | the snapshot dir | `$QUOKKA_ROOT/plt0655228` |
| `DESPOTIC_TABLE_LVG` | the prebuilt table `.npz` | `$QUOKKA_ROOT/output_tables_3D_GOW_LVG/despotic_table.npz` |
| `QUOKKA_OUTPUT_ROOT` | where figures + task results go | `$QUOKKA_ROOT/output` |
| `QUOKKA_CACHE_ROOT` | where derived-field caches go (set this if the dataset is on a **read-only** mount) | next to the dataset |
| `DESPOTIC_HOME` | DESPOTIC LAMDA data (avoids network fetch on offline nodes) | auto-probed |
| `MPLBACKEND` | matplotlib backend | `Agg` (headless, auto-set) |
| `QK_SPECTRUM_WORKERS` | spectrum-builder threads (RAM-bound; full process peaks around 13–14 GB on the tested snapshot) | `2` |

SLURM template:

```bash
#!/bin/bash
#SBATCH --job-name=quokka2s --time=08:00:00 --mem=64G --cpus-per-task=8
set -euo pipefail
PROJECT=$HOME/quokka_postprocess_3D          # on a node-visible filesystem (NFS/Lustre)

module load anaconda && conda activate quokka2s     # editable install lives here
# — or, if the cluster python is read-only:  export PYTHONPATH=$PROJECT/src:$PROJECT/deps

export QUOKKA_ROOT=$PROJECT
export YT_DATASET=$SCRATCH/plt0655228
export DESPOTIC_TABLE_LVG=$SCRATCH/tables/despotic_table.npz
export QUOKKA_OUTPUT_ROOT=$SCRATCH/quokka_out
export QUOKKA_CACHE_ROOT=$SCRATCH/intermediates    # writable mount for field caches
export MPLBACKEND=Agg

cd "$PROJECT"
LEXT_KPC=15 python -m quokka2s.pipeline.tasks.run_pipeline --mode compute
LEXT_KPC=15 python -m quokka2s.pipeline.tasks.run_pipeline --mode plot
```

One-time on the **login node** (it has network; compute nodes often don't):
`pip install -r requirements.txt && pip install -e .` then **pre-warm CHIANTI**
`python -c "import astropy.units as u; import fiasco; print(fiasco.Ion('C 2', 1e4*u.K))"`
(the 2.3 GB DB download would
otherwise hang a batch job). Keep `~/.fiasco` on a node-visible filesystem. Make
sure `--mem` exceeds the **~14 GB** spectra peak.

---

## Known-good versions

Python 3.11.15 · yt 4.5.dev0 (git `main`) · numpy 2.2.6 (`<2.3`: fiasco) ·
scipy 1.16.3 · astropy 7.1.1 · h5py 3.16.0 · matplotlib 3.10.6 · unyt 3.0.4 ·
fiasco 0.6.2 + CHIANTI 10.1 · joblib 1.5.2 · tqdm 4.67.1 · tqdm-joblib 0.0.5 ·
PyYAML 6.0.3.  (despotic 2.2 — tables only.)  Pins are in `requirements.txt`.

## Operational gotchas

- **Never wrap a run in `conda run`** — it silently kills long jobs on macOS.
  Activate the env and call `python` directly.
- **Use `tmux`, `screen`, or a batch job for long runs** — the clean `down=1`
  compute took about 80 minutes on the tested workstation; hardware and caches
  change this substantially.
- **Changing `LEXT_KPC` or the downsample factor invalidates the field caches**
  (they're keyed by those); switching forces a recompute.
- **`down=1` needs several GB for field caches** (5.9 GB in the clean test), plus
  task results and figures; on macOS "purgeable" space can cause a mid-run
  `No space left on device`.

## Dependencies & citation

Built on the scientific-Python ecosystem and these tools — please cite them if you
publish results based on this pipeline:

- **[QUOKKA](https://github.com/quokka-astro/quokka)** — the R-MHD code producing the
  data (Wibking & Krumholz 2022; He et al. 2024a,b).
- **[yt](https://yt-project.org/)** — data loading/handling (Turk et al. 2011);
  used here from the `main` branch for the QUOKKA frontend.
- **[DESPOTIC](https://despotic.readthedocs.io)** — chemistry/cooling and line
  luminosities (Krumholz 2014).
- **[CHIANTI](https://www.chiantidatabase.org/)** via **[fiasco](https://fiasco.readthedocs.io)**
  — atomic data for the [C II] 158 µm line (Dere et al. 1997; Del Zanna et al. 2021).

## License

[MIT](LICENSE).
