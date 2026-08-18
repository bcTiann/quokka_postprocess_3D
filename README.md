# quokka2s — QUOKKA post-processing pipeline

Post-processing for [QUOKKA](https://github.com/quokka-astro/quokka) radiation-MHD
snapshots: turns a simulation `plt*` output into **synthetic line emission and
multi-phase ISM diagnostics**, so simulations can be compared against real
observations. Line emissivities use a pre-built [DESPOTIC](https://despotic.readthedocs.io)
chemistry/cooling table plus an HM2012 shielded Cloudy [C II] table.

### What it produces

- **Synthetic line emission** — CO J=1–0 and J=2–1, [C II] 158 µm, Hα,
  and H I 21 cm:
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
        └── analysis.py  data_handling.py  plotting.py
```

---

# Reproducing from scratch

The whole setup — fresh conda environment to final figures. This procedure was
tested end-to-end in a clean clone on 2026-06-30 with Python 3.11.15.

## 0. What you need

- **`conda`/`mamba`** and a **C compiler** (yt builds Cython extensions from source —
  on macOS install the Xcode Command Line Tools, on Linux `gcc`).
- **At least 18 GB free disk** for the 8 GB snapshot, roughly 6 GB of
  full-resolution (`down=1`) field caches, figures, and task results.
- From the author: the dataset directory (e.g. `plt0655228/`) and the prebuilt
  table `output_tables_3D_GOW_LVG/despotic_table_co10_co21_clean.npz`.
  **You do not rebuild the table** — place both inputs as shown in step 4.

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

pip install -r requirements.txt      # yt-from-git + numpy/scipy/astropy/h5py/…
pip install -e .                      # the quokka2s package itself
```

> `despotic` is **not** installed — it's only needed to *rebuild* tables. If you
> ever want to, run `python -m pip install -e ".[tables]"`. The extra installs a
> pinned commit of the official DESPOTIC 2.2 source because 2.2 is not published
> on PyPI (and PyPI 2.1 predates the GOW chemistry network used here).

Verify the install:

```bash
python -c "import quokka2s, yt; print('yt', yt.__version__); print('quokka2s', quokka2s.__file__)"
```

### Rebuild the current six-line Cloudy tables

The portable Cloudy workflow is independent of the QUOKKA snapshot pipeline.
It requires a separately compiled Cloudy 17.02 executable. From a fresh clone:

```bash
export CLOUDY_EXE=/path/to/cloudy/c17.02/source/cloudy.exe

# Fast one-point validation first
python scripts/build_cloudy_sixline_tables.py \
  --cloudy-exe "$CLOUDY_EXE" --smoke-only

# Full column-density and Jeans-length tables
python scripts/build_cloudy_sixline_tables.py \
  --cloudy-exe "$CLOUDY_EXE" --workers 11 --force
```

The command uses the tracked `vendor/cloudy_cooling_tools/CIAOLoop_lines` and
portable `.par.in` templates. Generated runtime files go to
`runtime/cloudy_sixline/`; final NPZ tables go to `data/`. See
[`CLOUDY_HM12_FILTERED_ISM_WORKFLOW.md`](CLOUDY_HM12_FILTERED_ISM_WORKFLOW.md)
for the radiation field, physical settings, axes, failure policy, and exact
artifact chain.

## 3. Build the runtime Cloudy [C II] table

The Cloudy 17.02 HM2012 runs are converted once into the compact runtime table.
The 42 failed cells are filled linearly in
`log10(T)–log10(epsilon_CII/n_H²)` while their original mask is retained:

```bash
PYTHONPATH=src python scripts/build_cloudy_cii_table.py
```

This writes `data/cloudy_cii_hm2012_z0_coarse.npz`. Override its runtime path
with `CLOUDY_CII_TABLE=/absolute/path/to/table.npz` when needed.

## 4. Put the snapshot and lookup tables in place

Do this **before** running `MODE=compute`. The runner derives the repository root
from its own location, so the standard local setup does not require editing
`config.py` or setting absolute paths. Put the two non-Git inputs here:

```text
quokka_postprocess_3D/
├── plt0655228/                         # complete QUOKKA plotfile directory
│   ├── metadata.yaml
│   └── ...
├── output_tables_3D_GOW_LVG/
│   └── despotic_table_co10_co21_clean.npz  # precomputed 3D GOW/LVG lookup table
├── data/
│   └── cloudy_cii_hm2012_z0_coarse.npz      # HM2012 Cloudy [C II] lookup table
├── scripts/
└── src/
```

Both input paths are ignored by Git and are not downloaded by `git clone`. Copy
them from local storage, for example:

```bash
cd ~/quokka_postprocess_3D
cp -a /path/to/plt0655228 ./
mkdir -p output_tables_3D_GOW_LVG
cp /path/to/despotic_table_co10_co21_clean.npz output_tables_3D_GOW_LVG/
```

For a large snapshot, a symlink is also valid:

```bash
ln -s /absolute/path/to/plt0655228 ./plt0655228
```

To rebuild the canonical table, install the table extra, run the fixed GOW/LVG
builder, then apply the conservative convex-hull cleaner. The production grid
and physics are fixed; only output path, worker count, and overwrite permission
are configurable:

```bash
python -m pip install -e ".[tables]"
python -m quokka2s.tables.build_table \
  --output output_tables_3D_GOW_LVG/despotic_table_co10_co21.npz \
  --workers -1 --force
python scripts/fill_table_convex_hull_only.py \
  output_tables_3D_GOW_LVG/despotic_table_co10_co21.npz \
  output_tables_3D_GOW_LVG/despotic_table_co10_co21_clean.npz
```

If the older raw and clean tables are already present, add CO(2–1) without
repeating the expensive chemistry/thermal solve:

```bash
python -m quokka2s.tables.augment_co21 --workers -1
```

This writes `despotic_table_co10_co21.npz` and
`despotic_table_co10_co21_clean.npz`. The legacy table token `CO` remains the
CO(1–0) compatibility name; the added token is `CO21`.

Check both inputs before starting the expensive run:

```bash
test -d plt0655228 && echo "snapshot: OK"
test -f output_tables_3D_GOW_LVG/despotic_table_co10_co21_clean.npz && echo "table: OK"
python -c "import numpy as np; p='output_tables_3D_GOW_LVG/despotic_table_co10_co21_clean.npz'; z=np.load(p, allow_pickle=True); print(p, len(z.files), 'arrays')"
```

To analyze a different plotfile stored under the repository root, pass its
directory name to the runner, for example
`scripts/run_dataset_series.sh plt0857000`. For inputs stored elsewhere, either
symlink them into the layout above or use the direct module with
`YT_DATASET=/absolute/path/to/plt...` and
`DESPOTIC_TABLE=/absolute/path/to/despotic_table_co10_co21_clean.npz` and
`CLOUDY_CII_TABLE=/absolute/path/to/cloudy_cii_hm2012_z0_coarse.npz`.

## 5. Run the pipeline

Activate the environment and run from the clone root. The driver resolves
`python` from the active conda environment and runs each task group in its own
process so memory is released between groups. The canonical setup is full
resolution, `L_ext = 15 kpc`, and the GOW/LVG table:

```bash
conda activate test-env
cd ~/quokka_postprocess_3D

# heavy physics → caches the derived fields + per-task results
MODE=compute LEXT_KPC=15 scripts/run_dataset_series.sh plt0655228

# render all figures from the caches (fast)
MODE=plot    LEXT_KPC=15 scripts/run_dataset_series.sh plt0655228
```

The canonical C+ field uses `T_QUOKKA` as its model selector: cells below
3000 K use the DESPOTIC GOW/LVG emissivity, while cells at or above 3000 K use
the HM2012 shielded Cloudy table at `(T_QUOKKA, n_H, N_H)`. Temperatures above
the intentionally truncated Cloudy grid return zero. The dedicated C+ split
task plots the cold, hot, and channel-by-channel total spectra.

The CO/C+ temperature policy is species-specific throughout their spectral
products and phase panels: both CO lines use each cell's
`temperature_despotic`, while C+ uses `temperature_quokka`. Their thermal-width fields and phase-histogram
temperature axes follow the same mapping; neither uses
`temperature_two_regime`.

Hα retains its two-regime treatment. The main 2×4 phase figure also uses the
two-regime H I result: DESPOTIC below 3000 K and the QUOKKA mean-molecular-weight
method at and above 3000 K. For comparison, H I is additionally computed as two
independent all-cell results. The pure-DESPOTIC result uses the table `n_HI` and
`temperature_despotic` for every cell. The pure-QUOKKA result uses
`temperature_quokka` in the mean-molecular-weight inversion for every cell,
with `n_HI=(1-x_e)n_H` when `x_e<=1` and `n_HI=0` otherwise. Their spectra use
matching thermal widths and are compared without peak normalization in the
dedicated H I task.

The first line printed by the runner includes the resolved Python executable.
It should point into the active environment, not another hard-coded conda env.
Detailed logs are written to `logs/dataset_series/`; each task group should end
with `RC=0`. Do not start a second runner against the same dataset/output while
one is active.

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
integrated spectra, the 2×4 `phase_combined.png`, the dedicated
`HI_phase_comparison.png`, `HI_spectrum_DESPOTIC_Rinf.png`,
`HI_spectrum_QUOKKA_Rinf.png`, and `HI_spectrum_overlay_Rinf.png`, and ten
multi-field slices. The `*.h5`
under `task_intermediates/` hold the underlying numbers for a precise comparison.

```bash
# no output means no recorded traceback/error/non-zero task return code
rg 'Traceback|ERROR|RC=[1-9]' logs/dataset_series
```

---

## Dependencies & citation

Built on the scientific-Python ecosystem and these tools — please cite them if you
publish results based on this pipeline:

- **[QUOKKA](https://github.com/quokka-astro/quokka)** — the R-MHD code producing the
  data (Wibking & Krumholz 2022; He et al. 2024a,b).
- **[yt](https://yt-project.org/)** — data loading/handling (Turk et al. 2011);
  used here from the `main` branch for the QUOKKA frontend.
- **[DESPOTIC](https://despotic.readthedocs.io)** — chemistry/cooling and line
  luminosities (Krumholz 2014).
- **Cloudy 17.02** — HM2012 shielded [C II] 158 µm emissivity grid.

## License

[MIT](LICENSE).
