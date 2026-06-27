# quokka2s — QUOKKA post-processing pipeline

Post-processing for [QUOKKA](https://github.com/quokka-astro/quokka) radiation-MHD
snapshots: turns a simulation `plt*` output into **synthetic line emission and
multi-phase ISM diagnostics**, so simulations can be compared against real
observations. Line emissivities use a pre-built [DESPOTIC](https://despotic.readthedocs.io)
chemistry/cooling table plus CHIANTI atomic data (via `fiasco`).

### What it produces

- **Synthetic line emission** — CO J=1–0, [C II] 158 µm, Hα, and H I 21 cm:
  1D spectra (intrinsic + instrument-convolved) and surface-brightness maps.
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
├── docs/                          physics notes (line emission, validation, audits)
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

The whole setup — fresh conda environment to final figures. Tested end-to-end:
yt installed this way reads the data identically and produces bit-identical results.

## 0. What you need

- **`conda`/`mamba`** and a **C compiler** (yt builds Cython extensions from source —
  on macOS install the Xcode Command Line Tools, on Linux `gcc`).
- **~8 GB free disk** for a full-resolution (`down=1`) run's field caches, plus
  **~2.3 GB** for the CHIANTI atomic database (step 3).
- From the author: the dataset directory (e.g. `plt0655228/`) and the prebuilt
  table `output_tables_3D_GOW_LVG/despotic_table.npz`. **You do not rebuild the
  table** — just point the config at it (step 4).

## 1. Create the environment

```bash
conda create -n quokka python=3.11
conda activate quokka
```

## 2. Install dependencies + the package

`requirements.txt` pins the known-good versions **and** installs yt from git (the
QUOKKA frontend that reads `plt*` data is only complete on yt's `main` branch — the
pip-stable release, currently 4.4.2, ships only a partial reader and will fail):

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

## 3. Download the CHIANTI database (for fiasco)

The `[C II]` 158 µm emissivity tables are built at import from CHIANTI via `fiasco`.
On first use, `fiasco` downloads and builds the database (~2.3 GB) into `~/.fiasco/`
(shared across conda environments since it lives in your home directory):

```bash
# triggers the one-time CHIANTI download/build — accept the prompt
python -c "import fiasco; fiasco.Ion('C 2', [1e4])"
```

This work used **CHIANTI 10.1**; if `fiasco` offers a different version, pick 10.1
to match exactly.

## 4. Point the config at your data + table

Edit `src/quokka2s/pipeline/prep/config.py` (or set the env vars shown):

| What | Where | Set to |
|---|---|---|
| Dataset | `YT_DATASET` env *or* the `YT_DATASET_PATH` default | your `…/plt0655228` |
| Table (LVG) | `DESPOTIC_TABLE_PATH_LVG` | your `…/output_tables_3D_GOW_LVG/despotic_table.npz` |
| Output root | `_OUTPUT_ROOT` | a writable directory for `output/…` |

The output directory name is derived automatically as
`output/<dataset>_down<N>_Lext<L>kpc<_tag>/`.

## 5. Run the pipeline

The driver runs each task in its own process (releases memory between tasks —
needed to avoid OOM at full resolution). Canonical config = full resolution,
L_ext = 15 kpc, GOW LVG table:

```bash
# heavy physics → caches the 7 derived fields + per-task results
MODE=compute LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh

# render all figures from the caches (fast)
MODE=plot    LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh
```

Or call the module directly:

```bash
# whole pipeline (compute + plot)
LEXT_KPC=15 RUN_TAG=v4 python -m quokka2s.pipeline.tasks.run_pipeline

# only re-plot from cached results
LEXT_KPC=15 RUN_TAG=v4 python -m quokka2s.pipeline.tasks.run_pipeline --mode plot

# a single task
LEXT_KPC=15 RUN_TAG=v4 python -m quokka2s.pipeline.tasks.run_pipeline \
    --mode plot --task Plot_VelocityPhase

# wipe both intermediate stores and exit
python -m quokka2s.pipeline.tasks.run_pipeline --clean-intermediates
```

- **`--mode compute`** runs the `Build_*` tasks (derive + cache fields/results);
  **`--mode plot`** runs the `Plot_*` tasks (render from cache); **`--mode all`**
  (default) does both.
- The first `--mode compute` run is the expensive one — it derives and caches seven
  fields (`column_density_H`, `dVdr_lvg`, `temperature_despotic`, and the four
  `*_luminosity` fields). After that, `--mode plot` re-renders in seconds.

## 6. Verify

| Path | Contents |
|---|---|
| `output/<dataset>_down<N>_Lext<L>kpc<_tag>/` | the figures (PNG) |
| `output/<…>/task_intermediates/` | per-task results (HDF5) — for quantitative diffs |
| `<dataset_dir>/intermediates/<dataset>/fields/` | cached derived fields (HDF5) |

Compare the PNGs (e.g. `PhaseSpectrumOverlay_*_los{x,y,z}_*.png`, `PhaseSigmaV_*.png`,
the multi-field slices) against the author's reference figures; the `*.h5` under
`task_intermediates/` hold the underlying numbers for a precise comparison.

---

## Known-good versions

Python 3.11.14 · yt 4.5.dev0 (git `main`) · numpy 2.2.6 (`<2.3`: fiasco) ·
scipy 1.16.3 · astropy 7.1.1 · h5py 3.16.0 · matplotlib 3.10.6 · unyt 3.0.4 ·
fiasco 0.6.2 + CHIANTI 10.1 · joblib 1.5.2 · tqdm 4.67.1 · tqdm-joblib 0.0.5 ·
PyYAML 6.0.3.  (despotic 2.2 — tables only.)  Pins are in `requirements.txt`.

## Operational gotchas

- **Never wrap a run in `conda run`** — it silently kills long jobs on macOS.
  Activate the env and call `python` directly.
- **Run long jobs in the background** — a full `down=1` compute is tens of minutes.
- **Changing `LEXT_KPC` or the downsample factor invalidates the field caches**
  (they're keyed by those); switching forces a recompute.
- **`down=1` needs ~8 GB free disk** for the field caches; on macOS "purgeable"
  space can cause a mid-run `No space left on device`.

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
