# quokka2s — QUOKKA post-processing pipeline

Post-processing for [QUOKKA](https://github.com/quokka-astro/quokka) radiation-MHD
snapshots: turns a simulation `plt*` output into **synthetic line emission and
multi-phase ISM diagnostics**, so simulations can be compared against real
observations. Line emissivities use a pre-built [DESPOTIC](https://despotic.readthedocs.io)
chemistry/cooling table plus CHIANTI atomic data (via `fiasco`).

## What it produces

- **Synthetic line emission** — CO J=1–0, [C II] 158 µm, Hα, and H I 21 cm:
  1D spectra (intrinsic + instrument-convolved) and surface-brightness maps.
- **Multi-phase ISM diagnostics** — mass- and luminosity-weighted histograms over
  the five ISM phases (CNM/UNM/WNM/WIM/HIM), N_H–ρ phase planes, and
  per-phase velocity dispersions σ_x/σ_y/σ_z.
- **Field slices** — temperature, density, column density, dV/dr, … as multi-panel
  slice figures.

## Reproduce it

Full step-by-step (data, table, CHIANTI download, config, verification) is in
**[REPRODUCE.md](REPRODUCE.md)**. The short version:

```bash
# 1. Fresh environment
conda create -n quokka python=3.11 && conda activate quokka

# 2. yt MUST come from git, NOT pip.  The QUOKKA frontend that reads plt* data is
#    only complete on yt's main branch; the pip-stable release (4.4.x) can't read it.
python -m pip install "git+https://github.com/yt-project/yt"

# 3. This package (pulls numpy/scipy/astropy/h5py/fiasco/…; NOT despotic)
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D && python -m pip install -e .

# 4. One-time: fiasco downloads the CHIANTI atomic database (~2.3 GB) on first use
python -c "import fiasco; fiasco.Ion('C 2', [1e4])"
```

Then point `src/quokka2s/pipeline/prep/config.py` at your QUOKKA snapshot and the
prebuilt `despotic_table.npz` (both supplied by the author — you do **not** rebuild
the table), and run the pipeline. See [REPRODUCE.md](REPRODUCE.md) for the exact
paths to edit and the operational gotchas (disk needs, cache invalidation, never
wrap runs in `conda run`).

> **despotic is optional** — only needed to *rebuild* tables: `pip install -e ".[tables]"`.

## Running the pipeline

```bash
# Each task in its own process (avoids OOM at full resolution), canonical config:
MODE=compute LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh   # heavy physics → caches
MODE=plot    LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh   # render figures (fast)

# Or the module directly, one task at a time:
python -m quokka2s.pipeline.tasks.run_pipeline --mode plot --task Plot_VelocityPhase
```

The first `--mode compute` run is the expensive one — it derives and caches seven
fields (`column_density_H`, `dVdr_lvg`, `temperature_despotic`, and the four
`*_luminosity` fields). After that, `--mode plot` re-renders all figures in seconds.

Outputs:

| Path | Contents |
|---|---|
| `output/<dataset>_down<N>_Lext<L>kpc<_tag>/` | the figures (PNG) |
| `output/<…>/task_intermediates/` | per-task results (HDF5) |
| `<dataset_dir>/intermediates/<dataset>/fields/` | cached derived fields (HDF5) |

## Repository layout

```
src/quokka2s/        the package (import name: quokka2s)
  pipeline/          tasks, derived-field physics, caching, services
  tables/            DESPOTIC table builder + lookup
scripts/             driver + standalone analysis/validation scripts
docs/                physics notes (line emission, validation, audits)
REPRODUCE.md         from-scratch setup guide
```

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

Python 3.11; numpy <2.3 (fiasco constraint). Known-good versions are listed in
[REPRODUCE.md](REPRODUCE.md).

## License

[MIT](LICENSE).
