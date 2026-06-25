# Reproducing the quokka2s pipeline

This guide sets up a fresh environment, installs the package, and reproduces the
figures. It assumes you have **the simulation snapshot** (`plt0655228/`) and **the
prebuilt DESPOTIC table** (`output_tables_3D_GOW_LVG/despotic_table.npz`) from the
author — you do **not** need to rebuild the table.

The one non-obvious dependency is **yt**: the QUOKKA frontend that reads the `plt*`
datasets is only complete on yt's `main` branch, *not* in the pip-stable release —
so yt must be installed from git (step 2).

---

## 0. What you need up front

- `conda`/`mamba` (or any Python 3.11 environment manager) and a C compiler
  (yt builds Cython extensions from source).
- **~8 GB free disk** for a full-resolution (`down=1`) run's field caches, plus
  **~2.3 GB** for the CHIANTI atomic database (step 4).
- From the author: the dataset directory `plt0655228/` and the table file
  `output_tables_3D_GOW_LVG/despotic_table.npz`.

## 1. Create the environment

```bash
conda create -n quokka-repro python=3.11
conda activate quokka-repro
```

## 2. Install yt from git (REQUIRED — not pip-stable)

The pip release (`pip install yt`, currently 4.4.2) ships only a *partial*
`QuokkaDataset` and lacks `QuokkaHierarchy` (particle handling). The full frontend
is on `main`:

```bash
python -m pip install "git+https://github.com/yt-project/yt"
```

> For a *byte-identical* reproduction, pin the commit:
> `pip install "git+https://github.com/yt-project/yt@<sha>"`. The author's working
> copy is the `chongchonghe/yt` fork (branch `Rongjun-ANUquokka-frontend`,
> commit `581ad28`), whose only delta over upstream `main` is a packaging tweak —
> not physics — so upstream `main` reads the data identically.

## 3. Clone and install this package

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
python -m pip install -e .            # installs numpy/scipy/astropy/h5py/fiasco/… 
```

`despotic` is **not** installed — it's only needed to *rebuild* tables. If you ever
want to rebuild: `pip install -e ".[tables]"`.

## 4. Download the CHIANTI database (for fiasco)

The `[C II]` 158 µm emissivity tables are built at import from CHIANTI via `fiasco`.
On first use, `fiasco` downloads and builds the database (~2.3 GB) into `~/.fiasco/`:

```bash
# triggers the one-time CHIANTI download/build; accept the prompt
python -c "import fiasco; fiasco.Ion('C 2', [1e4])"
```

This work used **CHIANTI 10.1**. If `fiasco` offers a different version, pick 10.1
to match exactly.

## 5. Point the config at your data + table

Edit `src/quokka2s/pipeline/prep/config.py` (or set the env vars shown):

| What | Where | Set to |
|---|---|---|
| Dataset | `YT_DATASET` env *or* `YT_DATASET_PATH` default | your `…/plt0655228` |
| Table (LVG) | `DESPOTIC_TABLE_PATH_LVG` | your `…/output_tables_3D_GOW_LVG/despotic_table.npz` |
| Output root | `_OUTPUT_ROOT` | a writable dir for `output/…` |

The output dir name is derived automatically as
`output/<dataset>_down<N>_Lext<L>kpc<_tag>/`.

## 6. Run

Use the per-task driver (each task in its own process — avoids OOM at `down=1`):

```bash
# canonical config: full resolution, L_ext = 15 kpc, GOW LVG table
MODE=compute LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh   # heavy physics → caches
MODE=plot    LEXT_KPC=15 RUN_TAG=v4 scripts/run_dataset_series.sh   # render figures
```

Or a single task directly:

```bash
LEXT_KPC=15 RUN_TAG=v4 python -m quokka2s.pipeline.tasks.run_pipeline \
    --mode plot --task Plot_VelocityPhase
```

The first `--mode compute` run is the expensive one (derives + caches 7 fields:
`column_density_H`, `dVdr_lvg`, `temperature_despotic`, and the four
`*_luminosity` fields). Subsequent `--mode plot` runs are fast.

## 7. Verify

Outputs land in `output/plt0655228_down1_Lext15kpc_v4/` as PNGs. Compare against
the author's reference figures (e.g. `PhaseSpectrumOverlay_*_los{x,y,z}_*.png`,
`PhaseSigmaV_*.png`, the multi-field slices). The numeric results live in
`output/<dir>/task_intermediates/*.h5` if you want a quantitative diff.

---

## Known-good versions (this work)

Python 3.11.14 · yt 4.5.dev0 (main / fork) · numpy 2.2.6 · scipy 1.16.3 ·
astropy 7.1.1 · h5py 3.16.0 · matplotlib 3.10.6 · unyt 3.0.4 · fiasco 0.6.2 +
CHIANTI 10.1 · joblib · tqdm · tqdm-joblib · pyyaml. (despotic 2.2 — tables only.)

## Operational gotchas

- **Never wrap the run in `conda run`** — it silently kills long jobs on macOS.
  Call the env's `python` binary (or activate the env) directly.
- **Run long jobs in the background**; a full `down=1` compute is tens of minutes.
- **Changing `LEXT_KPC` or downsample invalidates the field caches** (they're keyed
  by those). Switching forces a recompute.
- **`down=1` needs ~8 GB free disk** for the field caches; macOS "purgeable" space
  can cause a mid-run `No space left on device`.
