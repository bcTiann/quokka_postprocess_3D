# quokka2s — QUOKKA emission post-processing

Process one QUOKKA snapshot, then plot its saved emission maps and spectra.

## Install

The project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to install dependencies.

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
uv sync
```

## Process and plot

Pass the absolute path to one snapshot directory containing `Header`.
`--jobname` names the output group. Use the same selection for both commands.

```bash
uv run python process_snapshot.py --snapshot /absolute/path/plt0008703 --jobname my-run
uv run python plot_emission_results.py --snapshot /absolute/path/plt0008703 --jobname my-run
```

Alternatively, set `dataset` and `jobname` in
[`configs/emission.yaml`](configs/emission.yaml) and run both scripts without those flags.

```yaml
dataset: /absolute/path/plt0008703
jobname: my-run
```

```bash
uv run python process_snapshot.py
uv run python plot_emission_results.py
```

By default, numerical products go to `output/<jobname>/<snapshot>/processed/`.
Plotting reads those products and writes PNGs in the adjacent `figures_titled/png/`.

[Setonix](docs/setonix.md) · [Detailed usage](docs/usage.md) ·
[Methods](docs/emission_method.md) · [Code walkthrough](docs/code_walkthrough.md) ·
[Repository layout](docs/repository_layout.md) · [Published results](output/)
