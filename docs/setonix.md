# Run on Setonix

With `uv` available, clone the repository under `$MYSOFTWARE` and install its dependencies.

```bash
cd "$MYSOFTWARE"
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
uv sync
```

Run processing and plotting from the repository root in a Slurm compute allocation.

## Select a snapshot

Set `dataset` to the absolute path of one `plt` directory and `jobname` to the output group in [configs/emission.yaml](../configs/emission.yaml).

```yaml
dataset: /absolute/path/plt0008703
jobname: my-run
```

```bash
uv run python process_snapshot.py
uv run python plot_emission_results.py
```

Or pass the same arguments to both scripts.

```bash
uv run python process_snapshot.py --snapshot /absolute/path/plt0008703 --jobname my-run
uv run python plot_emission_results.py --snapshot /absolute/path/plt0008703 --jobname my-run
```

With the default config, processing saves data under `output/my-run/plt0008703/processed/`. Plotting reads those products and writes figures to `output/my-run/plt0008703/figures_titled/png/`.
