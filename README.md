# quokka2s — QUOKKA emission post-processing

## 1. Clone

```bash
git clone https://github.com/bcTiann/quokka_postprocess_3D.git
cd quokka_postprocess_3D
```

## 2. Configure the environment

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 3. Set the inputs

Paths are relative to the repository root:

| Input | Default path | Provided by |
|---|---|---|
| Simulation snapshot | `inputs/snapshots/plt0655228/` | The user |
| DESPOTIC table | `inputs/tables/despotic/interpolated.npz` | Included in Git |
| Cloudy table | `inputs/tables/cloudy/emission.npz` | Included in Git |

Place the complete snapshot directory at the default path, or point `dataset`
in [configs/emission_process.yaml](configs/emission_process.yaml) to its existing
location:

```yaml
dataset: /absolute/path/to/plt0655228
```

The snapshot directory must include `Header`, `metadata.yaml`, and its data
subdirectories. The bundled tables are for the `plt0655228` reference snapshot.

## 4. Process

Run from the repository root:

```bash
python -m quokka2s.process_snapshot
```

Settings: [configs/emission_process.yaml](configs/emission_process.yaml).
Results: `output/plt0655228/processed/`.
For another run, choose a new `output_dir` in the process config.

## 5. Plot

```bash
python -m quokka2s.plot_emission_results
```

Settings: [configs/emission_plot.yaml](configs/emission_plot.yaml).
The default `products` path reads `output/plt0655228/processed/`.

Figures: `output/plt0655228/figures/` and
`output/plt0655228/figures_titled/`, each with `png/` and `pdf/` subdirectories.

## Example outputs

Hα luminosity images for the complete `plt0655228` snapshot, viewed along
the z axis. Both images use the same colour scale.

| Intrinsic | Dust attenuated |
|---|---|
| ![Intrinsic Halpha luminosity image](docs/examples/line_luminosity_halpha_intrinsic.png) | ![Dust-attenuated Halpha luminosity image](docs/examples/line_luminosity_halpha_attenuated.png) |

Integrated Hα spectrum before and after dust attenuation:

![Integrated Halpha spectrum](docs/examples/spectrum_halpha.png)

## Repository structure

```text
configs/       Process and plot settings
src/quokka2s/  Snapshot reading, emission calculations and plotting
inputs/        Simulation snapshots and lookup tables
output/        Processed data and generated figures
tools/         Table-building tools and additional figures
vendor/        External code and reference data
docs/          Usage, methods, code guide and example images
```

[Detailed usage](docs/usage.md) · [Methods](docs/emission_method.md) ·
[Code guide](docs/code_walkthrough.md)
