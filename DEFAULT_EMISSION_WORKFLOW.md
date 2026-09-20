# Default-abundance emission products

`scripts/build_default_emission_products.py` is the explicit output entry point
for the accepted default-abundance setup. Historical output scripts and global
pipeline table paths are unchanged.

It reads the accepted interpolated DESPOTIC manifest, verifies the table and
exclusion hashes, and checks every fresh query against the exact shared exclusion
list. The original DESPOTIC solver failure mask is retained as provenance and is
not used to reject successfully interpolated values. The remaining unavailable
cells are omitted from every line, not assigned physical zero emissivity.

The temperature split uses `T_QUOKKA < 3000 K`:

| Lines | Cold branch | Hot branch |
| --- | --- | --- |
| CII | DESPOTIC | Cloudy |
| Halpha, HI 21 cm | Existing analytic expressions using DESPOTIC state | Cloudy |
| CIII 977/1907/1909, CIV 1548/1551 | Omitted (zero by prescription) | Cloudy |
| CO(1-0), CO(2-1) | DESPOTIC | DESPOTIC |

Atomic-line thermal widths use DESPOTIC temperature in cold cells and QUOKKA
temperature in hot cells. CO widths use DESPOTIC temperature in both branches.
The Jeans estimate uses actual simulation density and fixed mu=1. The original
default Cloudy table is reused only where the cell's model depth and all
contributing table nodes are capped. Its rounded cap of 3.086e20 cm differs from
100 pc by 0.01045%; this is a geometry difference, not a measured emission error.

The active eight-line table builder now explicitly supplies
`coolingMapHydrogenMassFraction = 0.7157683773530885`, matching the QUOKKA
rho-to-nH conversion. The adapted CIAOLoop uses `rho_J = nH*mH/X_H` with
fixed mu=1 and the unchanged 3.086e20 cm cap. This density conversion does
not change Cloudy or DESPOTIC elemental abundances.

The retained table was built with X_H=0.76. For snapshot `plt0655228`, all
343 interpolation nodes used by the hot branch remain capped under both
conversions: the smallest uncapped node length changes from 108.3453 to
105.1452 pc. The hot adapter checks both conversions before permitting
legacy reuse. This correction therefore requires no rebuild of the current
table or spectra. It does not establish equivalence for another snapshot
or grid; nodes near or below the cap can change. The existing 10 density
nodes and all other grid coordinates are retained.

Historical parameters lacking `coolingMapHydrogenMassFraction` do not
identify which runtime default was used. Their abundance/density metadata
must not be retroactively relabelled with the corrected value.

Example from the repository root:

```sh
python scripts/build_default_emission_products.py \
  --accepted-despotic output/despotic_default_parallel_20260918/interpolated/accepted_table.json \
  --cloudy-table data/cloudy_hm2012_attgrid_ism_nh21_cmb_cr_defaultabund_eightline_jeans_7x10x21.npz \
  --cloudy-audit output/default_table_reuse_20260918/cloudy_reuse_coverage.json \
  --dataset plt0655228 \
  --output-dir output/default_emission_20260919/full
```

Use a new output directory. `--max-slabs 1` runs a labelled diagnostic subset.
Default spectra have 300 channels over -200 to +200 km/s along z, with analytic
Gaussian channel integration. Spectra are not renormalized to hide luminosity
outside this velocity window. `spectra.npz` stores absolute dL/dv and integrated
luminosities separately for cold and hot gas. `spectra.png` and `spectra.pdf`
divide by the projected simulation area and therefore show surface luminosity,
not observer flux. `emission_report.json` records branch luminosities, omitted
mass, velocity-window losses, input/source hashes and coverage checks.

This is local emission with the inherited Cloudy escape treatment and DESPOTIC
LVG prescription. No additional foreground dust or transfer between simulation
cells is added. These spectra have not been spatially or instrumentally convolved.
