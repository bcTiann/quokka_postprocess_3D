"""Keep image, spectrum, gas-phase and per-line accounting totals.

Images, spectra and gas phases own their numerical algorithms. This module
groups their results and checks each product against its own cell totals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quokka2s.physics.line_emissivity import CIII_CIV_LINE_KEYS
from quokka2s.products.integrated_spectra import IntegratedSpectra, check_spectrum_luminosity
from quokka2s.products.gas_phase_velocity import GasPhaseVelocityAccumulator, check_phase_accounting
from quokka2s.products.line_luminosity_images import (
    LineLuminosityImageAccumulator,
    check_image_luminosity,
)

VELOCITY_RANGE_KMS = 200.
VELOCITY_CHANNELS = 400


@dataclass
class EmissionOutputs:
    """Numerical products returned by EmissionProducts.build_outputs().

    Attributes
    ----------
    full_snapshot : bool
        Whether the processed cell count equals the snapshot grid size.
    image_payload : dict
        Native images [erg/s], ordered (dust state, line, x, y).
    spectrum_payload : dict
        Spectra [erg/s/(km/s)], ordered (dust state, line, cold/hot, channel).
    phase_payload : dict
        Mass histograms of shape (6, Nchannel) [g]; group order is
        CNM, UNM, WNM, WIM, HIM, total.
    spectral_report, phase_report : dict
        Generated luminosity/window and gas-mass/moment summaries.
        Conservation checks are a separate products.check_outputs() action.
    """

    full_snapshot: bool
    image_payload: dict
    spectrum_payload: dict
    phase_payload: dict
    spectral_report: dict
    phase_report: dict


class EmissionProducts:
    """Running image/spectrum/phase sums; never retains individual cell arrays."""

    def __init__(self, snapshot, line_keys, spectral_workers):
        """Create image, spectrum and gas-phase accumulators for one run.

        Parameters
        ----------
        snapshot : Snapshot
            From load_processing_inputs(); shape supplies the native image size.
        line_keys : sequence of str
            From CellEmissionCalculator.line_keys; fixes the line order of every accumulator.
        spectral_workers : int
            Thread count for Gaussian channel integration within each batch.

        Notes
        -----
        Spectra and phase histograms use 400 channels over [-200, 200] km/s.
        Cell luminosity totals have shape (L, 2), ordered cold then hot.
        Only sums and statistics remain in this object, not cell arrays.

        Examples
        --------
        products = EmissionProducts(snapshot, emission_calculator.line_keys, spectral_workers=1)
        """
        self.line_keys = tuple(line_keys)
        # Retain only the small geometry values needed when finalizing.
        # Do not retain the dataset or lookup tables in a product accumulator.
        self.expected_cell_count = snapshot.cell_count  # Full box: 134,217,728.
        self.projected_area_cm2 = snapshot.projected_area_cm2  # Full x-y area.
        velocity_edges_kms = np.linspace(
            -VELOCITY_RANGE_KMS,
            VELOCITY_RANGE_KMS,
            VELOCITY_CHANNELS + 1,
        )

        self.images = LineLuminosityImageAccumulator(
            self.line_keys,       # ("cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909", "civ_1548", "civ_1551", "co10", "co21")
            snapshot.shape[:2],  # (256, 256): x and y pixel counts.
        )
        self.spectra = IntegratedSpectra(
            line_keys=self.line_keys,
            velocity_edges_kms=velocity_edges_kms,
            workers=spectral_workers,
        )
        self.phases = GasPhaseVelocityAccumulator(velocity_edges_kms)
        self.counts = {
            "all": 0,                      # Number of processed cells.
            "gas_temperature_missing": 0,  # No usable mixed temperature.
        }
        self.mass_g = {
            "all": 0.0,
            "cold": 0.0,
            "hot": 0.0,
            "gas_temperature_missing": 0.0,
        }
        # Missing light is counted separately for each line. For example, a
        # hot cell may have no CO emissivity while its Halpha is available.
        self.missing_emissivity_cells = {key: 0 for key in self.line_keys}
        self.missing_emissivity_mass_g = {key: 0.0 for key in self.line_keys}

        # 10 emission lines; cold and hot branches.
        self.intrinsic_luminosity_erg_s = np.zeros((len(self.line_keys), 2))
        self.attenuated_luminosity_erg_s = np.zeros_like(self.intrinsic_luminosity_erg_s)

        self.despotic_coordinate_clipped_cells = {
            "nH": 0,    # Cells whose density lookup coordinate was clipped.
            "NH": 0,    # Cells whose column lookup coordinate was clipped.
            "dVdr": 0,  # Cells whose velocity-gradient lookup coordinate was clipped.
        }

        self.cloudy_column_clipped_cells = {
            "below": 0,  # Cells below the minimum attenuation column.
            "above": 0,  # Cells above the maximum attenuation column.
        }
        self.failure_context = {}

    def record_batch(self, cells, emission):
        """Record independent cell totals after adding a batch to all products.

        Parameters
        ----------
        cells : CellBatch
            From SlabArrays.batch(); density_g_cm3 has shape (B,) [g/cm^3]
            and cell_volume_cm3 is the common cell volume [cm^3].
        emission : BatchEmission
            From CellEmissionCalculator.calculate(); carries cell masks, table clipping
            integer counts and named LineEmission records. Each record contains
            (B,) intrinsic/attenuated epsilon [erg/s/cm^3].

        Returns
        -------
        None
            Updates cell counts, gas masses [g] and line luminosities [erg/s].
            Each light total skips that line's missing emissivities. Gas-phase
            counts depend on mixed temperature, not on any line's availability.

        Examples
        --------
        products.record_batch(cells, emission)
        """
        self.record_lookup_clipping_counts(emission)
        self.record_line_luminosities(cells, emission)
        self.record_missing_line_emissivities(cells, emission)
        self.record_cell_counts_and_masses(cells, emission)

    def record_lookup_clipping_counts(self, emission):
        """Add lookup-coordinate clipping counts for retained cells only."""
        for key, count in emission.despotic_coordinate_clipped_cells.items():
            self.despotic_coordinate_clipped_cells[key] += count
        for key, count in emission.cloudy_column_clipped_cells.items():
            self.cloudy_column_clipped_cells[key] += count

    def record_line_luminosities(self, cells, emission):
        """Sum each line's available cell light [erg/s], keeping cold/hot separate.

        Both masks refer to the original B-cell batch. cold_cells classifies the
        gas regardless of table success. Each line then removes only its own
        missing emissivities, using the same rule as its image and spectrum.
        """
        for branch, selected_cells in enumerate((emission.cold_cells, ~emission.cold_cells)):
            intrinsic_luminosity, attenuated_luminosity = self.sum_named_line_luminosities(
                emission=emission,
                selected_cells=selected_cells,
                cell_volume_cm3=cells.cell_volume_cm3,
            )
            self.intrinsic_luminosity_erg_s[:, branch] += intrinsic_luminosity
            self.attenuated_luminosity_erg_s[:, branch] += attenuated_luminosity

    def sum_named_line_luminosities(self, emission, selected_cells, cell_volume_cm3):
        """Return independent intrinsic/attenuated line totals, each (L,) [erg/s].

        selected_cells is one cold/hot subset, shape (B,). A line's missing
        emissivities are removed from this selection before summing. These
        independent cell sums are later checked against images and spectra.
        """
        intrinsic_luminosity = np.zeros(len(self.line_keys))
        attenuated_luminosity = np.zeros(len(self.line_keys))
        for line_index, line_key in enumerate(self.line_keys):
            line = emission.lines[line_key]
            cells_with_emissivity = selected_cells & ~line.emissivity_is_missing
            intrinsic_luminosity[line_index] = (
                line.intrinsic_emissivity_erg_s_cm3[cells_with_emissivity].sum()
                * cell_volume_cm3
            )
            attenuated_luminosity[line_index] = (
                line.attenuated_emissivity_erg_s_cm3[cells_with_emissivity].sum()
                * cell_volume_cm3
            )
        return intrinsic_luminosity, attenuated_luminosity

    def record_missing_line_emissivities(self, cells, emission):
        """Count missing epsilon and its cell mass separately for every line.

        line.emissivity_is_missing is (B,) and includes no prescribed zeros.
        These small sums describe omitted light without a global exclusion mask.
        """
        for line_key in self.line_keys:
            missing = emission.lines[line_key].emissivity_is_missing
            self.missing_emissivity_cells[line_key] += int(np.count_nonzero(missing))
            self.missing_emissivity_mass_g[line_key] += float(
                cells.density_g_cm3[missing].sum() * cells.cell_volume_cm3,
            )

    def record_cell_counts_and_masses(self, cells, emission):
        """Record full gas masses and cells lacking a usable mixed temperature.

        A hot cell uses T_QUOKKA even when T_DESPOTIC is missing. Only cold
        cells need T_DESPOTIC to enter the gas-phase histogram.
        """
        cold_cells = emission.cold_cells
        mixed_temperature = np.where(
            cold_cells,
            emission.despotic_temperature_K,
            cells.temperature_QUOKKA_K,
        )
        missing_temperature = ~np.isfinite(mixed_temperature) | (mixed_temperature <= 0)
        self.counts["all"] += cold_cells.size
        self.counts["gas_temperature_missing"] += int(np.count_nonzero(missing_temperature))
        mass_selections = (
            ("all", np.ones(cold_cells.shape, dtype=bool)),
            ("cold", cold_cells),
            ("hot", ~cold_cells),
            ("gas_temperature_missing", missing_temperature),
        )
        for name, selected_cells in mass_selections:
            self.mass_g[name] += float(
                cells.density_g_cm3[selected_cells].sum() * cells.cell_volume_cm3,
            )

    def merge(self, other):
        """Merge a completed parallel batch into the main accumulated products.

        Parameters
        ----------
        other : EmissionProducts
            Batch sums on the same grid, line order and velocity channels.

        Returns
        -------
        None
            Updates the three products and shared counts/masses/luminosities.
            The caller merges in input order on the main thread.

        Examples
        --------
        products.merge(batch_products)
        """
        self.images.merge(other.images)
        self.spectra.merge(other.spectra)
        self.phases.merge(other.phases)
        self.intrinsic_luminosity_erg_s += other.intrinsic_luminosity_erg_s
        self.attenuated_luminosity_erg_s += other.attenuated_luminosity_erg_s
        for name in (
            'counts', 'mass_g', 'missing_emissivity_cells', 'missing_emissivity_mass_g',
            'despotic_coordinate_clipped_cells', 'cloudy_column_clipped_cells',
        ):
            target = getattr(self, name)
            source = getattr(other, name)
            for key in target:
                target[key] += source[key]
        self.failure_context = other.failure_context.copy()

    def build_outputs(self) -> EmissionOutputs:
        """Build images, spectra and gas-phase products; do not validate or save them.

        Uses the geometry and line order saved during construction. Returns
        EmissionOutputs with the numerical payloads and their descriptive reports.
        This does not change the accumulated cell sums.

        Example:
            outputs = products.build_outputs()
            image_conservation = products.check_outputs(outputs)
        """
        spectrum_payload, spectral_report = self.spectra.build_output(
            projected_area_cm2=self.projected_area_cm2,
        )
        image_payload = self.images.build_output()
        phase_payload, phase_report = self.phases.build_output()
        return EmissionOutputs(
            full_snapshot=self.counts['all'] == self.expected_cell_count,
            image_payload=image_payload,
            spectrum_payload=spectrum_payload,
            phase_payload=phase_payload,
            spectral_report=spectral_report,
            phase_report=phase_report,
        )

    def check_outputs(self, outputs: EmissionOutputs) -> dict:
        """Check generated products against independent cell totals before saving.

        outputs comes from build_outputs(). Checks masses, cold CIII/CIV,
        dust attenuation, spectrum/image luminosities and gas-phase accounting.
        Returns the image-conservation diagnostics for emission_report.json;
        other checks raise ValueError on failure. Does not change outputs.
        """
        self.check_output_luminosities(outputs=outputs)
        self.check_output_line_moments(spectrum_payload=outputs.spectrum_payload)
        self.check_accumulated_totals()
        # (2, Nline, 2): intrinsic/attenuated, line, cold/hot; units erg/s.
        expected_luminosity = np.stack((
            self.intrinsic_luminosity_erg_s,
            self.attenuated_luminosity_erg_s,
        ))
        check_spectrum_luminosity(outputs.spectrum_payload, expected_luminosity)
        image_conservation = self.check_line_products(
            image_payload=outputs.image_payload,
            spectrum_payload=outputs.spectrum_payload,
            expected_luminosity=expected_luminosity,
        )
        check_phase_accounting(
            groups=outputs.phase_report['groups'],
            mass_by_bin=outputs.phase_payload['histogram_mass_g'],
            gas_cell_count=self.counts['all'] - self.counts['gas_temperature_missing'],
            gas_mass_g=self.mass_g['all'] - self.mass_g['gas_temperature_missing'],
        )
        return image_conservation

    def check_output_luminosities(self, outputs: EmissionOutputs):
        """Reject nonfinite or negative light once, after all batches are summed.

        The batch readers and emission calculator already check their physical
        inputs. Here we check the finished arrays that will be saved, including
        independent cell sums and the light outside the spectral window.
        """
        luminosity_arrays = {
            "intrinsic cell luminosities": self.intrinsic_luminosity_erg_s,
            "attenuated cell luminosities": self.attenuated_luminosity_erg_s,
            "image pixels": outputs.image_payload["line_luminosity_image_erg_s"],
            "image totals": outputs.image_payload["total_luminosity_erg_s"],
            "spectral channels": outputs.spectrum_payload["dL_dv_erg_s_per_kms"],
            "combined spectral channels": outputs.spectrum_payload["total_dL_dv_erg_s_per_kms"],
            "spectrum input luminosities": outputs.spectrum_payload["input_luminosity_erg_s"],
            "spectrum window luminosities": outputs.spectrum_payload["captured_luminosity_erg_s"],
            "spectrum outside luminosities": outputs.spectrum_payload["outside_velocity_luminosity_erg_s"],
            "complete Gaussian luminosities": outputs.spectrum_payload["full_line_luminosity_erg_s"],
        }
        for name, values in luminosity_arrays.items():
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must be finite and nonnegative")

    def check_output_line_moments(self, spectrum_payload):
        """Check centroids and dispersions only for lines with nonzero light.

        Empty lines have NaN moments. Complete Gaussian moments use all cell
        light; window moments use only light in the saved velocity channels.
        Centroids may be negative, whereas dispersions and raw second moments
        must be nonnegative.
        """
        full_luminosity = spectrum_payload["full_line_luminosity_erg_s"]
        window_luminosity = spectrum_payload["captured_luminosity_erg_s"].sum(axis=-1)
        moment_arrays = (
            ("line_centroid_full_kms", full_luminosity, False),
            ("line_sigma_full_kms", full_luminosity, True),
            ("line_second_raw_moment_full_kms2", full_luminosity, True),
            ("line_centroid_window_kms", window_luminosity, False),
            ("line_sigma_window_kms", window_luminosity, True),
        )
        for name, luminosity, must_be_nonnegative in moment_arrays:
            occupied_line_values = spectrum_payload[name][luminosity > 0]
            if not np.isfinite(occupied_line_values).all():
                raise ValueError(f"{name} must be finite for lines with nonzero light")
            if must_be_nonnegative and np.any(occupied_line_values < 0):
                raise ValueError(f"{name} must be nonnegative")

    def check_accumulated_totals(self):
        """Check cell-level mass accounting and the selected line/dust rules.

        Reads the running totals stored on this object. Returns None; raises
        ValueError for inconsistent totals. Does not build or change products.
        """
        mass = self.mass_g
        if not np.isclose(mass['cold'] + mass['hot'], mass['all'], rtol=1e-12, atol=0):
            raise ValueError('Temperature-regime masses do not sum to the processed mass')
        for key in CIII_CIV_LINE_KEYS:
            line_index = self.line_keys.index(key)
            cold_luminosity = self.intrinsic_luminosity_erg_s[line_index, 0]
            if cold_luminosity != 0:
                raise ValueError('Cold CIII/CIV must be exactly zero')
        if np.any(self.attenuated_luminosity_erg_s >
                  self.intrinsic_luminosity_erg_s * (1 + 1e-12)):
            raise ValueError('Dust-transmitted luminosity exceeds intrinsic luminosity')

    def check_line_products(self, image_payload, spectrum_payload, expected_luminosity):
        """Check generated line products and return their image-conservation report.

        image_payload and spectrum_payload come from their accumulators'
        build_output() methods. expected_luminosity contains cell sums [erg/s] with
        shape (2, Nline, 2): dust state, line, temperature regime. This check
        runs after generation because it needs the finished image/spectrum arrays.
        """
        image_conservation = check_image_luminosity(
            image_payload['total_luminosity_erg_s'],
            expected_luminosity.sum(axis=-1),
            self.line_keys,
        )
        hi_index = self.line_keys.index('hi21')
        hi_images = image_payload['line_luminosity_image_erg_s'][:, hi_index]
        hi_spectra = spectrum_payload['dL_dv_erg_s_per_kms'][:, hi_index]
        if (not np.array_equal(hi_images[0], hi_images[1])
                or not np.array_equal(hi_spectra[0], hi_spectra[1])):
            raise ValueError('H I 21 cm must be unchanged by the adopted dust approximation')
        for field_name in (
            "line_centroid_window_kms",
            "line_sigma_window_kms",
            "line_centroid_full_kms",
            "line_sigma_full_kms",
        ):
            hi_values = spectrum_payload[field_name][:, hi_index]
            if not np.array_equal(hi_values[0], hi_values[1], equal_nan=True):
                raise ValueError(
                    "H I 21 cm moments must be unchanged by the adopted dust approximation"
                )
        return image_conservation
