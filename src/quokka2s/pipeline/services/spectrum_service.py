"""SpectrumStore — task-local memoiser for 1D emission spectra.

Concept: a spectrum-building task instantiates a store inside its
``compute()`` method:

    from ..services import SpectrumStore
    store = SpectrumStore(context.provider)
    v_axis, dsigma_dv = store.get_spectrum(species='CO', los='x',
                                            phase='WIM', R=1e6)

The store keeps each ``(species, los, phase_label)`` pre-LSF spectrum after
its first build, so repeated calls within the same task hit the cache.
LSF (post-processing convolution) is cheap and applied per call.

Lifetime: a single ``compute()`` invocation.  The store is GC'd when
``compute()`` returns, freeing its internal lum/width/doppler caches.  We
deliberately do NOT share across tasks — that was the old pipeline-level
design and it accumulated ~12 GB of species fields in memory.
"""
from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Optional

import numpy as np
from yt.units.yt_array import YTArray, YTQuantity

from ..prep.physics_fields import build_integrated_spectrum
from ..spectrum_units import (
    SPEED_OF_LIGHT_CGS,
    convert_dsigma_dnu_to_dsigma_dv,
)
from ..tasks.integrated_spectrum import SPECIES_CFG, N_CHANNELS, V_RANGE_KMS
from ..utils import (
    PHASE_ORDER,
    apply_spectral_lsf,
    classify_temperature_phase,
)


# Bare cgs value for the numpy channel kernel, derived from yt's unit-checked
# physical constant in spectrum_units.py.
_C_CGS = float(SPEED_OF_LIGHT_CGS.value)
_SPECIES_ALIASES = {'CO': 'CO10'}


def temperature_selection_mask(
    temperature_K: np.ndarray,
    operator: str,
    cutoff_K: float,
) -> np.ndarray:
    """Return a temperature-selection mask.

    Supported operators are ``lt``/``gt`` for strict selections and
    ``le``/``ge`` for their exact complements. Thus ``gt`` paired with ``le``
    (or ``lt`` paired with ``ge``) covers every cell exactly once.
    """
    temperature_K = np.asarray(temperature_K)
    if operator == 'lt':
        return temperature_K < cutoff_K
    if operator == 'gt':
        return temperature_K > cutoff_K
    if operator == 'le':
        return temperature_K <= cutoff_K
    if operator == 'ge':
        return temperature_K >= cutoff_K
    raise ValueError(
        f"unknown temperature selection operator {operator!r}; "
        "expected 'lt', 'gt', 'le', or 'ge'"
    )


class SpectrumStore:
    """Memoised 1D-spectrum builder, task-local (one store per ``compute()``)."""

    # Which plane (perpendicular to the LOS) belongs to each LOS choice.
    _PLANE_FOR_LOS = {'x': 'yz', 'y': 'xz', 'z': 'xy'}

    def __init__(self, provider, species_config: Sequence[dict] | None = None):
        self.provider = provider

        # Most callers use the canonical emission species.  Dedicated tasks
        # can supply a small task-local configuration without adding their
        # diagnostic variants to every standard spectrum/overlay figure.
        configured_species = SPECIES_CFG if species_config is None else species_config

        # Lazy-loaded primitives — populated on first call that needs them.
        self._volume_3d: Optional[np.ndarray] = None              # cm^3
        self._plane_cell_area: dict[str, float] = {}              # cm^2 per LOS plane
        self._doppler: dict[str, np.ndarray] = {}                 # {'x': arr, 'y': arr}
        self._species_lum: dict[str, np.ndarray] = {}             # erg/s/cm^3
        self._species_width: dict[str, np.ndarray] = {}           # cm/s
        self._species_freq0: dict[str, float] = {}                # Hz scalar (constant per species)
        self._phase_masks: dict[str, dict[str, np.ndarray]] = {}

        # Field-name lookups from the selected species configuration.
        self._lum_field = {
            sp['name']: sp['lum_field'] for sp in configured_species
        }
        self._width_field = {
            sp['name']: sp['width_field'] for sp in configured_species
        }
        self._freq_field = {
            sp['name']: sp['freq_field'] for sp in configured_species
        }
        self._temperature_selection = {
            sp['name']: (
                sp['selection_temperature_field'],
                sp['selection_operator'],
                float(sp['selection_cutoff_K']),
            )
            for sp in configured_species
            if 'selection_temperature_field' in sp
        }

        # The actual store: (species, los, phase_label) → (v_axis, dsigma_dv_preLSF).
        # dsigma_dv carries yt units: Lsun/pc^2/(km/s).
        self._spectra: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}

        # Lock guarding the lazy-loaded primitives so that parallel callers
        # don't redundantly trigger get_slab_z on the same field.  We rely on
        # callers (tasks) dispatching DISTINCT (species, los, phase) keys to
        # the thread pool, so the spectrum-store dict itself doesn't need a
        # lock — only the shared primitive caches do.
        self._load_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────
    def get_spectrum(self, species: str, los: str, *,
                     phase: Optional[str] = None,
                     R: float = float('inf')) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(v_axis_kms, dsigma_dv)`` for the requested 1D spectrum.

        Parameters
        ----------
        species : name from the store's species configuration. Legacy ``CO``
                  is accepted as an alias of ``CO10`` in the default config.
        los     : 'x' | 'y'
        phase   : one of the 5 ISM phase labels (CNM, UNM, WNM, WIM, HIM),
                  or ``None`` for the all-cell 'total'.
        R       : LSF resolving power. ``inf`` ⇒ no convolution.
        """
        if los not in self._PLANE_FOR_LOS:
            raise ValueError(f"unknown LOS: {los!r}")
        if species not in self._lum_field:
            species = _SPECIES_ALIASES.get(species, species)
        phase_label = phase if phase is not None else 'total'
        if phase_label != 'total' and phase_label not in PHASE_ORDER:
            raise ValueError(f"unknown phase label: {phase_label!r}")

        key = (species, los, phase_label)
        if key not in self._spectra:
            self._spectra[key] = self._build(species, los, phase_label)
        v_axis, dsigma_pre = self._spectra[key]

        if R is not None and np.isfinite(R) and R > 0:
            dv = abs(v_axis[1] - v_axis[0])
            return v_axis, apply_spectral_lsf(dsigma_pre, dv, R, axis=0)
        return v_axis, dsigma_pre

    # ── Internals ─────────────────────────────────────────────────────────
    def _build(self, species: str, los: str, phase_label: str
               ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_primitives(species, los)

        lum_3d  = self._species_lum[species]
        width   = self._species_width[species]
        doppler = self._doppler[los]
        volume  = self._volume_3d
        nu_0    = self._species_freq0[species]

        # Channel grid covers ±V_RANGE_KMS around the rest frequency.
        v_range_cgs = V_RANGE_KMS * 1.0e5
        bw_hz       = nu_0 * (v_range_cgs / _C_CGS) * 2.0
        freq_edges  = np.linspace(nu_0 - bw_hz / 2, nu_0 + bw_hz / 2, N_CHANNELS + 1)
        freq_ctr    = 0.5 * (freq_edges[:-1] + freq_edges[1:])
        v_axis_kms  = (_C_CGS * (nu_0 - freq_ctr) / nu_0) * 1.0e-5

        # Per-cell luminosity (erg/s) and Doppler-shifted frequency.
        # The big transients get freed when this function returns.
        lum_per_cell = lum_3d * volume
        if phase_label != 'total':
            lum_per_cell = lum_per_cell * self._get_phase_masks(species)[phase_label]
        shifted = nu_0 * doppler  # freq is constant across cells, so freq*doppler == nu_0*doppler

        total_lum = build_integrated_spectrum(
            shifted, lum_per_cell, width, freq_edges, _C_CGS,
        )
        # Mean surface brightness = total luminosity / projected area of the
        # plane perpendicular to the LOS. Use the true grid dimensions so the
        # normalisation remains correct for non-cubic grids and los='z'.
        nx, ny, nz    = self._volume_3d.shape
        n_sightlines  = {'x': ny * nz, 'y': nx * nz, 'z': nx * ny}[los]
        plane         = self._PLANE_FOR_LOS[los]
        total_area_cm = n_sightlines * self._plane_cell_area[plane]
        # build_integrated_spectrum returns dL/dnu [erg/s/Hz].  Dividing by
        # projected source area therefore first gives dSigma_L/dnu.  Since the
        # horizontal axis above is velocity, apply |dnu/dv| before returning.
        spectral_luminosity = YTArray(total_lum, "erg/s/Hz")
        projected_area = YTQuantity(total_area_cm, "cm**2")
        dsigma_dnu = spectral_luminosity / projected_area
        dsigma_dv = convert_dsigma_dnu_to_dsigma_dv(dsigma_dnu, nu_0)

        print(f'[spectrum-store] built  ({species:>8s}, los={los}, '
              f'phase={phase_label:<6s})')
        return v_axis_kms, dsigma_dv

    def _ensure_primitives(self, species: str, los: str) -> None:
        """Load volume, doppler, and per-species fields if not already cached.

        Lock-protected so that 4 parallel `get_spectrum` callers requesting
        the same species don't trigger 4 disk reads."""
        with self._load_lock:
            if self._volume_3d is None:
                dx, _ = self.provider.get_slab_z(('boxlib', 'dx'))
                dy, _ = self.provider.get_slab_z(('boxlib', 'dy'))
                dz, _ = self.provider.get_slab_z(('boxlib', 'dz'))
                self._volume_3d = (dx * dy * dz).in_cgs().value
                self._plane_cell_area['yz'] = float((dy * dz)[0, 0, 0].in_units('cm**2').value)
                self._plane_cell_area['xz'] = float((dx * dz)[0, 0, 0].in_units('cm**2').value)
                self._plane_cell_area['xy'] = float((dx * dy)[0, 0, 0].in_units('cm**2').value)

            if los not in self._doppler:
                field = {'x': 'Bulk_Doppler_factor_x',
                         'y': 'Bulk_Doppler_factor_y',
                         'z': 'Bulk_Doppler_factor_z'}[los]
                doppler, _ = self.provider.get_slab_z(('gas', field))
                self._doppler[los] = np.asarray(doppler)

            if species not in self._species_lum:
                lum, _ = self.provider.get_slab_z(('gas', self._lum_field[species]))
                lum_values = np.asarray(lum.in_units('erg/s/cm**3'))
                if species in self._temperature_selection:
                    field, operator, cutoff_K = self._temperature_selection[species]
                    temperature, _ = self.provider.get_slab_z(('gas', field))
                    mask = temperature_selection_mask(
                        np.asarray(temperature.in_units('K')),
                        operator,
                        cutoff_K,
                    )
                    if mask.shape != lum_values.shape:
                        raise ValueError(
                            f'temperature selection for {species!r} has shape '
                            f'{mask.shape}, expected {lum_values.shape}'
                        )
                    # np.where creates a private masked luminosity array.  Do
                    # not mutate the provider/yt cached field in place.
                    lum_values = np.where(mask, lum_values, 0.0)
                self._species_lum[species] = lum_values
                width, _ = self.provider.get_slab_z(('gas', self._width_field[species]))
                self._species_width[species] = np.asarray(width.in_units('cm/s'))
                freq, _ = self.provider.get_slab_z(('gas', self._freq_field[species]))
                self._species_freq0[species] = float(freq.in_units('Hz')[0, 0, 0])

    def _get_phase_masks(self, species: str) -> dict[str, np.ndarray]:
        """Classify phases using the temperature assigned to each result."""
        temperature_fields = {
            'CO10': 'temperature_despotic',
            'CO21': 'temperature_despotic',
            'C+': 'temperature_quokka',
            'H_alpha': 'temperature_two_regime',
            'HI': 'temperature_quokka',
            'HI_DESPOTIC': 'temperature_despotic',
            'HI_QUOKKA': 'temperature_quokka',
        }
        with self._load_lock:
            if species not in self._phase_masks:
                T, _ = self.provider.get_slab_z(
                    ('gas', temperature_fields[species])
                )
                self._phase_masks[species] = classify_temperature_phase(
                    np.asarray(T.in_units('K'))
                )
            return self._phase_masks[species]


# Backward-compat alias so any external caller using the old name still works.
SpectrumCubeService = SpectrumStore
