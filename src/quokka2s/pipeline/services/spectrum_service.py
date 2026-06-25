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
from typing import Optional

import numpy as np
import astropy.constants as _const
import astropy.units as _u

from ..prep.physics_fields import build_spectral_cube
from ..tasks.integrated_spectrum import SPECIES_CFG, N_CHANNELS, V_RANGE_KMS
from ..utils import (
    PHASE_ORDER,
    apply_spectral_lsf,
    classify_temperature_phase,
)


# Speed of light in cgs, derived from astropy (not hardcoded) → a float for the
# numpy spectral-cube path.  The assertion catches a unit slip.
_c_q = _const.c.to(_u.cm / _u.s)
assert _c_q.unit == _u.cm / _u.s
_C_CGS = float(_c_q.value)   # = 2.99792458e10 cm/s


class SpectrumStore:
    """Memoised 1D-spectrum builder, task-local (one store per ``compute()``)."""

    # Which plane (perpendicular to the LOS) belongs to each LOS choice.
    _PLANE_FOR_LOS = {'x': 'yz', 'y': 'xz', 'z': 'xy'}

    def __init__(self, provider):
        self.provider = provider

        # Lazy-loaded primitives — populated on first call that needs them.
        self._volume_3d: Optional[np.ndarray] = None              # cm^3
        self._plane_cell_area: dict[str, float] = {}              # cm^2 per LOS plane
        self._doppler: dict[str, np.ndarray] = {}                 # {'x': arr, 'y': arr}
        self._species_lum: dict[str, np.ndarray] = {}             # erg/s/cm^3
        self._species_width: dict[str, np.ndarray] = {}           # cm/s
        self._species_freq0: dict[str, float] = {}                # Hz scalar (constant per species)
        self._phase_masks: Optional[dict[str, np.ndarray]] = None

        # field-name lookups from SPECIES_CFG
        self._lum_field = {sp['name']: sp['lum_field']   for sp in SPECIES_CFG}
        self._width_field = {sp['name']: sp['width_field'] for sp in SPECIES_CFG}
        self._freq_field = {sp['name']: sp['freq_field']  for sp in SPECIES_CFG}

        # The actual store: (species, los, phase_label) → (v_axis, dsigma_dv_preLSF).
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
        species : 'CO' | 'C+' | 'H_alpha' | 'HI'
        los     : 'x' | 'y'
        phase   : one of the 5 ISM phase labels (CNM, UNM, WNM, WIM, HIM),
                  or ``None`` for the all-cell 'total'.
        R       : LSF resolving power. ``inf`` ⇒ no convolution.
        """
        if los not in self._PLANE_FOR_LOS:
            raise ValueError(f"unknown LOS: {los!r}")
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
            lum_per_cell = lum_per_cell * self._get_phase_masks()[phase_label]
        shifted = nu_0 * doppler  # freq is constant across cells, so freq*doppler == nu_0*doppler

        cube = build_spectral_cube(
            shifted, lum_per_cell, width, freq_edges, _C_CGS,
        )
        total_lum     = cube.sum(axis=(1, 2))
        # Mean surface brightness = total luminosity / projected area of the
        # plane PERPENDICULAR to the LOS.  build_spectral_cube always collapses
        # axis 0, so cube.shape[1:] == (ny, nz) for EVERY los — using it as the
        # sightline count is right only for los='x' (and 'y' when nx==ny).  Use
        # the true grid dims so los='z' (and any non-cubic grid) normalises right.
        nx, ny, nz    = self._volume_3d.shape
        n_sightlines  = {'x': ny * nz, 'y': nx * nz, 'z': nx * ny}[los]
        plane         = self._PLANE_FOR_LOS[los]
        total_area_cm = n_sightlines * self._plane_cell_area[plane]
        dsigma_dv     = total_lum / total_area_cm

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
                self._species_lum[species] = np.asarray(lum.in_units('erg/s/cm**3'))
                width, _ = self.provider.get_slab_z(('gas', self._width_field[species]))
                self._species_width[species] = np.asarray(width.in_units('cm/s'))
                freq, _ = self.provider.get_slab_z(('gas', self._freq_field[species]))
                self._species_freq0[species] = float(freq.in_units('Hz')[0, 0, 0])

    def _get_phase_masks(self) -> dict[str, np.ndarray]:
        """Classify T into 5 phase masks once per store lifetime.

        Uses ``temperature_two_regime`` (2026-06-18): T_DESPOTIC saturates at
        ~5×10⁴ K, so all real HIM cells (T_QUOKKA > 10⁵·⁵ K) were getting
        misclassified as WIM.  Switching to the unified T field keeps the
        spectrum's phase masks consistent with VelocityPhaseTask's PDFs.
        """
        with self._load_lock:
            if self._phase_masks is None:
                T, _ = self.provider.get_slab_z(('gas', 'temperature_two_regime'))
                self._phase_masks = classify_temperature_phase(np.asarray(T.in_units('K')))
            return self._phase_masks


# Backward-compat alias so any external caller using the old name still works.
SpectrumCubeService = SpectrumStore
