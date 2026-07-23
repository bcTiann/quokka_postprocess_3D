# physics_models.py

import yt
import numpy as np
import astropy.constants as _const_ap
import astropy.units as _u_ap
from scipy.special import erf as scipy_erf
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg, amu, kpc
from ...analysis import along_sight_cumulation
from ...cii_chianti_lookup import CiiUpperFractionLookup
from . import config as cfg
from ...tables import load_table
from ...tables.lookup import TableLookup
from ...line_regimes import (
    electron_fraction_from_mean_molecular_weight,
    emitter_temperature_field_name,
    temperature_regime_masks,
)


# --- Fundamental Physical Constants ---
m_H = mh.in_cgs()
lambda_Halpha = 656.3e-7 * cm
h = planck_constant
c = float(_const_ap.c.to('cm/s').value) * (cm / s)   # speed of light from astropy (not hardcoded)
# eV → K conversion (= e / k_B), derived from astropy (was the hardcoded 11604.518)
_EV_TO_K = float((1.0 * _u_ap.eV / _const_ap.k_B).to('K').value)
TABLE_LOOKUP_CACHE: TableLookup | None = None
CII_UPPER_FRACTION_LOOKUP_CACHE: CiiUpperFractionLookup | None = None

# Lower bound on |∇·v|/3 used as the LVG dVdr field. Cells with smaller
# divergence are pinned to this floor — the table's dVdr axis bottoms
# out at 1e-19, so 1e-18 keeps every cell strictly inside the
# interpolation range. ~0.04% of plt263168 cells fall below this floor
# (quiescent halo), measured 2026-05-09.
DVDR_FLOOR = 1e-18

# Methodology temperature boundaries, both applied to T_QUOKKA:
#     T < 3000 K              → DESPOTIC
#     3000 K ≤ T < 1.307e4 K → x_e from QUOKKA's mean molecular weight
#     T ≥ 1.307e4 K          → two-stage C+/C++ Saha for carbon;
#                                prebuilt CHIANTI CII excitation tables retain
#                                their H/He CIE collider model
T_QK_TWO_REGIME_K = 3000.0
T_CIE_K = 1.307e4


def _temperature_regime_masks(T_quokka_K):
    """Return low/intermediate/high masks using the two constants above."""
    return temperature_regime_masks(
        T_quokka_K,
        T_QK_TWO_REGIME_K,
        T_CIE_K,
    )

# ─── H atomic / line constants ──────────────────────────────────────
# Per user rule [[no-silent-simplification]] (2026-06-20): the three
# derivable values (Saha prefactor, T_H_ION, NU_H_ALPHA) are computed
# at import time from astropy.constants — hard-coded values were
# unverifiable.  The two 21 cm values are pinned from refereed sources
# (no Python package ships them; CHIANTI/LAMDA do not cover the H 21 cm
# hyperfine line).  See `docs/line_emission_calculations.md`
# for context.
#
# Textbook Saha (Draine 2011 Eq 3.17):
#                            ⎛ 2π m_e k_B T ⎞^(3/2)
#     n_e · n_H+ / n_HI  =   ⎜──────────────⎟        · exp(−I_H / k_B T)
#                            ⎝       h²      ⎠
#                       ≡   K(T)                                    [cm⁻³]
#     K(T)  =  (K_SAHA_PREF · T)^(3/2)  ·  exp(−T_H_ION / T)        [cm⁻³]
def _build_h_atomic_constants():
    """Import-time builder.  Returns:
        K_SAHA_PREF : float, 2π m_e k_B / h²   [cm⁻² K⁻¹]
        T_H_ION     : float, I_H / k_B = Ryd·h·c / k_B   [K]   (= 13.6 eV / k_B)
        NU_H_ALPHA  : float, Balmer α (n=3→2) frequency [Hz],
                      computed with hydrogen reduced-mass Rydberg
        NU_HI_21    : float, 21 cm hyperfine line frequency [Hz],
                      NIST/NRAO published value 1420.405751768 MHz
        A_HI_21     : float, Einstein A for 21 cm hyperfine [s⁻¹],
                      Furlanetto+2006 Phys.Rep. & Pritchard&Loeb 2012
                      Rep.Prog.Phys. both quote 2.85×10⁻¹⁵ s⁻¹
    """
    import astropy.units as u
    from astropy import constants as const_ap

    K_SAHA = ((2 * np.pi * const_ap.m_e * const_ap.k_B / const_ap.h**2)
              .to('cm**-2 / K').value)
    T_H = (const_ap.Ryd * const_ap.h * const_ap.c / const_ap.k_B).to('K').value
    R_H = const_ap.Ryd / (1 + const_ap.m_e / const_ap.m_p)
    nu_Ha = (R_H * const_ap.c * (1.0/4.0 - 1.0/9.0)).to('Hz').value
    nu_21 = 1420.405751768e6          # Hz; NIST / NRAO / HandWiki (12 sf)
    A_21  = 2.85e-15                  # s⁻¹; Furlanetto+2006 line below Eq.14,
                                      #       Pritchard&Loeb 2012 just below Eq.6

    # Sanity-check against the (now-removed) hard-coded values: must be
    # within 1% of the textbook approximations we used to ship.
    assert abs(K_SAHA - 1.80e10)  / 1.80e10  < 0.01, K_SAHA
    assert abs(T_H    - 1.578e5)  / 1.578e5  < 0.01, T_H
    assert abs(nu_Ha  - 4.5681e14)/ 4.5681e14< 0.01, nu_Ha
    return K_SAHA, T_H, nu_Ha, nu_21, A_21


(K_SAHA_PREF, T_H_ION, NU_H_ALPHA, NU_HI_21, A_HI_21) = _build_h_atomic_constants()
print(f"[physics_fields] H constants: "
      f"K_SAHA_PREF={K_SAHA_PREF:.4e} cm⁻² K⁻¹, "
      f"T_H_ION={T_H_ION:.4e} K, "
      f"ν_Hα={NU_H_ALPHA:.4e} Hz, "
      f"ν_21={NU_HI_21:.10e} Hz (NIST), "
      f"A_21={A_HI_21:.3e} s⁻¹ (Furlanetto+2006)")


# ── CHIANTI atomic data for [C II] 158 μm hot-branch LTE  (2026-06-18) ──
# Built ONCE at module import via fiasco/CHIANTI; per-cell evaluation in
# _Cplus_luminosity_model uses np.interp on these grids.  Constants
# (A_ul, ν, T_star, g_u/l, I_C) are T-independent — extracted from the
# 158 μm transition (CHIANTI level 1 → 2 of C II, λ=157.74 μm).  See
# memory `[[fiasco-chianti-installed]]` for install state.
#
# Per user-stated rule `[[no-silent-simplification]]`: if fiasco / CHIANTI
# is missing we RAISE rather than fall back to analytic fits.
A_C_TOTAL = cfg.A_C   # gas-phase C per H (moved to config.py; GOW xC default = 1.6e-4)


def _build_cii_lte_atomic_tables():
    """Import-time builder. Returns:
        T_grid     : (n_T,) ndarray in K, log-spaced 10^2.5 .. 10^7.5
        U_C0_grid  : (n_T,) C I   partition function
        U_Cp_grid  : (n_T,) C II  partition function (Σ_i g_i exp(-E_i/kT))
        U_Cpp_grid : (n_T,) C III partition function (added 2026-06-18
                     to handle C+ → C++ at high T)
        I_C_eV     : float, C I  → C II  ionisation potential [eV]  (≈11.26)
        I_C2_eV    : float, C II → C III ionisation potential [eV]  (≈24.38)
        A_ul       : float, 158 μm spontaneous-emission rate [s⁻¹]
        nu_Hz      : float, transition frequency [Hz]
        T_star_K   : float, ΔE/k_B of the transition [K]
        g_l, g_u   : floats, lower/upper level statistical weights
    """
    import fiasco
    import astropy.units as u
    from astropy import constants as const_ap

    T_grid_q = np.logspace(2.5, 7.5, 600) * u.K
    T_grid   = T_grid_q.to('K').value

    cii  = fiasco.Ion('C 2', T_grid_q)
    ci   = fiasco.Ion('C 1', T_grid_q)
    ciii = fiasco.Ion('C 3', T_grid_q)        # C++ for the 2-stage Saha

    # U(T) = Σ_i g_i exp(-E_i/kT) — full CHIANTI level lists
    beta = (1.0 / (const_ap.k_B * T_grid_q)).to('1/erg')           # (n_T,)
    E_cii  = cii .levels.energy.to('erg').value[:, None]            # (n_lev,1)
    E_ci   = ci  .levels.energy.to('erg').value[:, None]
    E_ciii = ciii.levels.energy.to('erg').value[:, None]
    g_cii  = cii .levels.weight[:, None]
    g_ci   = ci  .levels.weight[:, None]
    g_ciii = ciii.levels.weight[:, None]
    exp_cii  = np.exp(-(E_cii  * beta.value[None, :]))              # (n_lev, n_T)
    exp_ci   = np.exp(-(E_ci   * beta.value[None, :]))
    exp_ciii = np.exp(-(E_ciii * beta.value[None, :]))
    U_Cp_grid  = (g_cii  * exp_cii ).sum(axis=0)                    # (n_T,)
    U_C0_grid  = (g_ci   * exp_ci  ).sum(axis=0)
    U_Cpp_grid = (g_ciii * exp_ciii).sum(axis=0)

    # Ionisation potentials — T-independent
    I_C_eV  = float(np.atleast_1d(fiasco.Ion('C 1', np.array([8000.0])*u.K)
                    .ionization_potential.to(u.eV).value)[0])       # ≈ 11.26
    I_C2_eV = float(np.atleast_1d(fiasco.Ion('C 2', np.array([8000.0])*u.K)
                    .ionization_potential.to(u.eV).value)[0])       # ≈ 24.38

    # 158 μm transition: CHIANTI 1-indexed levels 1 → 2 (ground 2P_{1/2} → 2P_{3/2})
    lo = cii.transitions.lower_level
    up = cii.transitions.upper_level
    sel = np.where((lo == 1) & (up == 2))[0]
    if sel.size == 0:
        raise RuntimeError("CHIANTI C II: 158 μm transition (1→2) not found.")
    i158 = int(sel[0])
    A_ul     = float(cii.transitions.A[i158].to('1/s').value)
    nu_Hz    = float((cii.transitions.delta_energy[i158] / const_ap.h).to('Hz').value)
    T_star_K = float((cii.transitions.delta_energy[i158] / const_ap.k_B).to('K').value)
    g_l      = float(cii.levels.weight[0])   # level 0 in numpy = CHIANTI level 1
    g_u      = float(cii.levels.weight[1])   # level 1 in numpy = CHIANTI level 2

    return (T_grid, U_C0_grid, U_Cp_grid, U_Cpp_grid,
            I_C_eV, I_C2_eV, A_ul, nu_Hz, T_star_K, g_l, g_u)


# Module-level cache, built at import; fail loud if CHIANTI absent.
(_CII_T_GRID, _CII_U_C0, _CII_U_CP, _CII_U_CPP,
 _CII_I_C_EV, _CII_I_C2_EV, _CII_A_UL, _CII_NU_HZ,
 _CII_T_STAR, _CII_G_L, _CII_G_U) = _build_cii_lte_atomic_tables()
_CII_T_C_K  = _CII_I_C_EV  * _EV_TO_K   # eV → K via k_B (astropy-derived)
_CII_T_C2_K = _CII_I_C2_EV * _EV_TO_K   # C+ → C++ ionisation T scale
print(f"[physics_fields] CHIANTI [C II] LTE tables built: "
      f"A_ul={_CII_A_UL:.3e} s⁻¹, ν={_CII_NU_HZ:.3e} Hz, "
      f"T*={_CII_T_STAR:.3f} K, g_l={_CII_G_L:.0f}, g_u={_CII_G_U:.0f}, "
      f"I_C={_CII_I_C_EV:.4f} eV, I_C2={_CII_I_C2_EV:.4f} eV")


def _temperature_two_regime(field, data):
    """Two-regime per-cell temperature retained for H lines and diagnostics.

    For each cell:
        T_use = T_DESPOTIC    if T_QUOKKA <  T_QK_TWO_REGIME_K (= 3000 K)
        T_use = T_QUOKKA      if T_QUOKKA >= T_QK_TWO_REGIME_K

    Cold cells (T_QK<3000K) get DESPOTIC's chemistry-thermal-equilibrium
    temperature; hot cells (T_QK>=3000K) keep QUOKKA's sim temperature
    because DESPOTIC's 3D-table tg_final saturates around ~5×10^4 K.

    Cheap derived field (just np.where), not in CACHED_FIELDS.  CO and C+
    products deliberately do not use it: CO uses T_DESPOTIC and C+ uses
    T_QUOKKA. Boundary convention: T_QK == 3000 K → hot branch.
    """
    T_qk  = data[('gas', 'temperature_quokka')].to('K').value
    T_dsp = data[('gas', 'temperature_despotic')].to('K').value
    return yt.YTArray(
        np.where(T_qk < T_QK_TWO_REGIME_K, T_dsp, T_qk),
        'K',
    )


def ensure_table_lookup(path: str | None) -> TableLookup:
    global TABLE_LOOKUP_CACHE
    if TABLE_LOOKUP_CACHE is None:
        table = load_table(path or cfg.DESPOTIC_TABLE_PATH)
        TABLE_LOOKUP_CACHE = TableLookup(table)
    return TABLE_LOOKUP_CACHE


def _number_density_H(field, data):
    # 返回"总 H 核数密度" n_H_tot = n_HI + n_H+ + 2*n_H2
    # ρ·X_H/m_H 自动满足: H₂ 分子质量 ≈ 2·m_H, 每个分子贡献 2 个 H 核
    # 与 DESPOTIC "per H nucleus" 约定一致 (despotic/composition.py computeEint)
    density_3d = data[('gas', 'density')].in_cgs()
    n_H_3d = (density_3d * cfg.X_H) / m_H
    return n_H_3d.to('cm**-3')



def _column_density_H(field, data):
    density_3d = data[('gas', 'density')].in_cgs()

    # Check if data is 1D (flattened cells) - common in PhasePlot or certain yt operations
    if density_3d.ndim == 1:
        # For 1D data, we cannot compute column density along axes
        # Return a reasonable estimate based on local density and a characteristic length
        # This is a fallback for when full 3D data is not available
        n_H = (density_3d * cfg.X_H) / m_H
        # Use a simple estimate: n_H * characteristic_length
        # For lack of better info, use domain size / sqrt(ndata) as characteristic length
        ds = data.ds
        domain_size = (ds.domain_right_edge - ds.domain_left_edge)
        char_length = (domain_size.prod() ** (1/3)) / (len(n_H) ** (1/3))
        return (n_H * char_length).to('cm**-2')

    dx_3d = data[("boxlib", "dx")].in_cgs()
    dy_3d = data[("boxlib", "dy")].in_cgs()
    dz_3d = data[("boxlib", "dz")].in_cgs()

    n_H_3d = (density_3d * cfg.X_H) / m_H

    # Lateral (x, y) extension: shearing-box faces are periodic BCs, not
    # physical edges.  Add  L_ext * <n_H>(z)  to each ±x, ±y ray, with
    # <n_H>(z) the (x, y)-mean number density at the cell's own height.
    # ±z gets no extension — the box already spans the stratified disk.
    # See cfg.COLUMN_EXTENSION_LATERAL_KPC for the rationale.
    L_ext_kpc = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
    if L_ext_kpc > 0.0:
        L_ext_qty   = (L_ext_kpc * kpc).in_units('cm')        # unyt, cm
        n_bar_z     = n_H_3d.mean(axis=(0, 1))                # unyt, cm^-3
        N_ext_lat_3d = (L_ext_qty * n_bar_z)[None, None, :]   # unyt, cm^-2
    else:
        N_ext_lat_3d = None

    # Streaming harmonic mean: H = 6 / Σ_k (1/N_k).
    # Build one directional cumulation at a time, fold 1/N_k into the
    # reciprocal accumulator, then free it. Function-internal peak: ~5 GB
    # (vs ~22 GB for the original stack-then-aggregate version at down=1).
    mean_method = getattr(cfg, 'COLUMN_DENSITY_MEAN', 'harmonic')
    accum = None
    for axis, sign, dxyz, lateral in (
        ("x", "+", dx_3d, True),
        ("x", "-", dx_3d, True),
        ("y", "+", dy_3d, True),
        ("y", "-", dy_3d, True),
        ("z", "+", dz_3d, False),
        ("z", "-", dz_3d, False),
    ):
        N = along_sight_cumulation(n_H_3d * dxyz, axis=axis, sign=sign)
        if lateral and N_ext_lat_3d is not None:
            N = N + N_ext_lat_3d
        # Streaming accumulator: depends on the chosen combination method.
        #   arithmetic -> Σ N_k       (finalised as Σ/6)
        #   harmonic   -> Σ 1/N_k     (finalised as 6/Σ)
        #   max / min  -> running max / min of N_k       (no finalisation)
        if mean_method == 'arithmetic':
            inc = N
        elif mean_method == 'max':
            accum = N.copy() if accum is None else np.maximum(accum, N)
            del N
            continue
        elif mean_method == 'min':
            accum = N.copy() if accum is None else np.minimum(accum, N)
            del N
            continue
        else:                                       # harmonic (default)
            inc = 1.0 / N
        del N
        accum = inc if accum is None else accum + inc
        del inc
    if mean_method == 'arithmetic':
        return (accum / 6.0).to('cm**-2')          # (1/6) Σ N_k
    if mean_method in ('max', 'min'):
        return accum.to('cm**-2')                   # running max / min over 6 directions
    return (6.0 / accum).to('cm**-2')               # 6 / Σ(1/N_k)


def _dVdr_lvg(field, data):
    """LVG radial-gradient proxy: |∇·v|/3 with central differences.

    Equivalent to dv_r/dr for a uniformly expanding/contracting cell.
    Cells with |∇·v|/3 < DVDR_FLOOR are pinned to the floor so the
    table interpolation never sees log10(0).
    """
    density = data[('gas', 'density')]
    if density.ndim == 1:
        # 1D fallback (e.g., PhasePlot context) — no spatial structure
        # available; assume a quiescent-cell floor so the field is finite.
        return np.full_like(density.in_cgs().value, DVDR_FLOOR) / s

    vx = data[('gas', 'velocity_x')].in_units('cm/s').v
    vy = data[('gas', 'velocity_y')].in_units('cm/s').v
    vz = data[('gas', 'velocity_z')].in_units('cm/s').v
    dx = float(data[('boxlib', 'dx')].in_units('cm').v.flat[0])
    dy = float(data[('boxlib', 'dy')].in_units('cm').v.flat[0])
    dz = float(data[('boxlib', 'dz')].in_units('cm').v.flat[0])

    div = (np.gradient(vx, dx, axis=0)
           + np.gradient(vy, dy, axis=1)
           + np.gradient(vz, dz, axis=2))
    out = np.maximum(np.abs(div) / 3.0, DVDR_FLOOR)
    return out / s


# --- YT Derived Fields ---


def _internal_energy_density(field, data):
    """QUOKKA gas internal energy density = total − kinetic (pure hydro, no B).

    QUOKKA's on-disk ('boxlib','gasEnergy') is the TOTAL energy density (yt's
    ('gas','total_energy_density') aliases it); there is no internal-energy
    field on disk.  Subtracting the kinetic part recovers E_int [erg/cm^3].
    """
    e_tot = data[('gas', 'total_energy_density')].in_cgs()
    e_kin = data[('gas', 'kinetic_energy_density')].in_cgs()
    return (e_tot - e_kin).in_cgs()


def _clip_to_table_domain(lookup, n_H, colDen_H, dVdr):
    """Clamp (n_H, N_H, dVdr) to the DESPOTIC table's covered domain so the
    trilinear interpolation never extrapolates past the grid edges.  Returns
    (n_H_safe, col_safe, dV_safe) as bare float arrays.  Factored out of the
    four call sites (temperature, line luminosity, number density, C+ cold
    branch) that each repeated this identical 6-line clamp."""
    t = lookup.table
    n_H_safe = np.clip(n_H,      t.nH_values.min(),          t.nH_values.max())
    col_safe = np.clip(colDen_H, t.col_density_values.min(), t.col_density_values.max())
    dV_safe  = np.clip(dVdr,     t.dVdr_values.min(),        t.dVdr_values.max())
    return n_H_safe, col_safe, dV_safe


def _table_emissivity(lookup, species, n_H, colDen_H, dVdr):
    """Volumetric line emissivity ε = n_H · lumPerH [erg/s/cm³] from the 3D
    DESPOTIC LAMDA table at (n_H, N_H, dVdr), with inputs clamped to the table
    domain.  Returns a bare float array (no yt units).  Shared by
    `_make_luminosity_field` and the C+ two-regime cold branch so the two stay
    bit-identical."""
    n_H_safe, col_safe, dV_safe = _clip_to_table_domain(lookup, n_H, colDen_H, dVdr)
    lumPerH = np.nan_to_num(
        np.asarray(lookup.line_field(species, "lumPerH", n_H_safe, col_safe, dV_safe),
                   dtype=float),
        nan=0.0,
    )
    return n_H_safe * lumPerH


def _temperature_despotic(field, data):
    """Gas temperature from the 3D DESPOTIC table's tg_final at (n_H, N_H,
    dVdr).  Saturates at ~5×10⁴ K because that's the upper bound of DESPOTIC's
    thermal-equilibrium solver; hot ionised gas is handled per-line in the
    multi-regime functions (e.g. _Halpha_luminosity) using T_QUOKKA directly,
    not by overriding this field.
    """
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)

    n_H      = data[('gas', 'number_density_H')].in_cgs().value
    colDen_H = data[('gas', 'column_density_H')].in_cgs().value
    dVdr     = data[('gas', 'dVdr_lvg')].in_cgs().value

    n_H_safe, col_safe, dV_safe = _clip_to_table_domain(lookup, n_H, colDen_H, dVdr)

    tg_final = lookup.temperature(n_H_safe, col_safe, dV_safe)
    return tg_final * K


def _build_spectral_cube_v0_legacy(
    shifted_freq_val: np.ndarray,
    lum_val: np.ndarray,
    thermal_val: np.ndarray,
    freq_edges_hz: np.ndarray,
    c_cms: float,
) -> np.ndarray:
    """LEGACY (pre-2026-06-08) build_spectral_cube. Kept ONLY for bitwise
    verification of the V3 erf-at-edges replacement below.  Do not use in
    production paths — V3 is mathematically equivalent and ~1.8× faster
    (each interior frequency edge was being erf'd twice; V3 erfs each edge
    once and differences yield bin_frac).  Verified bitwise identical on
    synthetic ISM-like inputs across multiple seeds + edge cases.
    """
    n_channels = len(freq_edges_hz) - 1
    nx, ny, nz = shifted_freq_val.shape
    spec_cube = np.zeros((n_channels, ny, nz))

    delta_nu_bin = float(freq_edges_hz[1] - freq_edges_hz[0])
    CHUNK = 150

    for ch0 in range(0, n_channels, CHUNK):
        ch1 = min(ch0 + CHUNK, n_channels)
        nu_lo = freq_edges_hz[ch0:ch1][:, None, None]
        nu_hi = freq_edges_hz[ch0 + 1:ch1 + 1][:, None, None]

        for i in range(nx):
            nu_gas  = shifted_freq_val[i, :, :][None, :, :]
            lum_gas = lum_val[i, :, :][None, :, :]
            sigma_v = np.maximum(thermal_val[i, :, :], 1.0)[None, :, :]
            sigma_nu = nu_gas * (sigma_v / c_cms)

            sqrt2_sigma = np.sqrt(2.0) * sigma_nu
            x_lo = (nu_lo - nu_gas) / sqrt2_sigma
            x_hi = (nu_hi - nu_gas) / sqrt2_sigma
            bin_frac = 0.5 * (scipy_erf(x_hi) - scipy_erf(x_lo))

            spec_cube[ch0:ch1] += lum_gas * bin_frac / delta_nu_bin

    return spec_cube


def build_spectral_cube(
    shifted_freq_val: np.ndarray,
    lum_val: np.ndarray,
    thermal_val: np.ndarray,
    freq_edges_hz: np.ndarray,
    c_cms: float,
) -> np.ndarray:
    """Build a spectral cube using analytic erf integration over each frequency bin.

    Each cell's line profile is a Gaussian centred at its Doppler-shifted
    frequency with thermal width sigma_nu = nu_gas * (sigma_v / c).
    Rather than sampling the Gaussian at the bin centre (which aliases when
    the line is narrower than the bin), we integrate analytically:

        bin_frac[k] = 0.5 * [erf(x_hi) - erf(x_lo)]
        x = (nu_edge - nu_gas) / (sqrt(2) * sigma_nu)

    Luminosity is conserved exactly: sum_k bin_frac[k] == 1 per cell.

    2026-06-08 V3 erf-at-edges optimisation: the legacy implementation
    computed erf(x_hi) and erf(x_lo) for every channel, double-evaluating
    each interior edge (each edge is the x_hi of channel k AND the x_lo of
    channel k+1).  V3 evaluates erf at every edge exactly once, then takes
    adjacent differences to recover bin_frac.  Same erf inputs → bitwise
    identical outputs, but ~half the erf work (verified ~1.8× faster on
    medium-size benchmarks).  Legacy version retained as
    `_build_spectral_cube_v0_legacy` for verification.

    Parameters
    ----------
    shifted_freq_val : (nx, ny, nz)  Doppler-shifted cell frequency [Hz]
    lum_val          : (nx, ny, nz)  cell luminosity [erg/s]
    thermal_val      : (nx, ny, nz)  thermal velocity width [cm/s]
    freq_edges_hz    : (n_channels+1,)  bin edges [Hz], uniform spacing
    c_cms            : float  speed of light [cm/s]

    Returns
    -------
    spec_cube : (n_channels, ny, nz)  spectral density [erg/s/Hz]
    """
    n_channels = len(freq_edges_hz) - 1
    nx, ny, nz = shifted_freq_val.shape
    spec_cube = np.zeros((n_channels, ny, nz))

    delta_nu_bin = float(freq_edges_hz[1] - freq_edges_hz[0])
    sqrt2 = np.sqrt(2.0)

    # Chunk size in CHANNELS (so chunk has CHUNK+1 edges).  150 channels
    # → ~3 GB transient at down=1 (n_ch=300, ny=256, nz=2048), same budget
    # as the legacy CHUNK.
    CHUNK = 150

    for i in range(nx):
        # Hoist invariants out of the inner channel-chunk loop.  Compute
        # sqrt2_sigma in the EXACT SAME ORDER as the legacy code so the
        # bitwise result is preserved (floating-point multiplication is
        # not associative — reordering would introduce ulp-level drift).
        nu_gas      = shifted_freq_val[i, :, :][None, :, :]    # (1, ny, nz)
        lum_gas     = lum_val[i, :, :][None, :, :]
        sigma_v     = np.maximum(thermal_val[i, :, :], 1.0)[None, :, :]
        sigma_nu    = nu_gas * (sigma_v / c_cms)
        sqrt2_sigma = np.sqrt(2.0) * sigma_nu

        for ch0 in range(0, n_channels, CHUNK):
            ch1 = min(ch0 + CHUNK, n_channels)
            # Edges that bound this channel chunk: [ch0, ch0+1, ..., ch1]
            edges        = freq_edges_hz[ch0:ch1 + 1][:, None, None]       # (n_edges, 1, 1)
            x_edges      = (edges - nu_gas) / sqrt2_sigma                  # (n_edges, ny, nz)
            erf_at_edges = scipy_erf(x_edges)                              # ½× erf calls vs legacy
            bin_frac     = 0.5 * (erf_at_edges[1:] - erf_at_edges[:-1])    # (chunk, ny, nz)
            spec_cube[ch0:ch1] += lum_gas * bin_frac / delta_nu_bin

    return spec_cube


def build_integrated_spectrum(
    shifted_freq_val: np.ndarray,
    lum_val: np.ndarray,
    thermal_val: np.ndarray,
    freq_edges_hz: np.ndarray,
    c_cms: float,
    *,
    cell_chunk: int = 65536,
) -> np.ndarray:
    """Build a spatially integrated 1D spectrum without allocating a cube.

    This uses the same analytic Gaussian-bin integral as
    :func:`build_spectral_cube`, but sums each cell chunk directly into the
    channel axis.  It is the appropriate path when no channel map, PV diagram,
    or per-pixel spectrum is required.

    Parameters are identical to ``build_spectral_cube``. ``cell_chunk`` limits
    the largest temporary array to roughly
    ``(CHANNEL_CHUNK + 1) * cell_chunk`` floating-point values.

    Returns
    -------
    integrated_spectrum : (n_channels,) spectral luminosity [erg/s/Hz]
    """
    shifted = np.asarray(shifted_freq_val).reshape(-1)
    luminosity = np.asarray(lum_val).reshape(-1)
    thermal = np.asarray(thermal_val).reshape(-1)
    if shifted.shape != luminosity.shape or shifted.shape != thermal.shape:
        raise ValueError(
            'shifted frequency, luminosity, and thermal width must have '
            'matching shapes'
        )
    if cell_chunk <= 0:
        raise ValueError('cell_chunk must be positive')

    n_channels = len(freq_edges_hz) - 1
    integrated = np.zeros(n_channels, dtype=np.result_type(luminosity, float))
    delta_nu_bin = float(freq_edges_hz[1] - freq_edges_hz[0])
    channel_chunk = 150

    for cell0 in range(0, shifted.size, cell_chunk):
        cell1 = min(cell0 + cell_chunk, shifted.size)
        nu_gas = shifted[cell0:cell1][None, :]
        lum_gas = luminosity[cell0:cell1]
        sigma_v = np.maximum(thermal[cell0:cell1], 1.0)[None, :]
        sqrt2_sigma = np.sqrt(2.0) * nu_gas * (sigma_v / c_cms)

        for ch0 in range(0, n_channels, channel_chunk):
            ch1 = min(ch0 + channel_chunk, n_channels)
            edges = freq_edges_hz[ch0:ch1 + 1, None]
            erf_at_edges = (edges - nu_gas) / sqrt2_sigma
            scipy_erf(erf_at_edges, out=erf_at_edges)
            bin_frac = 0.5 * (erf_at_edges[1:] - erf_at_edges[:-1])
            weighted_sum = np.einsum(
                'kc,c->k', bin_frac, lum_gas, optimize=False,
            )
            integrated[ch0:ch1] += weighted_sum / delta_nu_bin

    return integrated
    
def _make_luminosity_field(species: str):
    """3D DESPOTIC LAMDA-table line luminosity, returned as volumetric
    emissivity (erg/s/cm³). No temperature gate is applied here.  The only
    registered caller is CO; its table ``lumPerH`` and ``tg_final`` come from
    the same DESPOTIC thermal/chemical solution at (n_H, N_H, dVdr), so its
    luminosity temperature is intrinsically T_DESPOTIC.
    """
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')

    def _field(field, data):
        n_H      = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        dVdr     = data[('gas','dVdr_lvg')].in_cgs().value

        return _table_emissivity(lookup, species, n_H, colDen_H, dVdr) * (erg / s / cm**3)

    _field.__name__ = f"_luminosity_{yt_safe_name}"
    return yt_safe_name, _field


def _make_number_density_field(species: str, lamda_token: str | None = None):
    """Factory for ('gas', <species>) yt fields backed by the DESPOTIC 3D table.

    species      : the yt-field name (e.g. 'HI', 'H+', 'CO', 'C', 'C+', 'e-')
    lamda_token  : the species name LAMDA / DESPOTIC's chemistry network uses
                   internally.  Defaults to `species` when None.  The only
                   case where they differ is HI: LAMDA's `H.dat` calls it `H`
                   (= neutral atomic H by convention), but we register it on
                   yt as `HI` to remove the "is this total H or neutral H?"
                   ambiguity (user feedback 2026-06-20).
    """
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    token = lamda_token if lamda_token is not None else species

    def _field(field, data):
        n_H      = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        dVdr     = data[('gas','dVdr_lvg')].in_cgs().value

        n_H_safe, col_safe, dV_safe = _clip_to_table_domain(lookup, n_H, colDen_H, dVdr)

        val = np.nan_to_num(
            lookup.number_densities([token], n_H_safe, col_safe, dV_safe)[token],
            nan=0.0,
        )
        return val * cm**-3

    _field.__name__ = f"_number_density_{yt_safe_name}"
    return yt_safe_name, _field

def _Halpha_luminosity(field, data):
    """
    H-alpha Luminosity Density [erg / s / cm**3].  Draine (2011) Eq. 14.6:
        ε_Hα = 0.45 · E_Hα · α_B(T) · n_e · n_H+

    Two-regime hydrogen treatment:
      - **Low** (T_QUOKKA < 3000 K):
            n_e, n_H+ from the DESPOTIC 3D table;  α_B at T_DESPOTIC.
      - **T_QUOKKA ≥ 3000 K**:
            infer x_e from QUOKKA's e_int, rho, and T_two_regime;
            if x_e <= 1, set x_H+=x_e; otherwise set x_H+=1.
            Then n_e=x_e n_H and n_H+=x_H+ n_H.

    α_B uses temperature_two_regime: T_DESPOTIC below 3000 K and T_QUOKKA
    at and above the boundary.
    """
    # Photon energy is a constant.
    E_Halpha = ((h * c) / lambda_Halpha).in_cgs().value   # erg

    # Pull temperatures + densities as raw cgs arrays.
    T_qk = data[('gas', 'temperature_quokka')].to('K').value
    T_use = data[('gas', 'temperature_two_regime')].to('K').value
    n_H = data[('gas', 'number_density_H')].to('cm**-3').value
    low = T_qk < T_QK_TWO_REGIME_K

    # ── Cold branch (DESPOTIC, as before) ──
    n_e_cold = data[('gas', 'e-')].to('cm**-3').value
    n_Hp_cold = data[('gas', 'H+')].to('cm**-3').value

    # ── T_QK >= 3000 K: total electrons from QUOKKA mean molecular weight ──
    e_int = data[('gas', 'internal_energy_density')].to('erg/cm**3').value
    rho = data[('gas', 'density')].to('g/cm**3').value
    x_e = electron_fraction_from_mean_molecular_weight(
        e_int,
        rho,
        T_use,
        hydrogen_mass_g=float(m_H.to('g').value),
        boltzmann_erg_K=float(kb.to('erg/K').value),
    )
    x_Hp = np.where(x_e <= 1.0, x_e, 1.0)
    n_e_hot = x_e * n_H
    n_Hp_hot = x_Hp * n_H

    n_e = np.where(low, n_e_cold, n_e_hot)
    n_Hp = np.where(low, n_Hp_cold, n_Hp_hot)

    # α_B(T)  — Draine 2011 Eq. 14.6 hydrogenic fit, Z = 1.
    T4 = np.maximum(T_use / 1.0e4, 1.0e-10)   # guard log(0)
    expnt   = -0.8163 - 0.0208 * np.log(T4)
    alpha_B = 2.54e-13 * np.power(T4, expnt)         # cm^3 / s

    # ε_Hα — erg / s / cm^3.
    eps = 0.45 * E_Halpha * alpha_B * n_e * n_Hp

    # Return as a YTArray with explicit units (matches the field declaration).
    return yt.YTArray(eps, "erg/s/cm**3")


def _HI_luminosity_two_regime(field, data):
    """HI 21 cm volumetric emissivity, optically-thin spin-flip:
        ε_21 = (3/4) · n_HI · A_10 · h · ν_21        [erg s^-1 cm^-3]
    Factor 3/4 is the upper hyperfine state fraction in the high-T limit
    (kT >> hν_21/k = 0.07 K, valid for all ISM).

    Two-regime treatment of n_HI:
      - Low (T_QK < 3000 K): n_HI from the 3D DESPOTIC LAMDA table
        (UV+CR-aware chemistry), field ('gas','HI').
      - T_QK ≥ 3000 K: infer x_e from QUOKKA's mean molecular weight.
        If x_e <= 1, n_HI=(1-x_e)n_H; otherwise n_HI=0.
    """
    T_qk    = data[('gas', 'temperature_quokka')].to('K').value
    T_use   = data[('gas', 'temperature_two_regime')].to('K').value
    n_H_sim = data[('gas', 'number_density_H')].to('cm**-3').value
    low = T_qk < T_QK_TWO_REGIME_K

    # ── Cold branch: 3D DESPOTIC table neutral-H species ─────────────
    # data[('gas', 'HI')] = x_HI(n_H,N_H,dVdr) · n_H_tot from the LAMDA
    # chemistry equilibrium (UV+CR-aware), unchanged from the legacy path.
    # (Renamed 'H' → 'HI' on 2026-06-20: 'H' was ambiguous with total H.
    #  Underlying LAMDA species token is still 'H' — only the yt-field
    #  name was changed.  See _make_number_density_field for the mapping.)
    n_HI_cold = data[('gas', 'HI')].to('cm**-3').value

    # ── T_QK >= 3000 K: total electrons from QUOKKA mean molecular weight ──
    e_int = data[('gas', 'internal_energy_density')].to('erg/cm**3').value
    rho = data[('gas', 'density')].to('g/cm**3').value
    x_e = electron_fraction_from_mean_molecular_weight(
        e_int,
        rho,
        T_use,
        hydrogen_mass_g=float(m_H.to('g').value),
        boltzmann_erg_K=float(kb.to('erg/K').value),
    )
    n_HI_hot = np.where(
        x_e <= 1.0,
        (1.0 - x_e) * n_H_sim,
        0.0,
    )

    n_HI = np.where(low, n_HI_cold, n_HI_hot)

    A_10_val = A_HI_21                       # 2.85e-15 s^-1 (Furlanetto+2006)
    nu_val   = NU_HI_21                      # 1.420405751768e9 Hz (NIST)
    h_val    = float(h.in_cgs().value)       # erg·s
    eps      = 0.75 * n_HI * A_10_val * h_val * nu_val   # erg s^-1 cm^-3
    return yt.YTArray(eps, 'erg/s/cm**3')


def _HI_emissivity_from_number_density(n_HI):
    """Optically thin 21-cm emissivity for a supplied neutral-H density."""
    return 0.75 * n_HI * A_HI_21 * float(h.in_cgs().value) * NU_HI_21


def _HI_luminosity_despotic(field, data):
    """HI 21-cm emissivity using DESPOTIC n_HI for every cell."""
    n_HI = data[('gas', 'HI')].to('cm**-3').value
    eps = _HI_emissivity_from_number_density(n_HI)
    return yt.YTArray(eps, 'erg/s/cm**3')


def _HI_luminosity_quokka(field, data):
    """HI 21-cm emissivity using QUOKKA's mu-derived x_e for every cell."""
    T_qk = data[('gas', 'temperature_quokka')].to('K').value
    n_H = data[('gas', 'number_density_H')].to('cm**-3').value
    e_int = data[('gas', 'internal_energy_density')].to('erg/cm**3').value
    rho = data[('gas', 'density')].to('g/cm**3').value
    x_e = electron_fraction_from_mean_molecular_weight(
        e_int,
        rho,
        T_qk,
        hydrogen_mass_g=float(m_H.to('g').value),
        boltzmann_erg_K=float(kb.to('erg/K').value),
    )
    n_HI = np.where(x_e <= 1.0, (1.0 - x_e) * n_H, 0.0)
    eps = _HI_emissivity_from_number_density(n_HI)
    return yt.YTArray(eps, 'erg/s/cm**3')


def _HI_luminosity(field, data):
    """Backward-compatible alias of the DESPOTIC/QUOKKA two-regime result."""
    return _HI_luminosity_two_regime(field, data)


def _HI_freq(field, data):
    shape = data[('gas', 'density')].shape
    return np.full(shape, NU_HI_21, dtype=float) * yt.units.Hz


def _H_alpha_freq(field, data):
    shape = data[('gas', 'density')].shape
    return np.full(shape, NU_H_ALPHA, dtype=float) * yt.units.Hz


def _H_atom_thermal_width(field, data):
    """Doppler thermal width for H-emitting lines (m_emitter = 1 amu).
    Used by both HI 21 cm and Hα spectral cubes.

    Uses `temperature_two_regime` (2026-06-18) so hot-gas σ_v isn't
    underestimated by DESPOTIC's ~5×10⁴ K saturation.  See
    `[[two-regime-per-species]]` memory.
    """
    T = data[('gas', 'temperature_two_regime')].to('K')
    sigma_v = np.sqrt((kb * T) / (1.00794 * amu)).to('cm/s')
    return sigma_v


def _HI_thermal_width_despotic(field, data):
    """HI thermal width using T_DESPOTIC for every cell."""
    T = data[('gas', 'temperature_despotic')].to('K')
    return np.sqrt((kb * T) / (1.00794 * amu)).to('cm/s')


def _HI_thermal_width_quokka(field, data):
    """HI thermal width using T_QUOKKA for every cell."""
    T = data[('gas', 'temperature_quokka')].to('K')
    return np.sqrt((kb * T) / (1.00794 * amu)).to('cm/s')


def _get_cii_upper_fraction_lookup() -> CiiUpperFractionLookup:
    """Lazily load the CHIANTI N_u(T,n_H) table.

    LTE-only runs therefore preserve the previous startup behaviour and do
    not require the additional lookup artifact.
    """
    global CII_UPPER_FRACTION_LOOKUP_CACHE
    if CII_UPPER_FRACTION_LOOKUP_CACHE is None:
        CII_UPPER_FRACTION_LOOKUP_CACHE = CiiUpperFractionLookup.from_npz(
            cfg.CII_CHIANTI_NU_TABLE_PATH,
            hydrogen_mass_fraction=cfg.X_H,
            helium_mass_fraction=cfg.Y_HE,
        )
    return CII_UPPER_FRACTION_LOOKUP_CACHE


def _Cplus_luminosity_model(field, data, *, high_model: str):
    """[C II] 158 μm emissivity with a selectable high-T excitation model.

    Cold (T_QUOKKA < 3000 K)
        Replicate the legacy `_make_luminosity_field('C+')` lookup —
        ε = n_H · lumPerH from the 3D DESPOTIC LAMDA table at
        (n_H, N_H, dVdr).  Unchanged.

    Intermediate (3000 K ≤ T_QUOKKA < 1.307e4 K) — provisional
    Saha+LTE retained because the Methodology marks this regime "to be
    determined":

        Step 1.  Infer x_e from QUOKKA's mean molecular weight, following
                 the shared intermediate-temperature hydrogen treatment;
                 n_e = x_e n_H.

        Step 2.  Two-stage C Saha for x_C+ — 3-state conservation (added
                 2026-06-18 to fix the high-T x_C+ → 1 saturation bug):
            S_C1 = 2·(2π m_e k_B T / h²)^(3/2) · (U_Cp/U_C0)  · exp(-I_C /kT)
            S_C2 = 2·(2π m_e k_B T / h²)^(3/2) · (U_Cpp/U_Cp) · exp(-I_C2/kT)
            r1   = S_C1 / n_e        # n_C+ / n_C0
            r2   = S_C2 / n_e        # n_C++ / n_C+
            x_Cp = r1 / (1 + r1 + r1·r2)        # C0 : C+ : C++ = 1 : r1 : r1·r2
            (U_C0, U_Cp, U_Cpp from CHIANTI full level lists, NOT single
            g_ground — see 2026-06-17 test, the difference is 3× in S_C1.
            Low-T r2 → 0 recovers the old 2-state form r1/(1+r1).)

        Step 3.  n_Cp = x_Cp · A_C · n_H,   A_C = 1.6e-4 (GOW xC default)

        Step 4.  Boltzmann level pop (LTE, NOT collisional-radiative)
            r   = (g_u/g_l) · exp(-T_star/T)          # T_star=91.21 K
            n_u = n_Cp · r / (1 + r)

        Step 5.  P = n_u · A_ul · h · ν                [erg s⁻¹ cm⁻³]

    High (T_QUOKKA ≥ 1.307e4 K)
        The electron density remains the QUOKKA mean-molecular-weight result
        from Step 1.  Enforce n_C = n_C+ + n_C++ and use only the second Saha
        equilibrium:

            r2   = S_C2 / n_e = n_C++ / n_C+
            x_Cp = n_C+ / n_C = 1 / (1 + r2) = n_e / (n_e + S_C2)

        The emitting-ion density is n_Cp = x_Cp A_C n_H.  ``high_model`` then
        selects either two-level LTE or the CHIANTI statistical-equilibrium
        upper-level fraction; it does not change this Saha ion fraction.

    Atomic constants A_ul, ν, T_star, g_u/l, I_C, U_Cp(T), U_C0(T) all
    come from CHIANTI 10.1 via fiasco (see `[[fiasco-chianti-installed]]`).
    Tables built at module import.  n_H = sim's `('gas','number_density_H')`
    = ρ·X_H/m_H = total H nuclei (matches the per-H definition of A_C).
    """
    # ───────────────────── inputs (cgs values) ──────────────────────
    T_qk    = data[('gas', 'temperature_quokka')].to('K').value
    n_H_sim = data[('gas', 'number_density_H')].to('cm**-3').value

    _, intermediate, high = _temperature_regime_masks(T_qk)

    # ─────────────── COLD branch: DESPOTIC LAMDA table ──────────────
    lookup   = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    colDen_H = data[('gas', 'column_density_H')].to('cm**-2').value
    dVdr     = data[('gas', 'dVdr_lvg')].in_cgs().value
    eps_cold = _table_emissivity(lookup, 'C+', n_H_sim, colDen_H, dVdr)  # erg/s/cm³

    # ───── Shared analytic-branch inputs (intermediate + high) ─────
    # Step 1: Methodology mean-molecular-weight inversion → total n_e.
    T = T_qk
    e_int = data[('gas', 'internal_energy_density')].to('erg/cm**3').value
    rho = data[('gas', 'density')].to('g/cm**3').value
    x_e = electron_fraction_from_mean_molecular_weight(
        e_int,
        rho,
        T_qk,
        hydrogen_mass_g=float(m_H.to('g').value),
        boltzmann_erg_K=float(kb.to('erg/K').value),
    )
    n_e = x_e * n_H_sim                                                   # cm⁻³

    # Step 2: two-stage C Saha + 3-state conservation (2026-06-18).
    #   C0 → C+:  S_C1 = 2·(K·T)^1.5 · (U_Cp/U_C0)  · exp(-I_C  /kT)
    #   C+ → C++: S_C2 = 2·(K·T)^1.5 · (U_Cpp/U_Cp) · exp(-I_C2 /kT)
    # With n_e fixed by the Methodology's total electron fraction inferred
    # from mean molecular weight (individual electron donors are unresolved),
    # the two Saha equilibria decouple algebraically:
    #   r1 = S_C1/n_e = n_C+ /n_C0
    #   r2 = S_C2/n_e = n_C++/n_C+
    #   C0 : C+ : C++ = 1 : r1 : r1·r2
    # so  x_Cp = r1 / (1 + r1 + r1·r2).  Low-T r2 → 0 recovers the old
    # 2-state x_Cp = r1/(1+r1) automatically.
    U_Cp  = np.interp(T, _CII_T_GRID, _CII_U_CP)
    U_C0  = np.interp(T, _CII_T_GRID, _CII_U_C0)
    U_Cpp = np.interp(T, _CII_T_GRID, _CII_U_CPP)

    saha_prefactor = 2.0 * np.power(K_SAHA_PREF * T, 1.5)
    S_C1 = saha_prefactor * (U_Cp / U_C0) * np.exp(-_CII_T_C_K / T)
    S_C2 = saha_prefactor * (U_Cpp / U_Cp) * np.exp(-_CII_T_C2_K / T)

    r1 = S_C1 / n_e
    r2 = S_C2 / n_e
    x_Cp = r1 / (1.0 + r1 + r1 * r2)                                      # 3-state conserv.

    # Step 3: n_C+
    n_Cp = x_Cp * A_C_TOTAL * n_H_sim                                     # cm⁻³

    # Step 4: Boltzmann LTE for upper-level population
    r   = (_CII_G_U / _CII_G_L) * np.exp(-_CII_T_STAR / T)
    n_u = n_Cp * r / (1.0 + r)

    # Step 5: ε = n_u · A_ul · h · ν                                       [erg/s/cm³]
    h_cgs  = float(h.in_cgs().value)
    eps_hot = n_u * _CII_A_UL * h_cgs * _CII_NU_HZ

    # ── High branch (T ≥ 1.307e4 K): all C is C+ or C++. ──────────
    # The C+ <-> C++ Saha relation gives n_C++/n_C+ = S_C2/n_e.  Combining
    # that with n_C=n_C+ + n_C++ gives x_C+=n_e/(n_e+S_C2).
    x_Cp_high = n_e / (n_e + S_C2)
    n_Cp_high = x_Cp_high * A_C_TOTAL * n_H_sim

    if high_model == 'lte':
        upper_fraction_high = r / (1.0 + r)
    elif high_model == 'chianti':
        # CHIANTI statistical-equilibrium level-2 population on a (T,n_H)
        # table.  Its excitation colliders n_e and n_p were built explicitly
        # from H+He CIE charge neutrality.  The carbon ion fraction above uses
        # the Methodology's mu-derived n_e; carbon's own electrons are omitted
        # from both electron budgets because A_C is small.
        upper_fraction_high = _get_cii_upper_fraction_lookup()(T, n_H_sim)
    else:
        raise ValueError(f'Unknown C+ high-temperature model: {high_model!r}')

    n_u_high = n_Cp_high * upper_fraction_high
    eps_high = n_u_high * _CII_A_UL * h_cgs * _CII_NU_HZ

    # The Methodology leaves intermediate-temperature C+ "to be determined";
    # preserve Saha+LTE there.  At T>=1.307e4 both comparison fields use the
    # same two-stage Saha fraction, but select either LTE or CHIANTI excitation.
    # Nested np.where (not np.select — astropy's np.select helper mishandles an
    # array `default` during yt field-dependency detection).
    eps = np.where(high, eps_high, np.where(intermediate, eps_hot, eps_cold))
    return yt.YTArray(eps, 'erg/s/cm**3')


def _Cplus_luminosity_lte(field, data):
    """C+ result: DESPOTIC / three-stage Saha+LTE / two-stage Saha+LTE."""
    return _Cplus_luminosity_model(field, data, high_model='lte')


def _Cplus_luminosity_chianti(field, data):
    """Comparison result with CHIANTI N_u(T,n_H) only in the high-T branch."""
    return _Cplus_luminosity_model(field, data, high_model='chianti')


def _Cplus_luminosity_selected(field, data):
    """Cheap selector retaining the public ('gas', 'C+_luminosity') name."""
    return data[('gas', f'C+_luminosity_{cfg.CPLUS_HIGH_MODEL}')]


def _make_line_frequency_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    species_record = lookup.species_record(species)
    
    # 修正：加入 .ravel()[0] 取出單一純量數值
    freq_value = species_record.line.freq.ravel()[0] * yt.units.Hz
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')

    def _field(field, data):
        shape = data[('gas', 'density')].shape
        return np.full(shape, freq_value.value, dtype=float) * freq_value.units

    _field.__name__ = f"_frequency_{yt_safe_name}"
    return yt_safe_name, _field



def _make_thermal_width_field(species: str):
    # Use thermal broadening from gas temperature and molecular weight.
    # sigma_v = sqrt(k_B * T / m_species)
    species_masses = {
        'CO': 28.01 * amu,
        'C+': 12.01 * amu,
        'HCO+': 29.02 * amu,
    }
    if species not in species_masses:
        raise ValueError(f"Unknown species for thermal width: {species}")
    mass = species_masses[species]
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')

    def _field(field, data):
        # Species-specific temperature policy: molecular CO follows the
        # DESPOTIC thermal solution; ionic C+ follows QUOKKA's native T.
        temperature_field = emitter_temperature_field_name(species)
        T = data[('gas', temperature_field)].to('K')
        sigma_v = np.sqrt((kb * T) / mass).to('cm/s')
        return sigma_v

    _field.__name__ = f"_thermal_width_{yt_safe_name}"
    return yt_safe_name, _field

def _Bulk_Doppler_factor_x(field, data):
    """
    計算 X 方向的多普勒因子：(1 - v_x / c)
    假設觀測者位於 x 負方向，氣體向 x 正方向運動為遠離（紅移）。
    """
    # 獲取 X 方向的速度場
    v_x = data[("gas", "velocity_x")].in_units("cm/s")
    
    # 使用檔案中已定義的光速 c (2.99792458e10 cm/s)
    # 注意：檔案中定義的 c 是帶單位的 YTQuantity
    c_speed = c.in_units("cm/s")
    
    # 計算因子 (無單位)
    factor = 1.0 - (v_x / c_speed)
    
    return factor

def _Bulk_Doppler_factor_y(field, data):
    """(1 - v_y / c) — observer along -y direction."""
    v_y = data[("gas", "velocity_y")].in_units("cm/s")
    c_speed = c.in_units("cm/s")
    return 1.0 - (v_y / c_speed)

def _Bulk_Doppler_factor_z(field, data):
    """(1 - v_z / c) — observer along -z direction."""
    v_z = data[("gas", "velocity_z")].in_units("cm/s")
    c_speed = c.in_units("cm/s")
    return 1.0 - (v_z / c_speed)

def add_all_fields(ds):
    """Adds all derived fields to the yt dataset."""
    ds.add_field(name=('gas', 'number_density_H'), function=_number_density_H, sampling_type="cell", units="cm**-3", force_override=True)
    ds.add_field(name=('gas', 'column_density_H'), function=_column_density_H, sampling_type="cell", units="cm**-2", force_override=True)
    ds.add_field(name=('gas', 'dVdr_lvg'), function=_dVdr_lvg, sampling_type="cell", units="1/s", force_override=True)

    # Save QUOKKA's native temperature from the raw boxlib field.
    # ('boxlib', 'temperature') is the value QUOKKA wrote to disk (units K,
    # but yt reads it as dimensionless — we attach K explicitly here).
    def _temperature_quokka(field, data):
        return data[('boxlib', 'temperature')] * K
    ds.add_field(name=('gas', 'temperature_quokka'), function=_temperature_quokka,
                 sampling_type="cell", units="K", force_override=True)

    # Internal energy field (still useful as a sim diagnostic; kept even
    # though the μγ-bisection temperature_gamma_mu was removed 2026-06-13).
    ds.add_field(name=('gas', 'internal_energy_density'), function=_internal_energy_density,
                 sampling_type="cell", units="erg/cm**3", force_override=True)

    ds.add_field(name=('gas', 'temperature_despotic'), function=_temperature_despotic, sampling_type="cell", units="K", force_override=True)

    # Two-regime temperature retained for H-line widths, phase classification,
    # and diagnostics. CO/C+ products use their species-specific fields.
    ds.add_field(name=('gas', 'temperature_two_regime'),
                 function=_temperature_two_regime,
                 sampling_type="cell", units="K", force_override=True)
    ds.add_field(
        name=('gas', 'Bulk_Doppler_factor_x'),
        function=_Bulk_Doppler_factor_x,
        sampling_type="cell",
        units="",
        force_override=True
    )
    ds.add_field(
        name=('gas', 'Bulk_Doppler_factor_y'),
        function=_Bulk_Doppler_factor_y,
        sampling_type="cell",
        units="",
        force_override=True
    )
    ds.add_field(
        name=('gas', 'Bulk_Doppler_factor_z'),
        function=_Bulk_Doppler_factor_z,
        sampling_type="cell",
        units="",
        force_override=True
    )
    # SPECIES = ['H+', 'H2', 'H3+', 'He+', 'OHx', 'CHx', 'CO', 'C',
    #           'C+', 'HCO+', 'O', 'M+', 'H', 'He', 'M', 'e-']
    # 'H' added 2026-05-10 — needed by the HI 21-cm fields.
    # 2026-06-20: yt-field name renamed 'H' → 'HI' to remove ambiguity with
    # total H ('gas','number_density_H').  Tuple form is (yt_field_name,
    # lamda_token) — lamda_token=None means the same name is used on both
    # sides.  Only HI differs: yt='HI', LAMDA's H.dat still calls it 'H'.
    SPECIES: list[tuple[str, str | None]] = [
        ('H+', None),
        ('CO', None),
        ('C',  None),
        ('C+', None),
        ('e-', None),
        ('HI', 'H'),
    ]
    # C+ luminosity is handled by the comparison fields below: both use the
    # LAMDA table for cold cells and differ only in high-T excitation.
    # Leave CO + HCO+ on the auto-generated _make_luminosity_field path.)
    EMITTERS = ['CO']                              # HCO+ dropped 2026-06-23 (no longer analysed)
    EMITTERS_FREQ_WIDTH = ['CO', 'C+']             # HCO+ dropped 2026-06-23; freq + thermal_width for CO + C+
    for sp_yt, sp_lamda in SPECIES:
        _, func = _make_number_density_field(species=sp_yt, lamda_token=sp_lamda)
        ds.add_field(
            name=('gas', sp_yt),
            function=func,
            sampling_type="cell",
            units="cm**-3",
            force_override=True
        )

    for em in EMITTERS:
        _, lum_func = _make_luminosity_field(species=em)
        ds.add_field(
            name=('gas', f'{em}_luminosity'),
            function=lum_func,
            sampling_type="cell",
            units="erg/s/cm**3",
            force_override=True
        )

    # Register only the selected comparison field.  yt executes dependency
    # detection during add_field(), so registering the CHIANTI field in an LTE
    # run would otherwise require/load its lookup unnecessarily.  Both names
    # remain in CACHED_FIELDS; switching models later reuses the corresponding
    # independently retained file.
    cplus_model_fields = {
        'lte': _Cplus_luminosity_lte,
        'chianti': _Cplus_luminosity_chianti,
    }
    selected_cplus_name = f'C+_luminosity_{cfg.CPLUS_HIGH_MODEL}'
    ds.add_field(
        name=('gas', selected_cplus_name),
        function=cplus_model_fields[cfg.CPLUS_HIGH_MODEL],
        sampling_type="cell",
        units="erg/s/cm**3",
        force_override=True,
    )

    # Existing tasks keep using this public name; the environment-controlled
    # selector points them at one of the independently cached model fields.
    ds.add_field(
        name=('gas', 'C+_luminosity'),
        function=_Cplus_luminosity_selected,
        sampling_type="cell",
        units="erg/s/cm**3",
        force_override=True,
    )

    # freq + thermal_width still auto-registered for all 3 emitters
    # (including C+, whose luminosity uses the multi-regime function above).
    for em in EMITTERS_FREQ_WIDTH:
        _, freq_func = _make_line_frequency_field(species=em)
        ds.add_field(
            name=('gas', f'{em}_freq'),
            function=freq_func,
            sampling_type="cell",
            units="Hz",
            force_override=True
        )
        _, width_func = _make_thermal_width_field(species=em)
        ds.add_field(
            name=('gas', f'{em}_thermal_width'),
            function=width_func,
            sampling_type="cell",
            units="cm/s",
            force_override=True
        )


    ds.add_field(name=('gas', 'Halpha_luminosity'),
                 function=_Halpha_luminosity, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)

    # H_alpha and HI 21 cm closed-form fields. H_alpha_luminosity is
    # an alias of Halpha_luminosity for naming uniformity with the
    # other emitter-style fields ({sp}_luminosity, {sp}_freq,
    # {sp}_thermal_width).
    ds.add_field(name=('gas', 'H_alpha_luminosity'),
                 function=_Halpha_luminosity, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)
    ds.add_field(name=('gas', 'H_alpha_freq'),
                 function=_H_alpha_freq, sampling_type="cell",
                 units="Hz", force_override=True)
    ds.add_field(name=('gas', 'H_alpha_thermal_width'),
                 function=_H_atom_thermal_width, sampling_type="cell",
                 units="cm/s", force_override=True)

    ds.add_field(name=('gas', 'HI_luminosity_despotic'),
                 function=_HI_luminosity_despotic, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)
    ds.add_field(name=('gas', 'HI_luminosity_quokka'),
                 function=_HI_luminosity_quokka, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)
    ds.add_field(name=('gas', 'HI_luminosity_two_regime'),
                 function=_HI_luminosity_two_regime, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)
    # Backward-compatible public name for the original two-regime H I result.
    ds.add_field(name=('gas', 'HI_luminosity'),
                 function=_HI_luminosity, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)
    ds.add_field(name=('gas', 'HI_freq'),
                 function=_HI_freq, sampling_type="cell",
                 units="Hz", force_override=True)
    ds.add_field(name=('gas', 'HI_thermal_width_despotic'),
                 function=_HI_thermal_width_despotic, sampling_type="cell",
                 units="cm/s", force_override=True)
    ds.add_field(name=('gas', 'HI_thermal_width_quokka'),
                 function=_HI_thermal_width_quokka, sampling_type="cell",
                 units="cm/s", force_override=True)
    ds.add_field(name=('gas', 'HI_thermal_width'),
                 function=_HI_thermal_width_quokka, sampling_type="cell",
                 units="cm/s", force_override=True)

    print("Added CO/C+/H_alpha plus two-regime, DESPOTIC, and QUOKKA HI fields.")
