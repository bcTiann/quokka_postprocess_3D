# physics_models.py

import yt
import numpy as np
from scipy.special import erf as scipy_erf
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg, amu, kpc
from ...analysis import along_sight_cumulation
from . import config as cfg
from ...tables import load_table, load_table_4d
from ...tables.lookup import TableLookup, TableLookup4D


# --- Fundamental Physical Constants ---
m_H = mh.in_cgs()
lambda_Halpha = 656.3e-7 * cm
h = planck_constant
speed_of_light_value_in_ms = 299792458
c = speed_of_light_value_in_ms * m / s
TABLE_LOOKUP_CACHE: TableLookup | None = None
TABLE_LOOKUP_4D_CACHE: TableLookup4D | None = None
TABLE_LOOKUP_SPECIES: tuple[str, ...] = ()

# Lower bound on |∇·v|/3 used as the LVG dVdr field. Cells with smaller
# divergence are pinned to this floor — the table's dVdr axis bottoms
# out at 1e-19, so 1e-18 keeps every cell strictly inside the
# interpolation range. ~0.04% of plt263168 cells fall below this floor
# (quiescent halo), measured 2026-05-09.
DVDR_FLOOR = 1e-18

# Hα and HI 21 cm constants (closed-form per-cell emissivity formulas;
# see plan note 2026-05-10 — these lines bypass DESPOTIC LAMDA because
# Hα is recombination-cascade and HI 21 cm is hyperfine, neither of
# which DESPOTIC's collisional-excitation framework can treat).
NU_H_ALPHA = 4.5681e14   # Hz (Balmer α, 656.281 nm)
NU_HI_21   = 1.4204e9    # Hz (21.106 cm hyperfine F=1→0)
A_HI_21    = 2.876e-15   # s^-1  Einstein A for 21 cm

def ensure_table_lookup(path: str | None) -> TableLookup:
    global TABLE_LOOKUP_CACHE
    if TABLE_LOOKUP_CACHE is None:
        table = load_table(path or cfg.DESPOTIC_TABLE_PATH)
        TABLE_LOOKUP_CACHE = TableLookup(table)
    return TABLE_LOOKUP_CACHE


def ensure_table_lookup_4d(path: str | None) -> TableLookup4D:
    """Lazily load the fixed-T 4D table (nH, N_H, dVdr, T) for the high-T branch."""
    global TABLE_LOOKUP_4D_CACHE
    if TABLE_LOOKUP_4D_CACHE is None:
        table = load_table_4d(path or cfg.DESPOTIC_TABLE_4D_PATH)
        TABLE_LOOKUP_4D_CACHE = TableLookup4D(table)
    return TABLE_LOOKUP_4D_CACHE


def _high_T_mask(data) -> np.ndarray:
    """Boolean mask of cells where the high-T (μγ / 4D-table) branch applies."""
    T_qk = data[('gas', 'temperature_quokka')].to('K').value
    return T_qk > cfg.T_QK_HIGH_K


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


def _temperature_gamma_mu(field, data):
    """Temperature inferred from QUOKKA internal energy via the μγ bisection.

    Solves  e_int/ρ = k_B·T / [(γ(T)−1)·μ(T)·m_H]  for T, with μ(T)/γ(T) from
    the fixed-T 4D DESPOTIC table.  Computed for all cells but only consumed
    where the high-T branch applies (temperature_quokka > T_QK_HIGH_K).
    """
    lookup4d = ensure_table_lookup_4d(cfg.DESPOTIC_TABLE_4D_PATH)

    n_H      = data[('gas', 'number_density_H')].in_cgs().value
    colDen_H = data[('gas', 'column_density_H')].in_cgs().value
    dVdr     = data[('gas', 'dVdr_lvg')].in_cgs().value
    e_int    = data[('gas', 'internal_energy_density')].to('erg/cm**3').value
    rho      = data[('gas', 'density')].to('g/cm**3').value

    e_specific = e_int / rho   # erg/g

    nH_min,  nH_max  = lookup4d.table.nH_values.min(),         lookup4d.table.nH_values.max()
    col_min, col_max = lookup4d.table.col_density_values.min(), lookup4d.table.col_density_values.max()
    dv_min,  dv_max  = lookup4d.table.dVdr_values.min(),       lookup4d.table.dVdr_values.max()
    n_H_safe = np.clip(n_H,      nH_min,  nH_max)
    col_safe = np.clip(colDen_H, col_min, col_max)
    dV_safe  = np.clip(dVdr,     dv_min,  dv_max)

    # Pass cell dVdr — μ/cv are now genuinely dVdr-dependent after the
    # 2026-05-29 builder rewrite (per-dVdr setChemEq instead of broadcast).
    T = lookup4d.temperature_gamma_mu(n_H_safe, col_safe, e_specific, dVdr_cgs=dV_safe)
    return T * K


def _temperature_despotic(field, data):
    """Gas temperature: DESPOTIC equilibrium tg_final, with an optional high-T
    override.

    Cold cells (temperature_quokka ≤ T_QK_HIGH_K, or HIGH_T_4D_BLEND off) use
    direct interpolation of the (nH, NH, dVdr) table's tg_final.  Hot cells use
    T_gamma_mu — the μγ-bisection temperature from QUOKKA's internal energy —
    because tg_final saturates (~5e4 K) and is meaningless there.
    """
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)

    n_H      = data[('gas', 'number_density_H')].in_cgs().value
    colDen_H = data[('gas', 'column_density_H')].in_cgs().value
    dVdr     = data[('gas', 'dVdr_lvg')].in_cgs().value

    nH_min,  nH_max  = lookup.table.nH_values.min(),       lookup.table.nH_values.max()
    col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
    dv_min,  dv_max  = lookup.table.dVdr_values.min(),     lookup.table.dVdr_values.max()

    n_H_safe = np.clip(n_H,      nH_min,  nH_max)
    col_safe = np.clip(colDen_H, col_min, col_max)
    dV_safe  = np.clip(dVdr,     dv_min,  dv_max)

    tg_final = lookup.temperature(n_H_safe, col_safe, dV_safe)

    if not cfg.HIGH_T_4D_BLEND:
        return tg_final * K

    hot = _high_T_mask(data)
    if not np.any(hot):
        return tg_final * K

    T_gm = data[('gas', 'temperature_gamma_mu')].to('K').value
    return np.where(hot, T_gm, tg_final) * K


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

    # Process channels in chunks to bound per-iteration transient memory.
    # The unchunked version allocates ~5 arrays of (n_channels, ny, nz)
    # per LOS cell — at down=1 (n_ch=300, ny=256, nz=2048) that is ~6 GB
    # transient per inner step.  CHUNK=150 halves that to ~3 GB per build
    # (well within budget for 2-6 parallel workers at down=1) while keeping
    # only 2 outer iters so the Python loop overhead stays trivial.
    CHUNK = 150

    for ch0 in range(0, n_channels, CHUNK):
        ch1 = min(ch0 + CHUNK, n_channels)
        nu_lo = freq_edges_hz[ch0:ch1][:, None, None]          # (chunk, 1, 1)
        nu_hi = freq_edges_hz[ch0 + 1:ch1 + 1][:, None, None]

        for i in range(nx):
            nu_gas  = shifted_freq_val[i, :, :][None, :, :]    # (1, ny, nz)
            lum_gas = lum_val[i, :, :][None, :, :]
            sigma_v = np.maximum(thermal_val[i, :, :], 1.0)[None, :, :]
            sigma_nu = nu_gas * (sigma_v / c_cms)

            sqrt2_sigma = np.sqrt(2.0) * sigma_nu
            x_lo = (nu_lo - nu_gas) / sqrt2_sigma              # (chunk, ny, nz)
            x_hi = (nu_hi - nu_gas) / sqrt2_sigma
            bin_frac = 0.5 * (scipy_erf(x_hi) - scipy_erf(x_lo))

            spec_cube[ch0:ch1] += lum_gas * bin_frac / delta_nu_bin

    return spec_cube
    
# def _number_density_electron(field, data):
#     n_H = data[('gas', 'number_density_H')]
#     colDen_H = data[('gas', 'column_density_H')]
#     lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
#     number_density_electron = lookup.number_densities('e-', nH_cgs=n_H.value, colDen_cgs=colDen_H.value)

#     return number_density_electron * cm**-3

def _make_luminosity_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    cutoff = cfg.T_CUTOFF.get(species, cfg.T_CUTOFF_DEFAULT)

    def _field(field, data):
        n_H      = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        dVdr     = data[('gas','dVdr_lvg')].in_cgs().value
        T        = data[('gas','temperature_despotic')].to('K').value  # blended; for T_CUTOFF gating

        nH_min,  nH_max  = lookup.table.nH_values.min(),         lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        dv_min,  dv_max  = lookup.table.dVdr_values.min(),       lookup.table.dVdr_values.max()

        n_H_safe = np.clip(n_H,      nH_min,  nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        dV_safe  = np.clip(dVdr,     dv_min,  dv_max)

        lumPerH = np.asarray(lookup.line_field(species, "lumPerH", n_H_safe, col_safe, dV_safe), dtype=float)

        # High-T branch: hot cells take lumPerH from the fixed-T 4D table at T_gamma_mu.
        if cfg.HIGH_T_4D_BLEND:
            hot = _high_T_mask(data)
            if np.any(hot):
                lookup4d = ensure_table_lookup_4d(cfg.DESPOTIC_TABLE_4D_PATH)
                T_gm = data[('gas', 'temperature_gamma_mu')].to('K').value
                lumPerH_4d = lookup4d.line_field(species, "lumPerH", n_H_safe, col_safe, dV_safe, T_gm)
                lumPerH = np.where(hot, np.nan_to_num(lumPerH_4d, nan=0.0), lumPerH)

        lumPerH[T > cutoff] = 0.0
        lumPerH = np.nan_to_num(lumPerH, nan=0.0)

        return (n_H_safe * lumPerH) * (erg / s / cm**3)

    _field.__name__ = f"_luminosity_{yt_safe_name}"
    return yt_safe_name, _field


def _make_number_density_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    token = species

    def _field(field, data):
        n_H      = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        dVdr     = data[('gas','dVdr_lvg')].in_cgs().value
        T        = data[('gas','temperature_despotic')].to('K').value  # blended; for high-T mask

        nH_min,  nH_max  = lookup.table.nH_values.min(),         lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        dv_min,  dv_max  = lookup.table.dVdr_values.min(),       lookup.table.dVdr_values.max()

        n_H_safe = np.clip(n_H,      nH_min,  nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        dV_safe  = np.clip(dVdr,     dv_min,  dv_max)

        val = np.nan_to_num(lookup.number_densities([token], n_H_safe, col_safe, dV_safe)[token], nan=0.0)
        # NOTE 2026-05-29: removed legacy `val[T > 1e5 K] = 0.0` blanket gating.
        # That line zeroed ALL species (including e-, H+) in hot ionized gas,
        # which killed Hα emission everywhere T > 1e5 K regardless of the
        # HIGH_T_4D_BLEND path.  Hot-gas behaviour now goes through the proper
        # 4D-table override below (or, with blend off, through the saturated
        # tg_final interpolation — accepting that limitation explicitly).

        # High-T branch: hot cells take the abundance from the fixed-T 4D table
        # at T_gamma_mu (real hot chemistry — H+/e- → ~nH, cold tracers → 0).
        if cfg.HIGH_T_4D_BLEND:
            hot = _high_T_mask(data)
            if np.any(hot):
                lookup4d = ensure_table_lookup_4d(cfg.DESPOTIC_TABLE_4D_PATH)
                T_gm = data[('gas', 'temperature_gamma_mu')].to('K').value
                val_4d = lookup4d.number_densities([token], n_H_safe, col_safe, dV_safe, T_gm)[token]
                val = np.where(hot, np.nan_to_num(val_4d, nan=0.0), val)

        return val * cm**-3

    _field.__name__ = f"_number_density_{yt_safe_name}"
    return yt_safe_name, _field

def _Halpha_luminosity(field, data):
    """
    Calculate H-alpha Luminosity Density
    Units: erg / s / cm**3
    
    Draine (2011) Eq. 14.6
    """
    E_Halpha = (h * c) / lambda_Halpha # Energy of a single H-alpha photon
    density_3d = data[('gas', 'density')].in_cgs()
    temp = data[('gas', 'temperature_despotic')].in_cgs()
   
    n_e = data[('gas', 'e-')]
    n_ion = data[('gas', 'H+')]
    Z = 1.0
    T4 = temp / (1e4 * yt.units.K)

    exponent = -0.8163 - 0.0208 * np.log(T4 / Z**2)

    alpha_B = (2.54e-13 * Z**2 * (T4 / Z**2)**exponent) * cm**3 / s

    luminosity_density = 0.45 * E_Halpha * alpha_B * n_e * n_ion
    return luminosity_density.in_cgs()


def _HI_luminosity(field, data):
    """HI 21 cm volumetric emissivity, optically-thin spin-flip:
        ε_21 = (3/4) · n_HI · A_10 · h · ν_21        [erg s^-1 cm^-3]
    Factor 3/4 is the upper hyperfine state fraction in the high-T
    limit (kT >> hν_21/k = 0.07 K, valid for all ISM)."""
    n_HI = data[('gas', 'H')]                          # cm^-3
    A_10 = A_HI_21 / yt.units.s                        # s^-1
    nu   = NU_HI_21 * yt.units.Hz                      # 1/s
    return (0.75 * n_HI * A_10 * h * nu).in_cgs()


def _HI_freq(field, data):
    shape = data[('gas', 'density')].shape
    return np.full(shape, NU_HI_21, dtype=float) * yt.units.Hz


def _H_alpha_freq(field, data):
    shape = data[('gas', 'density')].shape
    return np.full(shape, NU_H_ALPHA, dtype=float) * yt.units.Hz


def _H_atom_thermal_width(field, data):
    """Doppler thermal width for H-emitting lines (m_emitter = 1 amu).
    Used by both HI 21 cm and Hα spectral cubes."""
    T = data[('gas', 'temperature_despotic')].to('K')
    sigma_v = np.sqrt((kb * T) / (1.00794 * amu)).to('cm/s')
    return sigma_v


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
        T = data[('gas', 'temperature_despotic')].to('K')
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

    # Internal energy + μγ-bisection temperature for the high-T branch.
    ds.add_field(name=('gas', 'internal_energy_density'), function=_internal_energy_density,
                 sampling_type="cell", units="erg/cm**3", force_override=True)
    ds.add_field(name=('gas', 'temperature_gamma_mu'), function=_temperature_gamma_mu,
                 sampling_type="cell", units="K", force_override=True)

    ds.add_field(name=('gas', 'temperature_despotic'), function=_temperature_despotic, sampling_type="cell", units="K", force_override=True)
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
    # SPECIES = ['H+', 'H2', 'H3+', 'He+', 'OHx', 'CHx', 'CO', 'C',
    #           'C+', 'HCO+', 'O', 'M+', 'H', 'He', 'M', 'e-']
    # 'H' added 2026-05-10 — needed by _HI_luminosity (HI 21 cm).
    SPECIES = ['H+', 'CO', 'C', 'C+', 'e-', 'H']
    EMITTERS = ['CO', 'C+', 'HCO+']
    for sp in SPECIES:
        _, func = _make_number_density_field(species=sp)
        ds.add_field(
            name=('gas', f'{sp}'),
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

    ds.add_field(name=('gas', 'HI_luminosity'),
                 function=_HI_luminosity, sampling_type="cell",
                 units="erg/s/cm**3", force_override=True)
    ds.add_field(name=('gas', 'HI_freq'),
                 function=_HI_freq, sampling_type="cell",
                 units="Hz", force_override=True)
    ds.add_field(name=('gas', 'HI_thermal_width'),
                 function=_H_atom_thermal_width, sampling_type="cell",
                 units="cm/s", force_override=True)

    print("Added derived fields including HI/H_alpha (closed-form, no DESPOTIC).")
