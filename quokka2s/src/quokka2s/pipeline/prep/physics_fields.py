# physics_models.py

import yt
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
import quokka2s as q2s
from quokka2s.despotic_tables import compute_average
from . import config as cfg
from quokka2s.tables import load_table
from quokka2s.tables.lookup import TableLookup


# --- Fundamental Physical Constants ---
m_H = mh.in_cgs()
lambda_Halpha = 656.3e-7 * cm
h = planck_constant
speed_of_light_value_in_ms = 299792458 
c = speed_of_light_value_in_ms * m / s
TABLE_LOOKUP_CACHE: TableLookup | None = None
TABLE_LOOKUP_SPECIES: tuple[str, ...] = ()

def ensure_table_lookup(path: str | None) -> TableLookup:
    global TABLE_LOOKUP_CACHE
    if TABLE_LOOKUP_CACHE is None:
        table = load_table(path or cfg.DESPOTIC_TABLE_PATH)
        TABLE_LOOKUP_CACHE = TableLookup(table)
    return TABLE_LOOKUP_CACHE


def _number_density_H(field, data):
    density_3d = data[('gas', 'density')].in_cgs()
    n_H_3d = (density_3d * cfg.X_H) / m_H
    return n_H_3d.to('cm**-3')



def _column_density_H(field, data):
    density_3d = data[('gas', 'density')].in_cgs()
    dx_3d = data[("boxlib", "dx")].in_cgs()
    dy_3d = data[("boxlib", "dy")].in_cgs()
    dz_3d = data[("boxlib", "dz")].in_cgs()

    n_H_3d = (density_3d * cfg.X_H) / m_H

    Nx_p = q2s.along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="+")
    Ny_p = q2s.along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="+")
    Nz_p = q2s.along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="+")
    Nx_n = q2s.along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="-")
    Ny_n = q2s.along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="-")
    Nz_n = q2s.along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="-")

    average_N_3d = compute_average(
        [Nx_p, Ny_p, Nz_p, Nx_n, Ny_n, Nz_n],
        method="harmonic",
    )
    return average_N_3d.to('cm**-2')

# --- YT Derived Fields ---


def _temperature(field, data):
    n_H = data[('gas', 'number_density_H')].to('cm**-3').value
    colDen_H = data[('gas', 'column_density_H')].to('cm**-2').value
    # for each point (nH, colDen) we get lookup our despotic table, get tempearture for that (nH, colDen)
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    # print("============================") 
    # print(f"table nH_min:{nH_min:.3e}")
    # print(f"table nH_max:{nH_max:.3e}")
    # print(f"table col_min:{col_min:.3e}")
    # print(f"table col_max:{col_max:.3e}")
    # print("============================")
    # print(f"data n_H min: {n_H.min():.3e}")
    # print(f"data n_H max: {n_H.max():.3e}")
    # print(f"data col min: {colDen_H.min():.3e}")
    # print(f"data col max: {colDen_H.max():.3e}")
    # print("============================")
    nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
    col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
    n_H_safe = np.clip(n_H, nH_min, nH_max)
    col_safe = np.clip(colDen_H, col_min, col_max)

    temps = lookup.temperature(nH_cgs=n_H_safe, colDen_cgs=col_safe)
    
    return temps * K


    
# def _number_density_electron(field, data):
#     n_H = data[('gas', 'number_density_H')]
#     colDen_H = data[('gas', 'column_density_H')]
#     lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
#     number_density_electron = lookup.number_densities('e-', nH_cgs=n_H.value, colDen_cgs=colDen_H.value)

#     return number_density_electron * cm**-3

def _make_luminosity_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    token = species
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    def _field(field, data):
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        n_H_safe = np.clip(n_H, nH_min, nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        lumPerH = lookup.line_field(species, "lumPerH", n_H_safe, col_safe)
        return (n_H_safe * lumPerH) * (erg / s / cm**3)
    _field.__name__ = f"_luminosity_{yt_safe_name}"
    return yt_safe_name, _field

def _make_number_density_field(species: str):
    print("============================") 
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    print("============================") 
    print(lookup.table.species_data.keys())
    token = species
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    def _field(field, data):
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        n_H_safe = np.clip(n_H, nH_min, nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        densities = lookup.number_densities([token], n_H_safe, col_safe)
        return densities[token] * cm**-3
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
    temp = data[('gas', 'temperature')].in_cgs()
   
    n_e = data[('gas', 'e-')]
    n_ion = data[('gas', 'H+')]
    # n_H = (density_3d * cfg.X_H) / m_H
    print("n_e finite?", np.isfinite(n_e).any(), "min/max", n_e.min(), n_e.max())
    print("n_ion finite?", np.isfinite(n_ion).any(), "min/max", n_ion.min(), n_ion.max())   
    Z = 1.0
    T4 = temp / (1e4 * yt.units.K)

    exponent = -0.8163 - 0.0208 * np.log(T4 / Z**2)

    alpha_B = (2.54e-13 * Z**2 * (T4 / Z**2)**exponent) * cm**3 / s

    luminosity_density = 0.45 * E_Halpha * alpha_B * n_e * n_ion
    print(f"lum density units:{luminosity_density.units}")
    luminosity_density = luminosity_density.in_cgs()
    print(f"lum density units in cgs:{luminosity_density.units}")
    return luminosity_density


def add_all_fields(ds):
    """Adds all derived fields to the yt dataset."""
    ds.add_field(name=('gas', 'number_density_H'), function=_number_density_H, sampling_type="cell", units="cm**-3", force_override=True)
    ds.add_field(name=('gas', 'column_density_H'), function=_column_density_H, sampling_type="cell", units="cm**-2", force_override=True)
    ds.add_field(name=('gas', 'temperature'), function=_temperature, sampling_type="cell", units="K", force_override=True)
    
    # SPECIES = ['H+', 'H2', 'H3+', 'He+', 'OHx', 'CHx', 'CO', 'C', 
    #           'C+', 'HCO+', 'O', 'M+', 'H', 'He', 'M', 'e-']
    SPECIES = ['H+', 'CO', 'C', 'C+', 'e-']
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
        
    ds.add_field(name=('gas', 'Halpha_luminosity'), function=_Halpha_luminosity, sampling_type="cell", units="erg/s/cm**3", force_override=True)
    print("Added derived fields: 'temp_neutral', 'temperature', 'ionized_mask', 'Halpha_luminosity'.")
