import yt
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg
from ..prep.physics_fields import add_all_fields
from ..prep import config as cfg
from ...data_handling import YTDataProvider
from ...despotic_tables import compute_average
from ...analysis import along_sight_cumulation
from ...tables.lookup import TableLookup
from ...tables import load_table
import numpy as np
from matplotlib.colors import LogNorm
TABLE_LOOKUP_CACHE: TableLookup | None = None

def ensure_table_lookup(path: str | None) -> TableLookup:
    global TABLE_LOOKUP_CACHE
    if TABLE_LOOKUP_CACHE is None:
        table = load_table(path or cfg.DESPOTIC_TABLE_PATH)
        TABLE_LOOKUP_CACHE = TableLookup(table)
    return TABLE_LOOKUP_CACHE

def _Avg_column_density_H(n_H, dx, dy, dz):
    Nx_p = along_sight_cumulation(n_H * dx, axis="x", sign="+")
    Ny_p = along_sight_cumulation(n_H * dy, axis="y", sign="+")
    Nz_p = along_sight_cumulation(n_H * dz, axis="z", sign="+")
    Nx_n = along_sight_cumulation(n_H * dx, axis="x", sign="-")
    Ny_n = along_sight_cumulation(n_H * dy, axis="y", sign="-")
    Nz_n = along_sight_cumulation(n_H * dz, axis="z", sign="-")

    average_N_3d = compute_average(
        [Nx_p, Ny_p, Nz_p, Nx_n, Ny_n, Nz_n],
        method="harmonic",
    )
    return average_N_3d.to('cm**-2')


def _temperature(n_H, ColDen_H):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    temperature = lookup.temperature(nH_cgs=n_H, colDen_cgs=ColDen_H)
    return temperature * K

def _halpha_luminosity(n_H, column_density, temperature, n_e, n_ion):

    lambda_Halpha = 656.3e-7 * cm
    h = planck_constant
    speed_of_light_value_in_ms = 299792458 
    c = speed_of_light_value_in_ms * m / s

    E_Halpha = (h * c) / lambda_Halpha
    Z = 1.0
    T4 = temperature / (1e4 * yt.units.K)

    exponent = -0.8163 - 0.0208 * np.log(T4 / Z**2)

    alpha_B = (2.54e-13 * Z**2 * (T4 / Z**2)**exponent) * cm**3 / s

    luminosity_density = 0.45 * E_Halpha * alpha_B * n_e * n_ion
    print(f"lum density units:{luminosity_density.units}")
    luminosity_density = luminosity_density.in_cgs()
    print(f"lum density units in cgs:{luminosity_density.units}")
    return luminosity_density

class SpeciesNumberDensityProvider:
    def __init__(self, n_H, column_density, table_path=None):
        """
        Parameters
        ----------
        n_H : unyt_array or ndarray
            3D hydrogen number-density map (cm^-3).
        column_density : unyt_array or ndarray
            3D hydrogen column-density map (cm^-2).
        table_path : str, optional
            Path to a DESPOTIC lookup table (defaults to cfg.DESPOTIC_TABLE_PATH).
        """
        self.lookup = ensure_table_lookup(table_path)
        self.n_H = n_H.in_cgs().value
        self.column_density = column_density.in_cgs().value
        self._species = {'H+', 'CO', 'C', 'C+', 'HCO+', 'e-'}

    def number_density(self, species):
        """Return number density field for a requested species."""
        if species not in self._species:
            raise ValueError(f"Sepcies '{species} not in {self._species}")
        densities = self.lookup.number_densities([species], self.n_H, self.column_density)
        return densities[species] * (cm**-3)

ds = yt.load(cfg.YT_DATASET_PATH)
add_all_fields(ds)

provider = YTDataProvider(ds)

n_H = provider.get_grid_data(('gas', 'number_density_H'))
dx = provider.get_grid_data(('boxlib', 'dx'))
dy = provider.get_grid_data(('boxlib', 'dy'))
dz = provider.get_grid_data(('boxlib', 'dz'))
extent = provider.get_plot_extent(axis='x')

ColDen_H = _Avg_column_density_H(n_H, dx, dy, dz)
print(ColDen_H.units)
species_provider = SpeciesNumberDensityProvider(n_H=n_H, column_density=ColDen_H, table_path=cfg.DESPOTIC_TABLE_PATH)
electron_number_density = species_provider.number_density("e-")
Hion_number_density = species_provider.number_density("H+")

ColDen_H_cgs = ColDen_H.in_cgs().value
n_H_cgs = n_H.in_cgs().value

temperature = _temperature(n_H=n_H_cgs, ColDen_H=ColDen_H_cgs, )
Halpha_lum = _halpha_luminosity(n_H=n_H, column_density=ColDen_H, temperature=temperature, n_e=electron_number_density, n_ion=Hion_number_density)

# q2s.create_plot(
#     electron_number_density[64, :, :].in_cgs().to_ndarray().T,
#     title="electron_number_density",
#     extent=extent,
#     xlabel=dx.units,
#     ylabel=dy.units,
#     filename="electron_number_density from DESPOTIC",
#     cbar_label="electron_number_density cm-3",
#     norm=LogNorm(),
# )

q2s.create_plot(
    temperature[64, :, :].in_cgs().to_ndarray().T,
    title="Temperautre",
    extent=extent,
    xlabel=dx.units,
    ylabel=dy.units,
    filename="Temperature from DESPOTIC",
    cbar_label="Temperature (K)",
    norm=LogNorm(),
)

q2s.create_plot(
    Halpha_lum.in_cgs().to_ndarray().sum(axis=0).T,
    title="Halpha no dust",
    extent=extent,
    xlabel=dx.units,
    ylabel=dy.units,
    filename="Hapha no dust from DESPOTIC",
    cbar_label="H-alpha lumniosity",
    norm=LogNorm(),
)


