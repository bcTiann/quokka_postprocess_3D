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



def ensure_table_lookup(table_path: str) -> TableLookup:
    """Ensure a TableLookup instance is cached and return it."""
    global TABLE_LOOKUP_CACHE
    if TABLE_LOOKUP_CACHE is None:
        table = load_table(table_path)
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


# def _temperature(field, data):
#     n_H = data[('gas', 'number_density_H')].to('cm**-3').value
#     colDen_H = data[('gas', 'column_density_H')].to('cm**-2').value
#     # for each point (nH, colDen) we get lookup our despotic table, get tempearture for that (nH, colDen)
#     lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
#     # print("============================") 
#     # print(f"table nH_min:{nH_min:.3e}")
#     # print(f"table nH_max:{nH_max:.3e}")
#     # print(f"table col_min:{col_min:.3e}")
#     # print(f"table col_max:{col_max:.3e}")
#     # print("============================")
#     # print(f"data n_H min: {n_H.min():.3e}")
#     # print(f"data n_H max: {n_H.max():.3e}")
#     # print(f"data col min: {colDen_H.min():.3e}")
#     # print(f"data col max: {colDen_H.max():.3e}")
#     # print("============================")
#     nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
#     col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
#     n_H_safe = np.clip(n_H, nH_min, nH_max)
#     col_safe = np.clip(colDen_H, col_min, col_max)

#     temps = lookup.temperature(nH_cgs=n_H_safe, colDen_cgs=col_safe, T_K=np.full_like(n_H_safe, 100.0))  # 暂时使用默认温度
    
#     return temps * K

def _temperature(field, data):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)

    n_H = data[('gas', 'number_density_H')]
    colDen_H = data[('gas', 'column_density_H')]
    # 1. 计算目标能量 (Target Energy) E_target_K = u / (n_H * k_B)
    E_total = data[('gas', 'total_energy_density')].to('erg/cm**3')
    E_kinetic = data[('gas', 'kinetic_energy_density')].to('erg/cm**3')
    E_int = E_total - E_kinetic
    E_target_K = (E_int / (n_H * kb)).in_cgs().value
    
    # 2. 限制输入范围，防止插值器报错
    nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
    col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
    T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()
    
    n_H_safe = np.clip(n_H.in_cgs().value, nH_min, nH_max)
    col_safe = np.clip(colDen_H.in_cgs().value, col_min, col_max)
    T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()

    n_H_safe = np.clip(n_H.in_cgs().value, nH_min, nH_max)
    col_safe = np.clip(colDen_H.in_cgs().value, col_min, col_max)

    # 3. 向量化二分法求 T (Vectorized Bisection)
    # 为所有的网格点初始化温度的上下界
    T_low = np.full_like(E_target_K, T_min)
    T_high = np.full_like(E_target_K, T_max)

    for _ in range(25):
        T_mid = (T_low + T_high) * 0.5
        # 查表获取 T_mid 对应的无因次内能，并计算当前测试的物理能量
        E_mid = T_mid * lookup.Eint(n_H_safe, col_safe, T_mid)

        # 如果测试能量小于目标能量，说明真解在右半区间，提高下界；否则降低上界
        mask = E_mid < E_target_K
        T_low[mask] = T_mid[mask]
        T_high[~mask] = T_mid[~mask]

    T_final = (T_low + T_high) * 0.5
    return T_final * K




    
# def _number_density_electron(field, data):
#     n_H = data[('gas', 'number_density_H')]
#     colDen_H = data[('gas', 'column_density_H')]
#     lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
#     number_density_electron = lookup.number_densities('e-', nH_cgs=n_H.value, colDen_cgs=colDen_H.value)

#     return number_density_electron * cm**-3

def _make_luminosity_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    
    def _field(field, data):
        # 1. 提取基础变量，直接向 yt 索要 temperature！
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        T = data[('gas', 'temperature')].to('K').value
        
        # 2. 限制安全边界
        nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()

        n_H_safe = np.clip(n_H, nH_min, nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        T_safe = np.clip(T, T_min, T_max)

        # 3. 传入 T_safe 查表
        lumPerH = lookup.line_field(species, "lumPerH", n_H_safe, col_safe, T_safe)
        
        # 返回最终光度密度
        return (n_H_safe * lumPerH) * (erg / s / cm**3)
        
    _field.__name__ = f"_luminosity_{yt_safe_name}"
    return yt_safe_name, _field


def _make_number_density_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    
    def _field(field, data):
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        T = data[('gas', 'temperature')].to('K').value # 直接读取温度
        
        nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()

        n_H_safe = np.clip(n_H, nH_min, nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        T_safe = np.clip(T, T_min, T_max)

        # 记得传入 T_safe
        densities = lookup.number_densities([species], n_H_safe, col_safe, T_safe)
        
        return densities[species] * cm**-3
        
    _field.__name__ = f"_number_density_{yt_safe_name}"
    return yt_safe_name, _field


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
        'CO': 28.0 * mp,
        'C+': 12.0 * mp,
        'HCO+': 29.0 * mp,
    }
    if species not in species_masses:
        raise ValueError(f"Unknown species for thermal width: {species}")
    mass = species_masses[species]
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')

    def _field(field, data):
        T = data[('gas', 'temperature')].to('K')
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

def get_one_sightline_spectrum(
    freq_los,   # [Hz] 視線上各點的觀測頻率 (已含多普勒位移) [Nx]
    lum_los,    # [erg/s] 視線上各點的總光度 [Nx]
    width_los,  # [cm/s] 視線上各點的譜線寬度 (sigma_v) [Nx]
    freq_axis,  # [Hz] 目標頻率網格的中心點 [n_channels]
    ):
    # 1. 定義常數 (確保與 width_los 單位一致，皆為 cm/s)
    c_cm_s = c.in_units("cm/s")

    # 2. 準備廣播維度
    nu_grid = freq_axis[:, None] # [n_channels, 1]
    nu_gas = freq_los[None, :]   # [1, Nx]
    lum_gas = lum_los[None, :]   # [1, Nx]
    
    # 3. 防止 sigma_v 為零導致崩潰
    width_los_cms = width_los.to('cm/s').value
    sigma_v = np.maximum(width_los_cms, 1.0) * width_los.units
    
    # 4. 計算頻率域的寬度: sigma_nu = nu * (sigma_v / c)
    sigma_nu = nu_gas * (sigma_v[None, :] / c_cm_s) # [1, Nx]

    # 5. 高斯歸一化係數: L / (sqrt(2*pi) * sigma_nu)
    norm = lum_gas / (np.sqrt(2 * np.pi) * sigma_nu)

    # 6. 計算指數部分: exp(-0.5 * (delta_nu / sigma_nu)^2)
    delta_nu = nu_grid - nu_gas
    exponent = -0.5 * (delta_nu / sigma_nu)**2
    
    # 7. 算出每個 cell 在每個 channel 的貢獻 [n_channels, Nx]
    profile = norm * np.exp(exponent)

    # 8. 對視線方向求和，得到該像素的光譜 [n_channels]
    spectrum_yz = np.sum(profile, axis=1)

    return spectrum_yz

def _Halpha_luminosity(field, data):
    """
    Calculate H-alpha Luminosity Density
    Units: erg / s / cm**3
    
    Draine (2011) Eq. 14.6
    """
    E_Halpha = (h * c) / lambda_Halpha # Energy of a single H-alpha photon
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
    ds.add_field(
        name=('gas', 'Bulk_Doppler_factor_x'),
        function=_Bulk_Doppler_factor_x,
        sampling_type="cell",
        units="",  # 無單位
        force_override=True
    )
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
        
    ds.add_field(name=('gas', 'Halpha_luminosity'), function=_Halpha_luminosity, sampling_type="cell", units="erg/s/cm**3", force_override=True)
    print("Added derived fields: 'temp_neutral', 'temperature', 'ionized_mask', 'Halpha_luminosity'.")
