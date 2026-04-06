# physics_models.py

import yt
import numpy as np
from yt.units import K, mp, kb, mh, planck_constant, cm, m, s, g, erg, amu
from ...analysis import along_sight_cumulation
from ...despotic_tables import compute_average
from . import config as cfg
from ...tables import load_table
from ...tables.lookup import TableLookup


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

    Nx_p = along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="+")
    Ny_p = along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="+")
    Nz_p = along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="+")
    Nx_n = along_sight_cumulation(n_H_3d * dx_3d, axis="x", sign="-")
    Ny_n = along_sight_cumulation(n_H_3d * dy_3d, axis="y", sign="-")
    Nz_n = along_sight_cumulation(n_H_3d * dz_3d, axis="z", sign="-")

    average_N_3d = compute_average(
        [Nx_p, Ny_p, Nz_p, Nx_n, Ny_n, Nz_n],
        method="harmonic",
    )
    return average_N_3d.to('cm**-2')

# --- YT Derived Fields ---


def _temperature_error(field, data):
    # 利用刚刚写好的 _temperature，它会被缓存，不会重复计算
    T = data[('gas', 'temperature')].to('K').value
    n_H = data[('gas', 'number_density_H')].to('cm**-3').value
    colDen = data[('gas', 'column_density_H')].to('cm**-2').value
    
    E_total = data[('gas', 'total_energy_density')].to('erg/cm**3')
    E_kinetic = data[('gas', 'kinetic_energy_density')].to('erg/cm**3')
    E_int = E_total - E_kinetic
    E_target_K = (E_int / (data[('gas', 'number_density_H')] * kb)).in_cgs().value
    
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    nH_safe = np.clip(n_H, lookup.table.nH_values.min(), lookup.table.nH_values.max())
    col_safe = np.clip(colDen, lookup.table.col_density_values.min(), lookup.table.col_density_values.max())
    
    E_final = T * lookup.Eint(nH_safe, col_safe, T)
    rel_error = np.abs(E_final - E_target_K) / (np.abs(E_target_K) + 1e-30)
    
    return rel_error * yt.units.dimensionless



def _temperature(field, data):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)

    n_H = data[('gas', 'number_density_H')]
    colDen_H = data[('gas', 'column_density_H')]

    # 1. 计算目标能量
    E_total = data[('gas', 'total_energy_density')].to('erg/cm**3')
    E_kinetic = data[('gas', 'kinetic_energy_density')].to('erg/cm**3')
    E_int = E_total - E_kinetic
    E_target_K = (E_int / (n_H * kb)).in_cgs().value
    E_target_K = np.maximum(E_target_K, 1e-30)

    # 2. 限制输入范围
    nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
    col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
    T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()

    n_H_safe = np.clip(n_H.in_cgs().value, nH_min, nH_max)
    col_safe = np.clip(colDen_H.in_cgs().value, col_min, col_max)

    # 3. 向量化二分法求 T
    T_low = np.full_like(E_target_K, T_min)
    T_high = np.full_like(E_target_K, T_max)

    for _ in range(25):
        T_mid = (T_low + T_high) * 0.5
        E_mid = T_mid * lookup.Eint(n_H_safe, col_safe, T_mid)
        mask = E_mid < E_target_K
        T_low[mask] = T_mid[mask]
        T_high[~mask] = T_mid[~mask]

    T_final = (T_low + T_high) * 0.5

    # ==========================================
    # 4. 收敛性检验 (Convergence Diagnostics)
    # ==========================================
    # 重新计算一次最终温度对应的能量
    E_final = T_final * lookup.Eint(n_H_safe, col_safe, T_final)

    # 计算相对误差 (Relative Error)，加一个小量 1e-30 防止除以零
    rel_error = np.abs(E_final - E_target_K) / (np.abs(E_target_K) + 1e-30)

    # 定义“不收敛”的阈值：相对误差大于 5% (0.05) 即视为找不到解
    tolerance = 0.05
    bad_mask = rel_error > tolerance
    n_bad = np.sum(bad_mask)
    n_total = bad_mask.size

    if n_bad > 0:
        print(f"\n[WARNING] Temperature Bisection Failed for {n_bad} / {n_total} points ({(n_bad/n_total)*100:.2f}%).")

        # 提取坏点的信息
        bad_nH = n_H_safe[bad_mask]
        bad_E_target = E_target_K[bad_mask]
        bad_T_final = T_final[bad_mask]
        bad_error = rel_error[bad_mask]

        # 判断这些坏点是不是撞到了表格的温度边界
        hit_min = np.sum(bad_T_final <= T_min + 0.1)
        hit_max = np.sum(bad_T_final >= T_max - 0.1)
        print(f"          -> {hit_min} points hit the T_min ({T_min} K) boundary.")
        print(f"          -> {hit_max} points hit the T_max ({T_max} K) boundary.")

        # 打印前 5 个最严重的坏点，供你物理检查
        print("          Sample Bad Points (Top 5):")
        for i in range(min(5, n_bad)):
            print(f"          {i+1}. nH = {bad_nH[i]:.2e} cm^-3, Target Energy(K) = {bad_E_target[i]:.2e}, "
                  f"Stuck at T = {bad_T_final[i]:.2f} K, Error = {bad_error[i]:.1%}")
        print("-" * 50)

    return T_final * K

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
        # 1. 提取基础变量，增加提取 temperature
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        T = data[('gas', 'temperature')].to('K').value  # <--- 新增
        
        # 2. 限制安全边界 (增加 T 的边界)
        nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()

        n_H_safe = np.clip(n_H, nH_min, nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        T_safe = np.clip(T, T_min, T_max) # <--- 新增

        # 3. 传入 T_safe 查表！(必须要有 4 个参数)
        lumPerH = lookup.line_field(species, "lumPerH", n_H_safe, col_safe, T_safe)

        lumPerH[T > 100000.0] = 0.0
        lumPerH = np.nan_to_num(lumPerH, nan=0.0)

        return (n_H_safe * lumPerH) * (erg / s / cm**3)
        
    _field.__name__ = f"_luminosity_{yt_safe_name}"
    return yt_safe_name, _field


def _make_number_density_field(species: str):
    lookup = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
    yt_safe_name = species.replace('+', '_plus').replace('-','_minus')
    token = species
    
    def _field(field, data):
        # 1. 提取基础变量，增加提取 temperature
        n_H = data[('gas','number_density_H')].to('cm**-3').value
        colDen_H = data[('gas','column_density_H')].to('cm**-2').value
        T = data[('gas', 'temperature')].to('K').value # <--- 新增
        
        # 2. 限制安全边界 (增加 T 的边界)
        nH_min, nH_max = lookup.table.nH_values.min(), lookup.table.nH_values.max()
        col_min, col_max = lookup.table.col_density_values.min(), lookup.table.col_density_values.max()
        T_min, T_max = lookup.table.T_values.min(), lookup.table.T_values.max()

        n_H_safe = np.clip(n_H, nH_min, nH_max)
        col_safe = np.clip(colDen_H, col_min, col_max)
        T_safe = np.clip(T, T_min, T_max) # <--- 新增

        # 3. 传入 T_safe 查表！(必须要有 4 个参数)
        densities = lookup.number_densities([token], n_H_safe, col_safe, T_safe)
        densities[token] = np.nan_to_num(densities[token], nan=0.0)
        densities[token][T > 100000.0] = 0.0

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
        
    ds.add_field(name=('gas', 'temperature_error'), function=_temperature_error, sampling_type="cell", units="", force_override=True)
    ds.add_field(name=('gas', 'Halpha_luminosity'), function=_Halpha_luminosity, sampling_type="cell", units="erg/s/cm**3", force_override=True)
    print("Added derived fields: 'temp_neutral', 'temperature', 'ionized_mask', 'Halpha_luminosity'.")
