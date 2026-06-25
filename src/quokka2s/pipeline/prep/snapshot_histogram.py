import numpy as np
import matplotlib.pyplot as plt
from ...tables import load_table, plot_sampling_histogram
import yt
from yt.units import mh, kpc
from . import config as cfg
from ...data_handling import YTDataProvider
from ...analysis import along_sight_cumulation
from . import physics_fields as phys

table = load_table(cfg.DESPOTIC_TABLE_PATH)

ds = yt.load(cfg.YT_DATASET_PATH)
# phys.add_all_fields(ds)  # 注释掉，避免添加temperature字段
ds.add_field(name=('gas', 'number_density_H'), function=phys._number_density_H, sampling_type="cell", units="cm**-3", force_override=True)
ds.add_field(name=('gas', 'column_density_H'), function=phys._column_density_H, sampling_type="cell", units="cm**-2", force_override=True)
provider = YTDataProvider(ds)
dx_3d, dx_3d_extent = provider.get_slab_z(('boxlib', 'dx'))
dx_projection = dx_3d.sum(axis=0)

dy_3d, dy_3d_extent = provider.get_slab_z(('boxlib', 'dy'))
dy_projection = dy_3d.sum(axis=0)

dz_3d, dz_3d_extent = provider.get_slab_z(('boxlib', 'dz'))
dz_projection = dz_3d.sum(axis=0)

dv_3d = dx_3d * dy_3d * dz_3d

factor = 1
nx, ny, nz = dy_3d.shape
mid_z = nz//factor//2
mid_x = nx//factor//2

X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_slab_z(
    field=('gas', 'density')
)

density_3d = provider.downsample_3d_array(density_3d, factor=factor)
##################################
n_H_3d = (density_3d * X_H) / m_H

dx_3d, dx_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dx')
)
dy_3d, dy_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dy')
)
dz_3d, dz_3d_extent = provider.get_slab_z(
    field=('boxlib', 'dz')
)

# Streaming harmonic mean (same as _column_density_H in physics_fields.py).
# Lateral ±x, ±y rays get the box-exterior extension  L_ext * <n_H>(z); ±z
# does not (the stratified box already covers the disk vertically).  Keep
# this in sync with cfg.COLUMN_EXTENSION_LATERAL_KPC and physics_fields.py.
L_ext_kpc = float(cfg.COLUMN_EXTENSION_LATERAL_KPC)
if L_ext_kpc > 0.0:
    L_ext_qty   = (L_ext_kpc * kpc).in_units('cm')        # unyt, cm
    n_bar_z     = n_H_3d.mean(axis=(0, 1))                # unyt, cm^-3
    N_ext_lat_3d = (L_ext_qty * n_bar_z)[None, None, :]   # unyt, cm^-2
else:
    N_ext_lat_3d = None

inv_sum = None
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
    inc = 1.0 / N
    del N
    if inv_sum is None:
        inv_sum = inc
    else:
        inv_sum = inv_sum + inc
    del inc
average_N_3d = 6.0 / inv_sum

n_H_array = n_H_3d.ravel()
col_den_array = average_N_3d.ravel()

finite_mask = (
    np.isfinite(n_H_array)
    & np.isfinite(col_den_array)
    & (n_H_array > 0.0)
    & (col_den_array > 0.0)
)
log_samples = np.column_stack(
    (
        np.log10(n_H_array[finite_mask]),
        np.log10(col_den_array[finite_mask]),
    )
)
np.save("log_samples.npy", log_samples)

ax = plot_sampling_histogram(table, log_samples, log_space=True, show_failure_mask=False)

ax.set_title("Snapshot sampling vs DESPOTIC failures")
plt.savefig("snapshot_hist.png", dpi=800)
plt.show()
