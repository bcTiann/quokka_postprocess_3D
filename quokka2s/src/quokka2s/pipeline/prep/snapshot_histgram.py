import numpy as np
import matplotlib.pyplot as plt
from quokka2s.tables import load_table, plot_sampling_histogram
import yt
from yt.units import mh
from quokka2s.pipeline.prep import config as cfg
import quokka2s as q2s
from quokka2s.pipeline.prep import physics_fields as phys
from quokka2s.despotic_tables import compute_average




table = load_table("/Users/baochen/quokka_postprocessing/output_tables_testBig/despotic_table.npz")


ds = yt.load(cfg.YT_DATASET_PATH)
phys.add_all_fields(ds)
provider = q2s.YTDataProvider(ds)
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
temp_3d, temp_3d_extent = provider.get_slab_z(
    field=('gas', 'temperature')
)

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


X_H = cfg.X_H
m_H = mh.in_cgs()
density_3d, density_3d_extent = provider.get_slab_z(
    field=('gas', 'density')
)


n_H_3d = (density_3d * X_H) / m_H


n_H_array = n_H_3d.in_cgs().to_ndarray() if hasattr(n_H_3d, "in_cgs") else np.asarray(n_H_3d)
col_den_array = average_N_3d.in_cgs().to_ndarray() if hasattr(average_N_3d, "in_cgs") else np.asarray(average_N_3d)
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

ax = plot_sampling_histogram(table, log_samples, log_space=True)

ax.set_title("Snapshot sampling vs DESPOTIC failures")
plt.savefig("snapshot_hist.png", dpi=800)
plt.show()
