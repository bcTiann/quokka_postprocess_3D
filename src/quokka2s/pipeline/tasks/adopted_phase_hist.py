"""Current manuscript phase panels; independent of historical field aliases."""
from __future__ import annotations

import numpy as np


PANELS = (
    ('mass_T_QK', 'QUOKKA', 'mass'),
    ('mass_T_DSP', 'DESPOTIC', 'mass'),
    ('mass_T_2R', 'mixed', 'mass'),
    ('NH_rho', None, 'NH_rho'),
    ('halpha', 'mixed', 'halpha'),
    ('hi21', 'mixed', 'hi21'),
    ('cii', 'mixed', 'cii'),
    ('co10', 'DESPOTIC', 'co10'),
    ('co21', 'DESPOTIC', 'co21'),
)


def select_emissivities(t_quokka, despotic, cloudy):
    """DESPOTIC/analytic cold branch, Cloudy hot branch; CO always DESPOTIC.

    All dictionary values are volumetric emissivities in erg/s/cm^3.
    No missing line, negative value, or NaN is silently replaced with zero.
    """
    low = np.asarray(t_quokka) < 3000.0
    result = {key: np.where(low, despotic[key], cloudy[key])
              for key in ('cii', 'halpha', 'hi21')}
    result.update({key: np.asarray(despotic[key]) for key in ('co10', 'co21')})
    for key, value in result.items():
        if not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError(f'Invalid {key} emissivity')
    return result


def add_adopted_phase_chunk(histograms, rho, tq, td, column, emission, volume,
                            *, raw_mass_all_cells=False):
    """Accumulate the nine panels from the accepted emission calculation.

    Emission and DESPOTIC-dependent panels include only ``emission.valid``.
    With ``raw_mass_all_cells=True``, the raw QUOKKA-temperature and column
    mass panels include excluded cells as well. Per-line temperatures come
    directly from the emission result; this helper does not reselect branches
    or recompute line emissivities. Empty retained chunks are valid no-ops.
    """
    missing = {key for key, _, _ in PANELS} - histograms.keys()
    if missing:
        raise ValueError(f'Missing phase histograms: {sorted(missing)}')
    rho, tq, td, column = (np.asarray(value, dtype=float)
                          for value in (rho, tq, td, column))
    shape = rho.shape
    if any(value.shape != shape for value in (tq, td, column)):
        raise ValueError('Phase coordinate shapes must match')
    valid, excluded = np.asarray(emission.valid), np.asarray(emission.excluded)
    if (valid.dtype.kind != 'b' or excluded.dtype.kind != 'b'
            or valid.shape != shape or excluded.shape != shape
            or not np.array_equal(valid, ~excluded)):
        raise ValueError('Emission valid/excluded masks must be complementary boolean arrays')
    volume = np.asarray(volume, dtype=float)
    if volume.shape not in ((), shape):
        raise ValueError('Volume must be scalar or match the cell shape')
    for name, value in (('density', rho), ('QUOKKA temperature', tq),
                        ('column', column), ('volume', volume),
                        ('retained DESPOTIC temperature', td[valid])):
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError(f'Invalid {name}: expected finite positive values')
    keys = tuple(emission.line_keys)
    lines = tuple(key for key, _, _ in PANELS if not key.startswith('mass') and key != 'NH_rho')
    if len(keys) != len(set(keys)) or not set(lines).issubset(keys):
        raise ValueError('Emission must contain each plotted line exactly once')
    epsilon = np.asarray(emission.emissivity_erg_s_cm3, dtype=float)
    thermal = np.asarray(emission.thermal_temperature_K, dtype=float)
    if epsilon.shape != (len(keys), *shape) or thermal.shape != epsilon.shape:
        raise ValueError('Emission and thermal-temperature arrays must have line-first cell shape')
    for key in lines:
        index = keys.index(key)
        values, temperature = epsilon[index][valid], thermal[index][valid]
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f'Invalid retained {key} emissivity')
        if not np.isfinite(temperature).all() or np.any(temperature <= 0):
            raise ValueError(f'Invalid retained {key} thermal temperature')

    mass = rho * volume
    raw = np.ones(shape, dtype=bool) if raw_mass_all_cells else valid
    rows = [
        ('mass_T_QK', rho[raw], tq[raw], mass[raw]),
        ('mass_T_DSP', rho[valid], td[valid], mass[valid]),
        ('mass_T_2R', rho[valid], thermal[keys.index('halpha')][valid], mass[valid]),
        ('NH_rho', column[raw], rho[raw], mass[raw]),
    ]
    retained_volume = np.broadcast_to(volume, shape)[valid]
    rows.extend((key, rho[valid], thermal[keys.index(key)][valid],
                 epsilon[keys.index(key)][valid] * retained_volume) for key in lines)
    # Check every weight before modifying any histogram, including overflow in
    # conversion from density/emissivity to mass/luminosity.
    if any(not np.isfinite(weight).all() for _, _, _, weight in rows):
        raise ValueError('Nonfinite phase mass or luminosity')
    for key, x, y, weight in rows:
        if weight.size:
            histograms[key].add(np.log10(x), np.log10(y), weight)


class DexHistogram:
    """Streaming absolute sums in globally aligned, fixed-width dex bins.

    Grow the small histogram as new extrema arrive, not the cell arrays.
    Bins are left-closed/right-open, including exact dex-boundary values.
    """
    def __init__(self, step=0.2):
        self.step = float(step)
        if self.step <= 0:
            raise ValueError('Bin width must be positive')
        self.H = None
        self.origin = None
        self.total = 0.0
        self.count = 0

    def add(self, x, y, weight):
        x, y, weight = np.broadcast_arrays(x, y, weight)
        if not (np.isfinite(x).all() and np.isfinite(y).all()
                and np.isfinite(weight).all()) or np.any(weight < 0):
            raise ValueError('Nonfinite coordinates/weights or negative weights')
        ix = np.floor(x.ravel() / self.step).astype(np.int64)
        iy = np.floor(y.ravel() / self.step).astype(np.int64)
        lo = np.array([ix.min(), iy.min()])
        hi = np.array([ix.max(), iy.max()]) + 1
        if self.H is not None:
            lo = np.minimum(lo, self.origin)
            hi = np.maximum(hi, self.origin + self.H.shape)
        shape = tuple(hi - lo)
        grown = np.zeros(shape)
        if self.H is not None:
            offset = self.origin - lo
            grown[offset[0]:offset[0] + self.H.shape[0],
                  offset[1]:offset[1] + self.H.shape[1]] = self.H
        flat = (ix - lo[0]) * shape[1] + (iy - lo[1])
        grown += np.bincount(flat, weights=weight.ravel(),
                             minlength=int(np.prod(shape))).reshape(shape)
        self.H, self.origin = grown, lo
        self.total += float(np.sum(weight, dtype=np.float64))
        self.count += weight.size

    def result(self):
        if self.H is None:
            raise ValueError('Empty histogram')
        return dict(H=self.H,
                    x_edges=(self.origin[0] + np.arange(self.H.shape[0] + 1)) * self.step,
                    y_edges=(self.origin[1] + np.arange(self.H.shape[1] + 1)) * self.step)


def plot_panels(panels, png, pdf):
    """Nine panels, original absolute-bin coloring, with a shared mass scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from .phase_combined_plot import _unit_latex, COLORBAR_DYNAMIC_RANGE

    maxima = {}
    for key, _, group in PANELS:
        maxima[group] = max(maxima.get(group, 0.0), float(panels[key]['H'].max()))
    norms = {group: Normalize(np.log10(value) - np.log10(COLORBAR_DYNAMIC_RANGE),
                              np.log10(value))
             for group, value in maxima.items() if value > 0}
    temp_panels = [panels[key] for key, _, _ in PANELS if key != 'NH_rho']
    rho_lim = (min(p['x_edges'][0] for p in temp_panels),
               max(p['x_edges'][-1] for p in temp_panels))
    temp_lim = (min(p['y_edges'][0] for p in temp_panels),
                max(p['y_edges'][-1] for p in temp_panels))
    titles = {'co10': 'CO(1-0)', 'co21': 'CO(2-1)', 'cii': 'C II',
              'halpha': r'H$\alpha$', 'hi21': 'H I 21 cm'}
    fig = plt.figure(figsize=(12.8, 12.3))
    grid = fig.add_gridspec(3, 3, left=.085, right=.985, bottom=.06,
                            top=.96, wspace=.32, hspace=.42)
    for index, (key, temperature, group) in enumerate(PANELS):
        inner = grid[index // 3, index % 3].subgridspec(2, 1,
                      height_ratios=[.055, 1], hspace=.07)
        cax, ax = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        panel = panels[key]
        values = np.ma.masked_less_equal(panel['H'], 0)
        im = ax.pcolormesh(panel['x_edges'], panel['y_edges'],
                          np.ma.log10(values).T, cmap='viridis_r',
                          norm=norms.get(group, Normalize(0, 1)), rasterized=True)
        is_mass = key.startswith('mass') or key == 'NH_rho'
        unit = _unit_latex('g' if is_mass else 'erg/s')
        quantity = r'M_{\rm bin}' if is_mass else r'L_{\rm bin}'
        label = f'{titles.get(key, "")}  ' + rf'$\log_{{10}} {quantity}$ [${unit}$]'
        bar = fig.colorbar(im, cax=cax, orientation='horizontal')
        bar.ax.tick_params(labelsize=8, top=True, labeltop=True,
                           bottom=False, labelbottom=False)
        cax.set_title(label.strip(), fontsize=10, pad=5)
        if key == 'NH_rho':
            ax.set_xlabel(r'$\log_{10} N_{\rm H}$ [cm$^{-2}$]')
            ax.set_ylabel(r'$\log_{10} \rho$ [g cm$^{-3}$]')
        else:
            ax.set_xlim(*rho_lim)
            ax.set_ylim(*temp_lim)
            ax.set_xlabel(r'$\log_{10} \rho$ [g cm$^{-3}$]')
            ax.set_ylabel(rf'$\log_{{10}} T_{{\rm {temperature}}}$ [K]')
        ax.tick_params(labelsize=9)
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
