"""Check monotonicity of g(T) = T / [(γ-1)·μ] at sim-sampled (nH, N_H) points.

Equation: T = (γ-1)·E_int·μ·mH/(ρ·kB)
Rearranged: E_int·mH/(ρ·kB) = T / [(γ-1)·μ] ≡ g(T)

If g(T) is strictly monotone increasing in T, the bisection has a unique root.
Panel layout mirrors table_diagnostics_density_probes.png:
  - probes chosen by rank-sampling the sim 2D histogram (core→edge)
  - top-right inset shows sim histogram + red star at probe location
"""

import sys
import math
sys.path.insert(0, "/Users/baochen/quokka_postprocessing/quokka2s/src")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

from quokka2s.tables import load_table

TABLE_PATH   = "/Users/baochen/quokka_postprocessing/output_tables_3D_GOW_T2e7/despotic_table.npz"
SAMPLES_PATH = "/Users/baochen/quokka_postprocessing/quokka2s/src/log_samples.npy"
OUT_PATH     = "/Users/baochen/quokka_postprocessing/mu_gamma_monotonicity_check.png"
OUT_PATH_MAP = "/Users/baochen/quokka_postprocessing/mu_gamma_monotonicity_map.png"
N_PROBES     = 36
N_COLS       = 6


# ── helpers (mirror table_diagnostics.py) ────────────────────────────────────

def _log_edges(vals):
    log_v = np.log10(vals)
    d = np.diff(log_v)
    e = np.empty(len(vals) + 1)
    e[1:-1] = log_v[:-1] + d / 2
    e[0]    = log_v[0]  - d[0]  / 2
    e[-1]   = log_v[-1] + d[-1] / 2
    return e


def _density_rank_probes(log_samples, nH_vals, col_vals, n_probes=36):
    counts, _, _ = np.histogram2d(
        log_samples[:, 0], log_samples[:, 1],
        bins=[_log_edges(nH_vals), _log_edges(col_vals)],
    )
    flat     = counts.ravel()
    occupied = np.where(flat > 0)[0]
    order    = occupied[np.argsort(flat[occupied])[::-1]]
    n_occ    = len(order)
    pick_ranks = np.unique(
        np.round(np.linspace(0, n_occ - 1, n_probes)).astype(int)
    )
    probes = []
    for rank_i in pick_ranks:
        bin_flat = int(order[rank_i])
        nH_idx, col_idx = divmod(bin_flat, len(col_vals))
        rank_pct = int(round(100 * rank_i / max(n_occ - 1, 1)))
        probes.append({
            'rank_pct': rank_pct,
            'nH':       float(nH_vals[nH_idx]),
            'col':      float(col_vals[col_idx]),
            'n_cells':  int(flat[bin_flat]),
        })
    return probes


# ── load table ────────────────────────────────────────────────────────────────
print("Loading table...")
table    = load_table(TABLE_PATH)
T_vals   = table.T_values            # (n_T,)
nH_vals  = table.nH_values           # (n_nH,)
col_vals = table.col_density_values  # (n_col,)
mu_3d    = table.mu_values           # (n_nH, n_col, n_T)
cv_3d    = table.cv_values           # (n_nH, n_col, n_T)

print(f"T range: {T_vals.min():.1f} – {T_vals.max():.1f} K  ({len(T_vals)} pts)")

# guard
cv_safe  = np.where(np.isfinite(cv_3d) & (cv_3d > 0), cv_3d, np.nan)
mu_safe  = np.where(np.isfinite(mu_3d) & (mu_3d > 0), mu_3d, np.nan)
gamma_3d = (cv_safe + 1.0) / cv_safe
denom_3d = (gamma_3d - 1.0) * mu_safe       # (γ-1)·μ
g_3d     = T_vals[None, None, :] / denom_3d  # (n_nH, n_col, n_T)

# global monotonicity check
diff_g  = np.diff(g_3d, axis=2)
valid   = np.isfinite(diff_g)
n_bad   = int(np.sum((diff_g <= 0) & valid))
n_valid = int(valid.sum())
print(f"Non-monotone grid points: {n_bad} / {n_valid} "
      f"({100*n_bad/n_valid:.4f}%)")
if n_bad == 0:
    print("✅ g(T) is globally monotone — bisection root is unique everywhere.")
else:
    print("⚠️  Non-monotone regions exist — bisection may be ill-posed there.")

# ── load sim samples & pick probes ───────────────────────────────────────────
log_samples = np.load(SAMPLES_PATH)
probes      = _density_rank_probes(log_samples, nH_vals, col_vals, N_PROBES)
print(f"\nSelected {len(probes)} density-rank probes (core → edge).")

# ── compute g(T) at each probe by interpolation from g_3d ────────────────────
# nearest-neighbour in log space (table has 35 pts, interpolation overkill)
def _nearest_idx(val, arr):
    return int(np.argmin(np.abs(np.log10(arr) - np.log10(val))))

probe_g   = {}   # key=(nH,col) → g array (n_T,)
probe_mono = {}  # key=(nH,col) → bool (all differences positive?)
for p in probes:
    i_nH = _nearest_idx(p['nH'], nH_vals)
    i_col = _nearest_idx(p['col'], col_vals)
    g_curve = g_3d[i_nH, i_col, :]
    key = (p['nH'], p['col'])
    probe_g[key]    = g_curve
    dg = np.diff(g_curve[np.isfinite(g_curve)])
    probe_mono[key] = bool(np.all(dg > 0)) if dg.size > 0 else True

n_nonmono = sum(1 for v in probe_mono.values() if not v)
print(f"Probes with non-monotone g(T): {n_nonmono} / {len(probes)}")

# ── inset histogram (reused across panels) ───────────────────────────────────
ins_h, ins_xe, ins_ye = np.histogram2d(
    log_samples[:, 0], log_samples[:, 1], bins=40)
ins_h_masked = np.where(ins_h > 0, ins_h, np.nan)
ins_vmin = float(np.nanmin(ins_h_masked))
ins_vmax = float(np.nanmax(ins_h_masked))

# ── plot ──────────────────────────────────────────────────────────────────────
n_panels = len(probes)
n_rows   = math.ceil(n_panels / N_COLS)

fig, axes = plt.subplots(n_rows, N_COLS,
                         figsize=(4.5 * N_COLS, 4.2 * n_rows),
                         squeeze=False)

for panel_i, p in enumerate(probes):
    row_i = panel_i // N_COLS
    col_i = panel_i % N_COLS
    ax    = axes[row_i, col_i]
    key   = (p['nH'], p['col'])

    g_curve = probe_g[key]
    mono    = probe_mono[key]

    color = '#1f77b4' if mono else '#d62728'   # blue=monotone, red=non-monotone
    ax.plot(T_vals, g_curve, color=color, lw=1.6, drawstyle='steps-mid')

    # mark non-monotone dips
    if not mono:
        dg = np.diff(g_curve)
        bad_T = T_vals[:-1][dg <= 0]
        bad_g = g_curve[:-1][dg <= 0]
        ax.scatter(bad_T, bad_g, color='red', s=25, zorder=5, marker='v')
        ax.set_facecolor('#fff0f0')

    ax.set_xscale('log')
    ax.set_yscale('log')
    status_str = '' if mono else '  ⚠ non-monotone'
    ax.set_title(
        f'top {p["rank_pct"]}%  nH={p["nH"]:.1e}  N_H={p["col"]:.1e}{status_str}',
        fontsize=7, color='black' if mono else 'darkred')
    ax.grid(True, alpha=0.2, ls='--', lw=0.3)
    if col_i == 0:
        ax.set_ylabel('g(T) = T / [(γ-1)·μ]  [K]', fontsize=8)
    if row_i == n_rows - 1 or panel_i == n_panels - 1:
        ax.set_xlabel('T [K]', fontsize=8)

    # inset: sim histogram + probe marker
    ax_ins = ax.inset_axes([0.60, 0.56, 0.38, 0.40])
    ax_ins.set_facecolor('none')
    ax_ins.patch.set_alpha(0.0)
    ax_ins.pcolormesh(
        ins_ye, ins_xe, ins_h_masked,
        cmap='Blues',
        norm=LogNorm(vmin=ins_vmin, vmax=ins_vmax),
        shading='auto', alpha=0.7,
    )
    ax_ins.plot(np.log10(p['col']), np.log10(p['nH']),
                'r*', ms=7, zorder=10, markeredgewidth=0.3,
                markeredgecolor='darkred')
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])
    for spine in ax_ins.spines.values():
        spine.set_linewidth(0.5)
        spine.set_alpha(0.4)

# hide unused panels
for panel_i in range(n_panels, n_rows * N_COLS):
    axes[panel_i // N_COLS, panel_i % N_COLS].set_visible(False)

mono_global = "globally monotone" if n_bad == 0 else f"{n_bad} non-monotone table points"
fig.suptitle(
    f'g(T) = T / [(γ-1)·μ]  at sim density-rank probes  (core → edge)\n'
    f'Blue = monotone (bisection valid)  |  Red = non-monotone  |  '
    f'Global: {mono_global}',
    fontsize=12)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT_PATH}")

# ── Figure 2: 2D heatmap — min_slope over full (nH, col) grid ────────────────
# min_slope[i,j] = min(dg/dT) over T — negative means non-monotone
dT      = np.diff(T_vals)                          # (n_T-1,)
dg      = np.diff(g_3d, axis=2)                   # (n_nH, n_col, n_T-1)
dg_dT   = dg / dT[None, None, :]                  # (n_nH, n_col, n_T-1)
valid   = np.isfinite(dg_dT)
min_slope = np.where(
    np.any(valid, axis=2),
    np.nanmin(np.where(valid, dg_dT, np.inf), axis=2),
    np.nan
)

# sim histogram on the same log(nH) × log(col) grid for contour overlay
sim_h, sim_xe, sim_ye = np.histogram2d(
    log_samples[:, 0], log_samples[:, 1],
    bins=[_log_edges(nH_vals), _log_edges(col_vals)],
)

fig2, ax2 = plt.subplots(figsize=(9, 6))

# symmetric colormap centred on 0: negative (red) = non-monotone
v_abs = np.nanpercentile(np.abs(min_slope[np.isfinite(min_slope)]), 99)
v_abs = max(v_abs, 1.0)
norm2 = mcolors.TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs)

# use actual nH / col values + log scale axes (no pre-taking of log10)
# need bin edges in real space for pcolormesh
nH_edges  = 10 ** _log_edges(nH_vals)
col_edges = 10 ** _log_edges(col_vals)

im2 = ax2.pcolormesh(
    col_edges, nH_edges, min_slope,
    norm=norm2, cmap='RdBu', shading='auto')
cb2 = fig2.colorbar(im2, ax=ax2,
    label=r'Minimum slope $\min_{T}\,[\mathrm{d}g/\mathrm{d}T]$ (dimensionless)')

ax2.set_xscale('log')
ax2.set_yscale('log')

# sim data contour overlay — midpoints in real space
xe_mid = 10 ** (0.5 * (sim_xe[:-1] + sim_xe[1:]))   # nH  real values
ye_mid = 10 ** (0.5 * (sim_ye[:-1] + sim_ye[1:]))   # col real values
ax2.contour(
    ye_mid, xe_mid, sim_h,
    levels=[1, 10, 100, 1000],
    colors='black', linewidths=[0.6, 0.9, 1.2, 1.5],
    linestyles='solid', alpha=0.7)
# invisible proxy for legend
for lvl, lw in zip([1, 10, 100, 1000], [0.6, 0.9, 1.2, 1.5]):
    ax2.plot([], [], color='black', lw=lw, label=f'Gas-cell count $\\geq {lvl}$')

ax2.set_xlabel(r'Hydrogen column density $N_{\rm H}$ (cm$^{-2}$)', fontsize=12)
ax2.set_ylabel(r'Hydrogen volume density $n_{\rm H}$ (cm$^{-3}$)', fontsize=12)
ax2.set_title(
    r'Bisection validity of $g(T) \equiv T\,/\,[(\gamma - 1)\,\mu(T)]$'
    ' over the DESPOTIC lookup-table grid\n'
    'Black contours: simulation gas-cell count',
    fontsize=11)
ax2.legend(fontsize=8, loc='upper left', framealpha=0.7)
ax2.grid(True, alpha=0.2, ls='--', lw=0.4, which='both')

plt.tight_layout()
plt.savefig(OUT_PATH_MAP, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"Saved: {OUT_PATH_MAP}")
