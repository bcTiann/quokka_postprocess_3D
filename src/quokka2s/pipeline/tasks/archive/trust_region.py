"""TrustRegionTask: focus diagnostic on the regime where DESPOTIC is reliable.

DESPOTIC's equilibrium chemistry/cooling is valid when chemistry timescales
are short compared to dynamical times and the gas hasn't been recently
shocked.  In practice that means *dense, not-too-hot* cells.  We define a
mask in 3-D (per cell):

    trust = (log10 ρ  >  rho_threshold)  &  (log10 T_QK  <  T_QK_threshold)

with defaults log10 ρ > -23 (n_H > ~5 cm^-3, CNM-and-denser) and
log10 T_QK < 4 (T_QK < 10^4 K, excludes SN bubbles).  Inside the mask we
plot four diagnostics that answer "in the regime where DESPOTIC should be
trustworthy, what does it actually say compared to QUOKKA?":

    [Panel 1]  Spatial slice with mask overlay
        Show one (y, z) slice (default = mid-box).  Cells in the trust
        region are coloured by log10 ρ; non-trust cells are greyed out.
        Tells you what fraction of the box / where in physical space the
        trustworthy gas lives.

    [Panel 2]  (log ρ, log T_QK) phase plot with mask overlay
        Mass-weighted 2-D histogram.  Trust cells coloured (mass per
        bin); non-trust bins overlaid in transparent grey.  Tells you
        what corner of phase space the mask carves out.

    [Panel 3]  1-D mass-weighted T histograms (trust cells only)
        Blue line  = log10 T_QUOKKA  distribution.
        Red  line  = log10 T_DESPOTIC distribution.
        If DESPOTIC's extra cooling channels (CO, [CII]) really push the
        equilibrium colder than QUOKKA's simpler cooling, the red curve
        sits to the LEFT of the blue.

    [Panel 4]  Joint (log T_QK, log T_DSP) 2-D distribution + 1:1 line
        Mass-weighted 2-D histogram of trust cells only.
        Points BELOW the 1:1 dashed line  →  T_DSP < T_QK (DESPOTIC
        colder, as expected).  Points ABOVE  →  reverse, which would be
        unphysical for this regime.

Output: trust_region.png in OUTPUT_DIR.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap

from ..base import AnalysisTask, PipelinePlotContext


class TrustRegionTask(AnalysisTask):
    """4-panel diagnostic on the DESPOTIC-trustworthy gas regime."""

    def __init__(self, config,
                 rho_threshold:  float = -23.0,    # log10(g/cm^3)
                 T_QK_threshold: float = 4.0,      # log10(K)
                 slice_axis:     str   = 'x',
                 slice_index:    int | None = None,
                 figure_units:   str   = 'kpc',
                 bin_dex:        float = 0.1,
                 filename:       str   = 'trust_region.png'):
        super().__init__(config)
        self.rho_threshold  = float(rho_threshold)
        self.T_QK_threshold = float(T_QK_threshold)
        self.slice_axis     = slice_axis
        self.slice_index    = slice_index
        self.figure_units   = figure_units
        self.bin_dex        = float(bin_dex)
        self.filename       = filename
        self._axis_idx      = {'x': 0, 'y': 1, 'z': 2}[slice_axis]

    @staticmethod
    def _aligned_edges(lo: float, hi: float, step: float) -> np.ndarray:
        lo_snap = np.floor(lo / step) * step
        hi_snap = (np.ceil(hi / step) + 0.5) * step
        n_bins = int(np.round((hi_snap - lo_snap) / step))
        return np.linspace(lo_snap, hi_snap, n_bins + 1)

    def _take_slice(self, cube: np.ndarray, idx: int) -> np.ndarray:
        sl = [slice(None)] * 3
        sl[self._axis_idx] = int(idx)
        return cube[tuple(sl)]

    def compute(self, context: PipelinePlotContext) -> dict:
        p = context.provider
        rho_u, extent_dict = p.get_slab_z(('gas', 'density'))
        T_qk_u, _          = p.get_slab_z(('gas', 'temperature_quokka'))
        T_md_u, _          = p.get_slab_z(('gas', 'temperature_despotic'))
        dx_u, _ = p.get_slab_z(('boxlib', 'dx'))
        dy_u, _ = p.get_slab_z(('boxlib', 'dy'))
        dz_u, _ = p.get_slab_z(('boxlib', 'dz'))

        rho  = np.asarray(rho_u.in_cgs()).copy()
        T_qk = np.asarray(T_qk_u.in_cgs()).copy()
        T_md = np.asarray(T_md_u.in_cgs()).copy()
        dV   = (np.asarray(dx_u.in_cgs()) *
                np.asarray(dy_u.in_cgs()) *
                np.asarray(dz_u.in_cgs()))
        del rho_u, T_qk_u, T_md_u, dx_u, dy_u, dz_u

        mass = rho * dV
        del dV

        with np.errstate(divide='ignore', invalid='ignore'):
            log_rho  = np.log10(np.where(rho  > 0, rho,  np.nan))
            log_T_qk = np.log10(np.where(T_qk > 0, T_qk, np.nan))
            log_T_md = np.log10(np.where(T_md > 0, T_md, np.nan))

        del rho, T_qk, T_md

        mask = ((log_rho  > self.rho_threshold) &
                (log_T_qk < self.T_QK_threshold) &
                np.isfinite(log_T_md))

        # Mass / cell-count fractions captured by the mask.
        M_total = float(np.nansum(mass))
        M_trust = float(np.nansum(mass[mask]))
        N_total = int(np.prod(mask.shape))
        N_trust = int(mask.sum())
        f_M = M_trust / M_total if M_total > 0 else 0.0
        f_N = N_trust / N_total if N_total > 0 else 0.0
        print(f'TrustRegionTask: mask captures {N_trust} / {N_total} cells '
              f'({100*f_N:.2f}%), {100*f_M:.2f}% of total mass.')

        # Panel 1 — spatial slice.
        if self.slice_index is None:
            slice_idx = mask.shape[self._axis_idx] // 2
        else:
            slice_idx = int(self.slice_index)
        log_rho_slice = self._take_slice(log_rho, slice_idx).copy()
        mask_slice    = self._take_slice(mask,    slice_idx).copy()
        extent_unyt   = extent_dict[self.slice_axis]
        extent_units  = [float(v.in_units(self.figure_units).value)
                         for v in extent_unyt]

        # Panel 2 — (log ρ, log T_QK) 2-D mass-weighted histogram.
        rho_lo, rho_hi  = float(np.nanmin(log_rho)),  float(np.nanmax(log_rho))
        qk_lo,  qk_hi   = float(np.nanmin(log_T_qk)), float(np.nanmax(log_T_qk))
        rho_edges  = self._aligned_edges(rho_lo, rho_hi, self.bin_dex)
        T_qk_edges = self._aligned_edges(qk_lo,  qk_hi,  self.bin_dex)

        valid_qk  = np.isfinite(log_rho) & np.isfinite(log_T_qk) & np.isfinite(mass)
        H_all, _, _  = np.histogram2d(
            log_rho[valid_qk].ravel(),
            log_T_qk[valid_qk].ravel(),
            bins=[rho_edges, T_qk_edges],
            weights=mass[valid_qk].ravel(),
        )
        valid_in_mask = valid_qk & mask
        H_mask, _, _ = np.histogram2d(
            log_rho[valid_in_mask].ravel(),
            log_T_qk[valid_in_mask].ravel(),
            bins=[rho_edges, T_qk_edges],
            weights=mass[valid_in_mask].ravel(),
        )

        # Panel 3 — 1-D mass-weighted log T histograms (trust cells only).
        valid_md   = np.isfinite(log_T_md)
        in_mask    = mask & valid_qk & valid_md
        # Use a common log-T binning that covers both fields' data.
        t_lo = float(min(np.nanmin(log_T_qk[in_mask]) if in_mask.any() else 1.0,
                         np.nanmin(log_T_md[in_mask]) if in_mask.any() else 1.0))
        t_hi = float(max(np.nanmax(log_T_qk[in_mask]) if in_mask.any() else 4.0,
                         np.nanmax(log_T_md[in_mask]) if in_mask.any() else 4.0))
        # Pad a half-bin so both extremes land inside.
        T_edges = self._aligned_edges(t_lo, t_hi, self.bin_dex)
        hist_qk, _ = np.histogram(log_T_qk[in_mask].ravel(),
                                  bins=T_edges,
                                  weights=mass[in_mask].ravel())
        hist_md, _ = np.histogram(log_T_md[in_mask].ravel(),
                                  bins=T_edges,
                                  weights=mass[in_mask].ravel())

        # Panel 4 — joint 2-D (log T_QK, log T_DSP) for mask cells.
        joint_edges = T_edges
        H_joint, _, _ = np.histogram2d(
            log_T_qk[in_mask].ravel(),
            log_T_md[in_mask].ravel(),
            bins=[joint_edges, joint_edges],
            weights=mass[in_mask].ravel(),
        )

        return {
            'log_rho_slice':   log_rho_slice,
            'mask_slice':      mask_slice,
            'slice_idx':       int(slice_idx),
            'extent':          extent_units,

            'rho_edges':       rho_edges,
            'T_qk_edges':      T_qk_edges,
            'H_all':           H_all,
            'H_mask':          H_mask,

            'T_edges':         T_edges,
            'hist_qk':         hist_qk,
            'hist_md':         hist_md,

            'H_joint':         H_joint,

            'N_trust':         int(N_trust),
            'N_total':         int(N_total),
            'M_trust':         float(M_trust),
            'M_total':         float(M_total),
            'rho_threshold':   float(self.rho_threshold),
            'T_QK_threshold':  float(self.T_QK_threshold),
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        log_rho_slice = np.asarray(results['log_rho_slice'])
        mask_slice    = np.asarray(results['mask_slice']).astype(bool)
        extent        = list(results['extent'])
        ext_plot      = [extent[0], extent[1], extent[2], extent[3]]
        slice_idx     = int(results['slice_idx'])

        rho_edges  = np.asarray(results['rho_edges'])
        T_qk_edges = np.asarray(results['T_qk_edges'])
        H_all      = np.asarray(results['H_all'])
        H_mask     = np.asarray(results['H_mask'])

        T_edges = np.asarray(results['T_edges'])
        hist_qk = np.asarray(results['hist_qk'])
        hist_md = np.asarray(results['hist_md'])

        H_joint = np.asarray(results['H_joint'])

        N_trust = int(results['N_trust'])
        N_total = int(results['N_total'])
        M_trust = float(results['M_trust'])
        M_total = float(results['M_total'])
        rho_th  = float(results['rho_threshold'])
        T_qk_th = float(results['T_QK_threshold'])

        fig, axes = plt.subplots(
            2, 2, figsize=(12.5, 11),
            gridspec_kw={'hspace': 0.35, 'wspace': 0.30,
                         'left': 0.08, 'right': 0.96,
                         'top': 0.93,  'bottom': 0.08},
        )
        ax1, ax2 = axes[0]
        ax3, ax4 = axes[1]

        # ── Panel 1 — spatial slice, mask vs non-mask ─────────────────────
        # Show log10 ρ in viridis ONLY for mask cells; grey for non-mask.
        masked_rho = np.where(mask_slice, log_rho_slice, np.nan)
        non_mask_rho = np.where(~mask_slice, log_rho_slice, np.nan)
        vmin = float(np.nanpercentile(log_rho_slice, 1.0))
        vmax = float(np.nanpercentile(log_rho_slice, 99.0))
        # First: non-mask cells in grey (semi-transparent context).
        ax1.imshow(non_mask_rho.T, origin='lower', extent=ext_plot,
                   aspect='auto', cmap='Greys', alpha=0.35,
                   vmin=vmin, vmax=vmax)
        # Then: mask cells in viridis, full opacity.
        im1 = ax1.imshow(masked_rho.T, origin='lower', extent=ext_plot,
                         aspect='auto', cmap='viridis',
                         vmin=vmin, vmax=vmax)
        ax1.set_title(
            f'Trust mask: $\\log\\rho > {rho_th:g}$ & $\\log T_{{QK}} < {T_qk_th:g}$\n'
            f'slice {self.slice_axis} idx={slice_idx}   '
            f'(coloured = trust, grey = excluded)',
            fontsize=10,
        )
        plane = {'x': ('y', 'z'), 'y': ('x', 'z'), 'z': ('x', 'y')}[self.slice_axis]
        ax1.set_xlabel(f'{plane[0]} [{self.figure_units}]', fontsize=10)
        ax1.set_ylabel(f'{plane[1]} [{self.figure_units}]', fontsize=10)
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
        cbar1.set_label(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=9)
        cbar1.ax.tick_params(labelsize=8)

        # ── Panel 2 — phase plot of mass, trust vs full ───────────────────
        # Show full distribution as grey background, mask cells as colour overlay.
        extent2 = [rho_edges[0], rho_edges[-1], T_qk_edges[0], T_qk_edges[-1]]
        # Non-mask = H_all - H_mask (the cells outside the mask).
        H_outside = np.maximum(H_all - H_mask, 0.0)
        with np.errstate(divide='ignore'):
            log_out = np.where(H_outside > 0,
                               np.log10(np.where(H_outside > 0, H_outside, 1.0)),
                               np.nan)
            log_mask = np.where(H_mask > 0,
                                np.log10(np.where(H_mask > 0, H_mask, 1.0)),
                                np.nan)
        finite_all = np.concatenate([log_out[np.isfinite(log_out)],
                                     log_mask[np.isfinite(log_mask)]])
        if finite_all.size:
            v_lo = float(np.nanpercentile(finite_all, 1.0))
            v_hi = float(np.nanpercentile(finite_all, 99.0))
        else:
            v_lo, v_hi = 0.0, 1.0
        ax2.imshow(log_out.T, origin='lower', extent=extent2,
                   aspect='auto', cmap='Greys', alpha=0.5,
                   vmin=v_lo, vmax=v_hi)
        im2 = ax2.imshow(log_mask.T, origin='lower', extent=extent2,
                         aspect='auto', cmap='viridis',
                         vmin=v_lo, vmax=v_hi)
        # Mark threshold lines.
        ax2.axvline(rho_th, color='red', ls='--', lw=1.0, alpha=0.8)
        ax2.axhline(T_qk_th, color='red', ls='--', lw=1.0, alpha=0.8)
        ax2.set_xlabel(r'$\log_{10}\,\rho$ [g cm$^{-3}$]', fontsize=10)
        ax2.set_ylabel(r'$\log_{10}\,T_{\rm QUOKKA}$ [K]', fontsize=10)
        ax2.set_title(
            f'Mass per (ρ, T_QK) bin — trust coloured, excluded grey\n'
            f'trust = {100*M_trust/M_total:.2f}% of mass, '
            f'{100*N_trust/N_total:.2f}% of cells',
            fontsize=10,
        )
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
        cbar2.set_label(r'$\log_{10}\,M_{\rm bin}$ [g]', fontsize=9)
        cbar2.ax.tick_params(labelsize=8)

        # ── Panel 3 — 1-D mass-weighted T histograms (trust cells) ────────
        centres = 0.5 * (T_edges[:-1] + T_edges[1:])
        width   = T_edges[1] - T_edges[0]
        # Normalise to PDFs so the two histograms are visually comparable
        # even if total mass differs (it shouldn't — both use the same mask).
        total_qk = hist_qk.sum()
        total_md = hist_md.sum()
        if total_qk > 0:
            pdf_qk = hist_qk / (total_qk * width)
        else:
            pdf_qk = hist_qk
        if total_md > 0:
            pdf_md = hist_md / (total_md * width)
        else:
            pdf_md = hist_md
        ax3.step(centres, pdf_qk, where='mid', color='C0', lw=1.6,
                 label=r'$T_{\rm QUOKKA}$')
        ax3.step(centres, pdf_md, where='mid', color='C3', lw=1.6,
                 label=r'$T_{\rm DESPOTIC}$')
        ax3.set_xlabel(r'$\log_{10}\,T$ [K]', fontsize=10)
        ax3.set_ylabel('mass PDF', fontsize=10)
        ax3.set_title('1-D T distribution within trust mask\n(mass-weighted)',
                      fontsize=10)
        ax3.legend(fontsize=9, loc='best')
        ax3.grid(True, alpha=0.3)

        # ── Panel 4 — joint (T_QK, T_DSP) in mask + 1:1 line ──────────────
        extent4 = [T_edges[0], T_edges[-1], T_edges[0], T_edges[-1]]
        with np.errstate(divide='ignore'):
            log_J = np.where(H_joint > 0,
                             np.log10(np.where(H_joint > 0, H_joint, 1.0)),
                             np.nan)
        if np.isfinite(log_J).any():
            jv_lo = float(np.nanpercentile(log_J[np.isfinite(log_J)], 1.0))
            jv_hi = float(np.nanpercentile(log_J[np.isfinite(log_J)], 99.0))
        else:
            jv_lo, jv_hi = 0.0, 1.0
        im4 = ax4.imshow(log_J.T, origin='lower', extent=extent4,
                         aspect='equal', cmap='viridis',
                         vmin=jv_lo, vmax=jv_hi)
        # 1:1 dashed reference line.
        diag = np.array([T_edges[0], T_edges[-1]])
        ax4.plot(diag, diag, color='red', ls='--', lw=1.0,
                 label='1:1 ($T_{\\rm DSP} = T_{\\rm QK}$)')
        ax4.set_xlabel(r'$\log_{10}\,T_{\rm QUOKKA}$ [K]', fontsize=10)
        ax4.set_ylabel(r'$\log_{10}\,T_{\rm DESPOTIC}$ [K]', fontsize=10)
        ax4.set_title(
            'Joint  T_QK  vs  T_DSP  in trust mask\n'
            '(points BELOW dashed = DESPOTIC colder)',
            fontsize=10,
        )
        ax4.legend(fontsize=9, loc='upper left')
        cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.04, pad=0.02)
        cbar4.set_label(r'$\log_{10}\,M_{\rm bin}$ [g]', fontsize=9)
        cbar4.ax.tick_params(labelsize=8)

        out = context.config.output_dir / self.filename
        fig.savefig(str(out), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {out}')
