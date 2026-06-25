from __future__ import annotations

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from ..base import AnalysisTask, PipelinePlotContext
from ..prep import config as cfg
from ..prep.physics_fields import ensure_table_lookup
from ..utils import weighted_percentile


# ── Coverage helpers ────────────────────────────────────────────────────────
def _log_edges(vals: np.ndarray) -> np.ndarray:
    """Log-space bin edges centred on each grid value."""
    log_v = np.log10(vals)
    d = np.diff(log_v)
    e = np.empty(len(vals) + 1)
    e[1:-1] = log_v[:-1] + d / 2
    e[0]    = log_v[0]  - d[0]  / 2
    e[-1]   = log_v[-1] + d[-1] / 2
    return e


def _build_coverage(log_samples: np.ndarray,
                    nH_vals: np.ndarray,
                    col_vals: np.ndarray) -> np.ndarray:
    """Bool array (len(nH_vals), len(col_vals)): True where sim data exists."""
    counts, _, _ = np.histogram2d(
        log_samples[:, 0],   # log10(nH)
        log_samples[:, 1],   # log10(N_H)
        bins=[_log_edges(nH_vals), _log_edges(col_vals)],
    )
    return counts > 0


def _nearest_idx(val: float, arr: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.log10(arr) - np.log10(val))))


def _covered(nH: float, col: float,
             coverage: np.ndarray | None,
             nH_vals: np.ndarray, col_vals: np.ndarray) -> bool:
    """Return True if (nH, col) falls in a table bin that has sim data."""
    if coverage is None:
        return True
    return bool(coverage[_nearest_idx(nH, nH_vals), _nearest_idx(col, col_vals)])

def _density_rank_probes(
        log_samples: np.ndarray,
        nH_vals: np.ndarray,
        col_vals: np.ndarray,
        n_probes: int = 36,
) -> list[dict]:
    """Pick n_probes bins uniformly in rank-space of the sim 2D histogram.

    All occupied bins are sorted by count descending (rank 0 = densest core,
    rank N-1 = least-dense edge).  Probe indices are chosen at evenly-spaced
    positions along that ranked list, so every density level gets equal
    representation — unlike CDF-mass sampling which crowds the core.

    'rank_pct' in each dict is the percentile position among occupied bins
    (0 = core, 100 = edge).
    """
    counts, _, _ = np.histogram2d(
        log_samples[:, 0], log_samples[:, 1],
        bins=[_log_edges(nH_vals), _log_edges(col_vals)],
    )
    flat = counts.ravel()
    occupied = np.where(flat > 0)[0]
    if len(occupied) == 0:
        return []

    # sort occupied bins by density descending
    order = occupied[np.argsort(flat[occupied])[::-1]]
    n_occ = len(order)

    # evenly-spaced rank indices (always includes rank 0 = core and rank N-1 = edge)
    pick_ranks = np.unique(
        np.round(np.linspace(0, n_occ - 1, n_probes)).astype(int)
    )

    probes: list[dict] = []
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


SPECIES_CFG = [
    {'name': 'CO',   'color': '#1f77b4'},
    {'name': 'C+',   'color': '#ff7f0e'},
    {'name': 'HCO+', 'color': '#2ca02c'},
]


class TableDiagnosticsTask(AnalysisTask):
    """Plot lumPerH diagnostics from the DESPOTIC table for CO, C+, HCO+.

    Does NOT read the yt simulation dataset — only the lookup table.
    Produces two figures:

    table_diagnostics_lumPerH_vs_T.png (1 × 4)
      ax0: normalized lumPerH vs T, each species at its own lum-weighted N_H_ref
      ax1-3: CO / C+ / HCO+ absolute lumPerH vs T at 5 nH probes (at N_H_ref)

    table_diagnostics_lumPerH_vs_colDen.png (1 × 3)
      ax0-2: CO / C+ / HCO+ lumPerH vs N_H at 5 T probes (mid nH)
    """

    def prepare(self, context: PipelinePlotContext) -> None:
        pass  # no simulation data needed

    # ------------------------------------------------------------------
    # internal helper
    # ------------------------------------------------------------------
    def _lum_weighted_col(self, lookup, sp: str,
                          nH_fixed: float, T_ref: float,
                          col_vals: np.ndarray) -> float:
        """Luminosity-weighted median N_H for one species at (nH_fixed, T_ref).

        Scans lumPerH across col_vals and returns the weighted-median column
        density where emission is concentrated.  Falls back to col_vals midpoint
        if all weights are zero (species absent in this regime).
        """
        lums = np.array([
            float(lookup.line_field(sp, 'lumPerH',
                                    np.array([nH_fixed]),
                                    np.array([c]),
                                    np.array([T_ref]))[0])
            for c in col_vals
        ])
        lums = np.maximum(lums, 0.0)
        result = weighted_percentile(col_vals, lums, 50)
        if np.isnan(result):
            return float(col_vals[len(col_vals) // 2])
        return result

    # ------------------------------------------------------------------
    # pipeline interface
    # ------------------------------------------------------------------
    def compute(self, context: PipelinePlotContext) -> dict:
        lookup   = ensure_table_lookup(cfg.DESPOTIC_TABLE_PATH)
        T_vals   = lookup.table.T_values
        nH_vals  = lookup.table.nH_values
        col_vals = lookup.table.col_density_values

        # ── Load sim-data coverage (optional) ────────────────────────────────
        coverage: np.ndarray | None = None
        if os.path.exists(cfg.LOG_SAMPLES_PATH):
            log_samples = np.load(cfg.LOG_SAMPLES_PATH)
            coverage = _build_coverage(log_samples, nH_vals, col_vals)
            print(f"[TableDiagnostics] Coverage loaded from {cfg.LOG_SAMPLES_PATH}: "
                  f"{int(coverage.sum())}/{coverage.size} table bins have sim data.")
        else:
            print(f"[TableDiagnostics] {cfg.LOG_SAMPLES_PATH} not found — "
                  "coverage marking disabled (run snapshot_histogram.py first).")

        col_mid = col_vals[len(col_vals) // 2]
        nH_mid  = nH_vals[len(nH_vals) // 2]

        # ── Step A: find T_peak and N_H_ref per species ──────────────────────
        # First pass: T sweep at col_mid (cheap, just to locate T_peak)
        T_peak_vals: dict[str, float] = {}
        for sp in ['CO', 'C+', 'HCO+']:
            lums = np.array([
                float(lookup.line_field(sp, 'lumPerH',
                                        np.array([nH_mid]),
                                        np.array([col_mid]),
                                        np.array([T]))[0])
                for T in T_vals
            ])
            lums = np.maximum(lums, 0.0)
            T_peak_vals[sp] = float(T_vals[np.argmax(lums)])

        # Compute lum-weighted median N_H for each species at its own T_peak
        N_H_ref: dict[str, float] = {}
        for sp in ['CO', 'C+', 'HCO+']:
            N_H_ref[sp] = self._lum_weighted_col(
                lookup, sp, nH_mid, T_peak_vals[sp], col_vals)

        print("=== Table diagnostics: luminosity-weighted N_H_ref per species ===")
        for sp, val in N_H_ref.items():
            print(f"  {sp:5s}  T_peak={T_peak_vals[sp]:.2e} K  "
                  f"N_H_ref={val:.2e} cm⁻²  (col_mid={col_mid:.2e})")
        print("==================================================================")

        # ── Step B: T sweeps at N_H_ref (all species) ────────────────────────
        species_curves: dict[str, np.ndarray] = {}
        for sp in ['CO', 'C+', 'HCO+']:
            col_ref = N_H_ref[sp]
            lums = np.array([
                float(lookup.line_field(sp, 'lumPerH',
                                        np.array([nH_mid]),
                                        np.array([col_ref]),
                                        np.array([T]))[0])
                for T in T_vals
            ])
            species_curves[sp] = np.maximum(lums, 0.0)

        # ── Step C: multi-nH T sweeps at N_H_ref (all species) ───────────────
        nH_probes = [
            nH_vals[5],
            nH_vals[len(nH_vals) // 4],
            nH_vals[len(nH_vals) // 2],
            nH_vals[3 * len(nH_vals) // 4],
            nH_vals[-5],
        ]
        species_nH_curves: dict[str, dict] = {}
        for sp in ['CO', 'C+', 'HCO+']:
            col_ref = N_H_ref[sp]
            species_nH_curves[sp] = {}
            for nH in nH_probes:
                lums = np.array([
                    float(lookup.line_field(sp, 'lumPerH',
                                            np.array([nH]),
                                            np.array([col_ref]),
                                            np.array([T]))[0])
                    for T in T_vals
                ])
                species_nH_curves[sp][nH] = np.maximum(lums, 0.0)

        # ── Step D: N_H sweep at 5 T probes (all species, mid nH) ────────────
        n_T = len(T_vals)
        T_probes = [T_vals[i] for i in [
            n_T // 8, n_T // 4, n_T // 2, 3 * n_T // 4, 7 * n_T // 8]]

        species_col_curves: dict[str, dict] = {}
        for sp in ['CO', 'C+', 'HCO+']:
            species_col_curves[sp] = {}
            for T in T_probes:
                lums = np.array([
                    float(lookup.line_field(sp, 'lumPerH',
                                            np.array([nH_mid]),
                                            np.array([col]),
                                            np.array([T]))[0])
                    for col in col_vals
                ])
                species_col_curves[sp][T] = np.maximum(lums, 0.0)

        # ── Step E: 5×5 grid — T curves at all (nH_probe, col_probe) pairs ──
        # 用於 5×5 normalized 總覽圖 和 5×3 per-species 絕對值圖
        n_col = len(col_vals)
        col_probes = [col_vals[i] for i in [
            n_col // 8, n_col // 4, n_col // 2, 3 * n_col // 4, 7 * n_col // 8]]

        species_nH_col_curves: dict[str, dict] = {}
        for sp in ['CO', 'C+', 'HCO+']:
            species_nH_col_curves[sp] = {}
            for nH in nH_probes:
                species_nH_col_curves[sp][nH] = {}
                for col in col_probes:
                    lums = np.array([
                        float(lookup.line_field(sp, 'lumPerH',
                                                np.array([nH]),
                                                np.array([col]),
                                                np.array([T]))[0])
                        for T in T_vals
                    ])
                    species_nH_col_curves[sp][nH][col] = np.nan_to_num(np.maximum(lums, 0.0), nan=0.0)

        # ── Step F: density-CDF probes ────────────────────────────────────────
        density_probes: list[dict] | None = None
        density_probe_curves: dict | None = None
        log_samples_for_plot: np.ndarray | None = None
        if coverage is not None:
            log_samples_loaded = np.load(cfg.LOG_SAMPLES_PATH)
            log_samples_for_plot = log_samples_loaded
            density_probes = _density_rank_probes(
                log_samples_loaded, nH_vals, col_vals, n_probes=36)
            print(f"[TableDiagnostics] Density-CDF probes ({len(density_probes)} unique bins):")
            for p in density_probes:
                print(f"  rank={p['rank_pct']:3d}%  nH={p['nH']:.2e}  N_H={p['col']:.2e}"
                      f"  n_cells={p['n_cells']}")
            density_probe_curves = {}
            for p in density_probes:
                key = (p['nH'], p['col'])
                density_probe_curves[key] = {
                    sp: np.maximum(
                        np.array([
                            float(lookup.line_field(sp, 'lumPerH',
                                                    np.array([p['nH']]),
                                                    np.array([p['col']]),
                                                    np.array([T]))[0])
                            for T in T_vals
                        ]), 0.0)
                    for sp in ['CO', 'C+', 'HCO+']
                }

        return {
            'T_vals':                 T_vals,
            'nH_vals':                nH_vals,
            'col_vals':               col_vals,
            'nH_mid':                 nH_mid,
            'col_mid':                col_mid,
            'nH_probes':              nH_probes,
            'col_probes':             col_probes,
            'T_probes':               T_probes,
            'N_H_ref':                N_H_ref,
            'species_curves':         species_curves,
            'species_nH_curves':      species_nH_curves,
            'species_col_curves':     species_col_curves,
            'species_nH_col_curves':  species_nH_col_curves,
            'coverage':               coverage,
            'density_probes':         density_probes,
            'density_probe_curves':   density_probe_curves,
            'log_samples':            log_samples_for_plot,
        }

    def plot(self, context: PipelinePlotContext, results: dict) -> None:
        T_vals                = results['T_vals']
        nH_vals               = results['nH_vals']
        col_vals              = results['col_vals']
        nH_mid                = results['nH_mid']
        col_mid               = results['col_mid']
        nH_probes             = results['nH_probes']
        col_probes            = results['col_probes']
        T_probes              = results['T_probes']
        N_H_ref               = results['N_H_ref']
        species_curves        = results['species_curves']
        species_nH_curves     = results['species_nH_curves']
        species_col_curves    = results['species_col_curves']
        species_nH_col_curves = results['species_nH_col_curves']
        coverage              = results['coverage']

        def _line_kw(nH: float, col: float, base_lw: float = 1.8) -> dict:
            """Return plot kwargs: dashed + low alpha if probe not in sim data."""
            if _covered(nH, col, coverage, nH_vals, col_vals):
                return dict(lw=base_lw, ls='-', alpha=1.0)
            return dict(lw=base_lw * 0.8, ls='--', alpha=0.45)

        # ── Coverage report ────────────────────────────────────────────────
        if coverage is not None:
            uncovered: list[str] = []
            # probes used in Fig1 ax1-3 and Fig2 top (nH_probes × N_H_ref per species)
            for sp in ['CO', 'C+', 'HCO+']:
                for nH in nH_probes:
                    if not _covered(nH, N_H_ref[sp], coverage, nH_vals, col_vals):
                        uncovered.append(
                            f"  nH={nH:.2e}  N_H_ref[{sp}]={N_H_ref[sp]:.2e}"
                            f"  → Fig1-ax1-3 / Fig2-top ({sp} curve)")
            # probes used in 5×5 overview and 5×3 full (nH_probes × col_probes)
            seen: set[tuple] = set()
            for nH in nH_probes:
                for col in col_probes:
                    if not _covered(nH, col, coverage, nH_vals, col_vals):
                        key = (round(nH, 3), round(col, 3))
                        if key not in seen:
                            seen.add(key)
                            uncovered.append(
                                f"  nH={nH:.2e}  N_H={col:.2e}"
                                f"  → 5×5-overview / 5×3-full")
            if uncovered:
                print(f"[TableDiagnostics] {len(uncovered)} probe(s) NOT in sim data "
                      "(will be drawn as dashed / grey background):")
                for line in uncovered:
                    print(line)
            else:
                print("[TableDiagnostics] All probes are within sim data coverage.")

        sp_color = {s['name']: s['color'] for s in SPECIES_CFG}

        # ══════════════════════════════════════════════════════════════════════
        # Figure 1: lumPerH vs T  (1 × 4)
        #   ax0 — normalized all-species overview
        #   ax1-3 — absolute single-line per species at N_H_ref / nH_mid
        # ══════════════════════════════════════════════════════════════════════
        fig1, axes = plt.subplots(1, 4, figsize=(22, 5))
        ax_norm = axes[0]

        # ax0 — normalized, all species at their own N_H_ref
        for sp_cfg in SPECIES_CFG:
            sp   = sp_cfg['name']
            lums = species_curves[sp]
            peak = lums.max()
            norm = lums / peak if peak > 0 else lums
            ax_norm.plot(T_vals, norm, color=sp_cfg['color'], lw=2.0,
                         drawstyle='steps-mid',
                         label=f"{sp}  (N_H_ref={N_H_ref[sp]:.1e})")
            # T_CUTOFF removed 2026-06-13 — no per-species cutoff annotation.

        ax_norm.set_xscale('log')
        ax_norm.set_xlabel('Temperature [K]', fontsize=12)
        ax_norm.set_ylabel('Normalized lumPerH  (/ peak)', fontsize=11)
        ax_norm.set_title(
            f'All species  (nH={nH_mid:.1e} cm⁻³)\neach at its lum-weighted N_H_ref',
            fontsize=10)
        ax_norm.set_ylim(-0.05, 1.15)
        ax_norm.legend(fontsize=10, loc='upper right', framealpha=0.7)
        ax_norm.grid(True, alpha=0.25, ls='--', lw=0.5)

        # ax1-3 — absolute lumPerH vs T, multi-nH per species at N_H_ref.
        # (2026-06-13) T_CUTOFF dashed vertical line + red shaded region
        # removed — pipeline no longer zeroes lumPerH above any per-species
        # cutoff.  GOW network is still single-ionization only, but that's
        # documented elsewhere now.
        cmap_nH = plt.get_cmap('plasma', len(nH_probes))
        for col_idx, sp_cfg in enumerate(SPECIES_CFG, start=1):
            sp  = sp_cfg['name']
            ax  = axes[col_idx]
            for i, nH in enumerate(nH_probes):
                kw = _line_kw(nH, N_H_ref[sp])
                label = f'nH={nH:.1e}' + ('' if kw['ls'] == '-' else ' [no sim data]')
                ax.plot(T_vals, species_nH_curves[sp][nH],
                        color=cmap_nH(i), drawstyle='steps-mid',
                        label=label, **kw)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Temperature [K]', fontsize=12)
            ax.set_ylabel('lumPerH  [erg s⁻¹ H⁻¹]', fontsize=11)
            ax.set_title(
                f'{sp}  lumPerH vs T — multi nH\n'
                f'(N_H_ref={N_H_ref[sp]:.1e} cm⁻²)',
                fontsize=10)
            ax.legend(fontsize=8, loc='lower left', framealpha=0.7)
            ax.grid(True, alpha=0.25, ls='--', lw=0.5)

        fig1.suptitle('DESPOTIC Table: Luminosity vs Temperature', fontsize=13)
        plt.tight_layout()
        out1 = self.config.output_dir / 'table_diagnostics_lumPerH_vs_T.png'
        plt.savefig(str(out1), dpi=200, bbox_inches='tight')
        plt.close(fig1)
        print(f'Saved: {out1}')

        # ══════════════════════════════════════════════════════════════════════
        # Figure 1b: 5×5 all-species normalized grid
        #   rows (top→bottom) = N_H probes (low → high)
        #   cols (left→right) = nH probes  (low → high)
        #   each cell: CO / C+ / HCO+ normalized lumPerH vs T
        # ══════════════════════════════════════════════════════════════════════
        n_row = len(col_probes)
        n_col_grid = len(nH_probes)
        fig_ov, axes_ov = plt.subplots(n_row, n_col_grid,
                                        figsize=(4 * n_col_grid, 3.2 * n_row),
                                        sharex=True)

        for row_i, col_val in enumerate(col_probes):
            for col_i, nH_val in enumerate(nH_probes):
                ax = axes_ov[row_i, col_i]
                cell_covered = _covered(nH_val, col_val, coverage, nH_vals, col_vals)
                if not cell_covered:
                    ax.set_facecolor('#f0f0f0')
                for sp_cfg in SPECIES_CFG:
                    sp   = sp_cfg['name']
                    lums = species_nH_col_curves[sp][nH_val][col_val]
                    peak = lums.max()
                    norm = lums / peak if peak > 0 else lums
                    kw_ov = dict(lw=1.4, ls='-', alpha=1.0) if cell_covered \
                            else dict(lw=1.0, ls='--', alpha=0.5)
                    ax.plot(T_vals, norm, color=sp_cfg['color'],
                            drawstyle='steps-mid', label=sp, **kw_ov)
                if not cell_covered:
                    ax.text(0.97, 0.97, 'no sim data',
                            transform=ax.transAxes, fontsize=6, color='#888888',
                            ha='right', va='top')
                ax.set_xscale('log')
                ax.set_ylim(-0.05, 1.15)
                ax.grid(True, alpha=0.2, ls='--', lw=0.4)

                # top row: nH label
                if row_i == 0:
                    ax.set_title(f'nH={nH_val:.1e}', fontsize=9)
                # left col: N_H label
                if col_i == 0:
                    ax.set_ylabel(f'N_H={col_val:.1e}\nnorm. lumPerH', fontsize=8)
                # bottom row: x label
                if row_i == n_row - 1:
                    ax.set_xlabel('T [K]', fontsize=9)
                # top-left only: legend
                if row_i == 0 and col_i == n_col_grid - 1:
                    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)

        fig_ov.suptitle(
            'DESPOTIC Table: Normalized lumPerH vs T\n'
            '(rows = N_H probes, cols = nH probes)',
            fontsize=12)
        plt.tight_layout()
        out_ov = self.config.output_dir / 'table_diagnostics_lumPerH_overview_5x5.png'
        plt.savefig(str(out_ov), dpi=150, bbox_inches='tight')
        plt.close(fig_ov)
        print(f'Saved: {out_ov}')

        # ══════════════════════════════════════════════════════════════════════
        # Figure 1c: 5×3 per-species absolute  (rows = N_H, cols = species)
        #   each cell: 5 nH curves (absolute lumPerH vs T)
        # ══════════════════════════════════════════════════════════════════════
        fig_sp, axes_sp = plt.subplots(n_row, 3,
                                        figsize=(17, 3.8 * n_row),
                                        sharex=True)
        cmap_nH_grid = plt.get_cmap('plasma', len(nH_probes))

        for row_i, col_val in enumerate(col_probes):
            for col_j, sp_cfg in enumerate(SPECIES_CFG):
                sp  = sp_cfg['name']
                ax  = axes_sp[row_i, col_j]
                for i, nH_val in enumerate(nH_probes):
                    lums = species_nH_col_curves[sp][nH_val][col_val]
                    kw_sp = _line_kw(nH_val, col_val, base_lw=1.5)
                    label = f'nH={nH_val:.1e}' + ('' if kw_sp['ls'] == '-' else ' [no sim data]')
                    ax.plot(T_vals, lums, color=cmap_nH_grid(i),
                            drawstyle='steps-mid', label=label, **kw_sp)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.2, ls='--', lw=0.4)

                # top row: species label
                if row_i == 0:
                    ax.set_title(f'{sp}', fontsize=11, color=sp_color[sp])
                # left col: N_H label
                if col_j == 0:
                    ax.set_ylabel(f'N_H={col_val:.1e}\nlumPerH [erg/s/H]', fontsize=8)
                # bottom row: x label + legend
                if row_i == n_row - 1:
                    ax.set_xlabel('T [K]', fontsize=9)
                    ax.legend(fontsize=6, loc='lower left', framealpha=0.6)

        fig_sp.suptitle(
            'DESPOTIC Table: Absolute lumPerH vs T\n'
            '(rows = N_H probes,  cols = species,  curves = nH probes)',
            fontsize=12)
        plt.tight_layout()
        out_sp = self.config.output_dir / 'table_diagnostics_lumPerH_vs_T_full.png'
        plt.savefig(str(out_sp), dpi=150, bbox_inches='tight')
        plt.close(fig_sp)
        print(f'Saved: {out_sp}')

        # ══════════════════════════════════════════════════════════════════════
        # Figure 2: lumPerH_T_and_NH  (2 × 3)
        #   Top row    — lumPerH vs T at multi-nH (sensitivity to density)
        #   Bottom row — lumPerH vs N_H at multi-T (sensitivity to shielding)
        # ══════════════════════════════════════════════════════════════════════
        fig2, axes2 = plt.subplots(2, 3, figsize=(17, 10))
        cmap_nH = plt.get_cmap('plasma', len(nH_probes))
        cmap_T  = plt.get_cmap('viridis', len(T_probes))

        for j, sp_cfg in enumerate(SPECIES_CFG):
            sp = sp_cfg['name']

            # ── top row: lumPerH vs T, multi-nH ─────────────────────────────
            ax_top = axes2[0, j]
            for i, nH in enumerate(nH_probes):
                kw2 = _line_kw(nH, N_H_ref[sp])
                label2 = f'nH={nH:.1e}' + ('' if kw2['ls'] == '-' else ' [no sim data]')
                ax_top.plot(T_vals, species_nH_curves[sp][nH],
                            color=cmap_nH(i), drawstyle='steps-mid',
                            label=label2, **kw2)
            ax_top.set_xscale('log')
            ax_top.set_yscale('log')
            ax_top.set_xlabel('Temperature [K]', fontsize=11)
            ax_top.set_ylabel('lumPerH  [erg s⁻¹ H⁻¹]', fontsize=11)
            ax_top.set_title(
                f'{sp}  lumPerH vs T — multi nH\n'
                f'(N_H_ref={N_H_ref[sp]:.1e} cm⁻²)',
                fontsize=10)
            ax_top.legend(fontsize=7, loc='lower left', framealpha=0.7)
            ax_top.grid(True, alpha=0.25, ls='--', lw=0.5)

            # ── bottom row: lumPerH vs N_H, multi-T ─────────────────────────
            ax_bot = axes2[1, j]
            for i, T in enumerate(T_probes):
                ax_bot.plot(col_vals, species_col_curves[sp][T],
                            color=cmap_T(i), lw=1.8, drawstyle='steps-mid',
                            label=f'T={T:.1e} K')
            ax_bot.axvline(col_mid, color='grey', ls=':', lw=1.2, alpha=0.7,
                           label=f'col_mid={col_mid:.1e}')
            ax_bot.axvline(N_H_ref[sp], color=sp_color[sp], ls='--', lw=1.5,
                           label=f'{sp} N_H_ref={N_H_ref[sp]:.1e}')
            ax_bot.set_xscale('log')
            ax_bot.set_yscale('log')
            ax_bot.set_xlabel('N_H  [cm⁻²]', fontsize=11)
            ax_bot.set_ylabel('lumPerH  [erg s⁻¹ H⁻¹]', fontsize=11)
            ax_bot.set_title(f'{sp}  lumPerH vs N_H  (nH={nH_mid:.1e} cm⁻³)',
                             fontsize=10)
            ax_bot.legend(fontsize=7, loc='best', framealpha=0.7)
            ax_bot.grid(True, alpha=0.25, ls='--', lw=0.5)

            if sp == 'CO':
                ax_bot.text(0.02, 0.04,
                            'UV-shielding enables CO formation → lumPerH ↑,\n'
                            'saturates at high N_H',
                            transform=ax_bot.transAxes, fontsize=8, color='grey',
                            va='bottom')

        fig2.suptitle('DESPOTIC Table: Species Sensitivity to T and N_H', fontsize=13)
        plt.tight_layout()
        out2 = self.config.output_dir / 'table_diagnostics_lumPerH_T_and_NH.png'
        plt.savefig(str(out2), dpi=200, bbox_inches='tight')
        plt.close(fig2)
        print(f'Saved: {out2}')

        # ══════════════════════════════════════════════════════════════════════
        # Figure 3: density-CDF probes  (6 × 6 grid)
        #   Each panel = one (nH, N_H) probe chosen at a density-CDF threshold.
        #   Reading order (left→right, top→bottom): core → edge.
        #   Top-right inset in each panel shows probe location on sim histogram.
        # ══════════════════════════════════════════════════════════════════════
        density_probes       = results.get('density_probes')
        density_probe_curves = results.get('density_probe_curves')
        log_samples_data     = results.get('log_samples')
        if not density_probes:
            print('[TableDiagnostics] Skipping density-probe figure: '
                  'log_samples.npy not found (run snapshot_histogram.py first).')
        else:
            n_panels  = len(density_probes)
            n_cols    = 6
            n_rows    = math.ceil(n_panels / n_cols)

            fig3, axes3 = plt.subplots(n_rows, n_cols,
                                       figsize=(4.5 * n_cols, 4.2 * n_rows),
                                       squeeze=False)

            # Pre-compute inset histogram in log10 space (once, reused per panel)
            if log_samples_data is not None:
                ins_h, ins_xe, ins_ye = np.histogram2d(
                    log_samples_data[:, 0], log_samples_data[:, 1], bins=40)
                ins_h_masked = np.where(ins_h > 0, ins_h, np.nan)
                ins_vmin = float(np.nanmin(ins_h_masked))
                ins_vmax = float(np.nanmax(ins_h_masked))
            else:
                ins_h_masked = ins_xe = ins_ye = None

            for panel_i, p in enumerate(density_probes):
                row_i = panel_i // n_cols
                col_i = panel_i % n_cols
                ax    = axes3[row_i, col_i]
                key   = (p['nH'], p['col'])

                for sp_cfg in SPECIES_CFG:
                    sp   = sp_cfg['name']
                    lums = density_probe_curves[key][sp]
                    ax.plot(T_vals, lums, color=sp_cfg['color'], lw=1.4,
                            drawstyle='steps-mid', label=sp)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_title(
                    f'top {p["rank_pct"]}%  nH={p["nH"]:.1e}  N_H={p["col"]:.1e}',
                    fontsize=7)
                ax.grid(True, alpha=0.2, ls='--', lw=0.3)
                if col_i == 0:
                    ax.set_ylabel('lumPerH [erg/s/H]', fontsize=8)
                if row_i == n_rows - 1 or panel_i == n_panels - 1:
                    ax.set_xlabel('T [K]', fontsize=8)
                if row_i == 0 and col_i == n_cols - 1:
                    ax.legend(fontsize=7, loc='lower left', framealpha=0.7)

                # ── Inset: sim histogram + probe marker ───────────────────
                if ins_h_masked is not None:
                    ax_ins = ax.inset_axes([0.60, 0.56, 0.38, 0.40])
                    ax_ins.set_alpha(0.7)
                    ax_ins.set_facecolor('none')  # 透明背景
                    ax_ins.patch.set_alpha(0.0)
                    ax_ins.pcolormesh(
                        ins_ye, ins_xe, ins_h_masked,
                        cmap='Blues',
                        norm=LogNorm(vmin=ins_vmin, vmax=ins_vmax),
                        shading='auto',
                        alpha=0.7,
                        )
                    ax_ins.plot(np.log10(p['col']), np.log10(p['nH']),
                                'r*', ms=7, zorder=10, markeredgewidth=0.3,
                                markeredgecolor='darkred')
                    ax_ins.set_xticks([])
                    ax_ins.set_yticks([])
                    for spine in ax_ins.spines.values():
                        spine.set_linewidth(0.5)
                        spine.set_alpha(0.4)   

            # Hide any unused panels
            for panel_i in range(n_panels, n_rows * n_cols):
                axes3[panel_i // n_cols, panel_i % n_cols].set_visible(False)

            fig3.suptitle(
                'DESPOTIC Table @ sim density-CDF probes  (core → edge)',
                fontsize=13)
            plt.tight_layout()
            out3 = self.config.output_dir / 'table_diagnostics_density_probes.png'
            plt.savefig(str(out3), dpi=150, bbox_inches='tight')
            plt.close(fig3)
            print(f'Saved: {out3}')
