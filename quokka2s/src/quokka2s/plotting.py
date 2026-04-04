import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, LogFormatterMathtext, MaxNLocator, LogLocator
import yt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import collections



def create_plot(data_2d: np.ndarray,
                title: str,
                cbar_label: str,
                filename: str,
                extent: Optional[list] = None,
                xlabel: str = None,
                ylabel: str = None,
                norm: Optional[LogNorm] = None,
                camp: str = 'viridis',
                use_scientific_formatter: bool = True
                ):
    """
    Generates and saves a 2D plot from a NumPy array.
    Parameters:
    - data_2d: 2D NumPy array to plot.  
    - title: Title of the plot.
    - cbar_label: Label for the color bar.
    - filename: Name of the file to save the plot.
    - extent: List defining the physical extent of the plot [Horizontal_min, Horizental_max, Vertical_min, Vertical_max].
    - xlabel: Label for the horizontal axis.
    - ylabel: Label for the vertical axis.
    - norm: Normalization for color scaling (e.g., LogNorm).
    - camp: Colormap to use for the plot.
    - use_scientific_formatter: Whether to use scientific notation for color bar ticks.
    """
    print(f"Plotting'{filename}'... Data Range: min={np.min(data_2d):.2e}, max={np.max(data_2d):.2e}")
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(data_2d, 
                   origin='lower',
                   extent=extent,
                   aspect='equal',
                   cmap=camp,
                   norm=norm)

    formatter = None
    if use_scientific_formatter:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,0))
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(filename, dpi=800)
    plt.show()
    plt.close(fig)
    print("="*40)
    print(f"Plot saved as {filename}")
    print("="*40)
    return fig, ax


def plot_multiview_grid(plots_info: List[Dict],
                        extent_top: List[float],
                        filename: str,
                        extent_bottom: Optional[List[float]] = None,
                        top_ylabel: str = "z",
                        top_xlabel: Optional[str] = None,
                        bottom_ylabel: str = "x",
                        bottom_xlabel: Optional[str] = None,
                        units: str = 'pc',
                        particles_top: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        particles_bottom: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                        particle_style: Optional[Dict] = None,
                        particle_stride: int = 1,
                        show_particles_top: bool = True,
                        show_particles_bottom: bool = True,
                        include_bottom: bool = True,
                        column_spacing: float = 0.18,
                        row_spacing: float = 0.04,
                        colorbar_height_ratio: float = 0.05
                        ):
    """
    Generates and saves multi-plots.
    Parameters: 
    - plots_info: A list of dictionaries, where each dictionary contains the info
                  for one column (required keys: 'data_top', 'label'; optional keys: 'norm', 'cmap',
                  'data_bottom', 'vector_field_top', 'vector_field_bottom').
    - extent_top: The physical extent for the top row of plots [Horizontal_min, Horizental_max, Vertical_min, Vertical_max].
    - filename: Name of the file to save the entire figure.
    - extent_bottom: The physical extent for the bottom row of plots [Horizontal_min, Horizental_max, Vertical_min, Vertical_max].
                     Optional when not drawing the bottom row.
    - top_ylabel: The y-axis label for the top row of plots.
    - top_xlabel: The x-axis label for the top row of plots. Optional.
    - bottom_ylabel: The y-axis label for the bottom row of plots.
    - bottom_xlabel: The x-axis label for the bottom row of plots. Optional.
    - units: The unit string to append to the axis labels.

    - particles_top: Optional tuple (px, py) of particle coordinates for the top row.
    - particles_bottom: Optional tuple (px, py) of particle coordinates for the bottom row.
                       Ignored when include_bottom=False.
    - particle_style: Optional dictionary to style the particles (e.g., {'s': 1, 'c': 'red'}).
    - particle_stride: Plot every Nth particle to declutter the scatter overlay.
    - show_particles_top / show_particles_bottom: Toggle scatter overlays per row.
    - include_bottom: Set to False to render only the top row (plus color bars).
    """
    bottom_data_present = any(info.get('data_bottom') is not None for info in plots_info)
    has_bottom_row = include_bottom and bottom_data_present
    if include_bottom and bottom_data_present and extent_bottom is None:
        raise ValueError("extent_bottom must be provided when include_bottom=True.")

    N_COLS = len(plots_info)
    fig_height = 7.5 if has_bottom_row else 5.5
    fig = plt.figure(figsize=(2 * N_COLS + 1.2, fig_height), layout="constrained")

    top_aspect = (extent_top[3] - extent_top[2]) / (extent_top[1] - extent_top[0])
    if has_bottom_row:
        bottom_aspect = (extent_bottom[3] - extent_bottom[2]) / (extent_bottom[1] - extent_bottom[0])
        height_ratios = [top_aspect, bottom_aspect, colorbar_height_ratio]
        n_rows = 3
    else:
        height_ratios = [top_aspect, colorbar_height_ratio]
        n_rows = 2
    
    gs = gridspec.GridSpec(n_rows, N_COLS,
                           figure=fig,
                           height_ratios=height_ratios,
                           hspace=row_spacing, wspace=column_spacing)

    top_axes = [fig.add_subplot(gs[0, i]) for i in range(N_COLS)]
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(N_COLS)] if has_bottom_row else []
    cbar_row_index = 2 if has_bottom_row else 1
    
    # --- Define a default style for the particles ---
    if particle_style is None:
        particle_style = {'s': 1, 'c': 'red', 'alpha': 0.7, 'marker': '.'}
    if particle_stride < 1:
        particle_stride = 1

    bottom_active_flags: List[bool] = []

    for i, info in enumerate(plots_info):
        ax_top = top_axes[i]
        ax_bottom = bottom_axes[i] if has_bottom_row else None
        norm = info.get('norm')
        cmap = info.get('cmap', 'viridis')

        data_top = info.get('data_top')
        if data_top is None:
            raise KeyError("Each plots_info entry must include 'data_top'.")

        im = ax_top.imshow(data_top, origin='lower', extent=extent_top,
                           aspect='equal', cmap=cmap, norm=norm)
        
        if has_bottom_row:
            data_bottom = info.get('data_bottom')
            if data_bottom is not None:
                ax_bottom.imshow(data_bottom, origin='lower', extent=extent_bottom, 
                                 aspect='equal', cmap=cmap, norm=info['norm'])
                bottom_active = True
            else:
                ax_bottom.axis('off')
                bottom_active = False
        else:
            bottom_active = False
        bottom_active_flags.append(bottom_active)

        if particles_top is not None and show_particles_top:
            px_t, py_t = particles_top
            if particle_stride > 1:
                px_t = px_t[::particle_stride]
                py_t = py_t[::particle_stride]
            ax_top.scatter(px_t, py_t, **particle_style)
        
        if has_bottom_row and bottom_active and particles_bottom is not None and show_particles_bottom:
            px_b, py_b = particles_bottom
            if particle_stride > 1:
                px_b = px_b[::particle_stride]
                py_b = py_b[::particle_stride]
            ax_bottom.scatter(px_b, py_b, **particle_style)


        quiver_style = {
            'color': 'black',
            'scale': 20000,          # 尝试减小这个值让箭头变长 (例如从 1000 减到 500)
            'scale_units': 'xy',   # 使用坐标轴单位来衡量箭头长度
            'width': 0.005,        # 增加这个值让箭头变粗
            'headwidth': 3,        # 箭头宽度
            'headlength': 5       # 箭头长度
        }
        
        if 'vector_field_top' in info and info['vector_field_top'] is not None:
            X, Y, U, V = info['vector_field_top']
            ax_top.quiver(X, Y, U, V, **quiver_style)
            
        if has_bottom_row and bottom_active and 'vector_field_bottom' in info and info['vector_field_bottom'] is not None:
            X, Y, U, V = info['vector_field_bottom']
            ax_bottom.quiver(X, Y, U, V, **quiver_style)

        label = info.get('label')
        if label is None:
            raise KeyError("Each plots_info entry must include 'label'.")

        ax_cbar = fig.add_subplot(gs[cbar_row_index, i])

        formatter = None
  
        if isinstance(norm, LogNorm):

            formatter = LogFormatterMathtext()
        else:

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True) # Use scientific notation
            formatter.set_powerlimits((0,1))
            formatter.set_useOffset(False) 
            

        cbar = fig.colorbar(im, cax=ax_cbar, orientation='horizontal')
        if isinstance(norm, LogNorm):
            cbar.ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        else:
            cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        cbar.ax.xaxis.set_major_formatter(formatter)
        ax_cbar.set_title(label, fontsize=8, y=0.85)
        ax_cbar.tick_params(axis='x', labelsize=7, pad=1)

        title = info.get('title')
        if title:
            ax_top.set_title(title, fontsize=10, wrap=True)
            if has_bottom_row and bottom_active:
                ax_bottom.set_title(title, fontsize=10, wrap=True)


    if has_bottom_row:
        for i, (ax_t, ax_b) in enumerate(zip(top_axes, bottom_axes)):
            if bottom_active_flags[i]:
                ax_t.tick_params(axis='x', labelbottom=False)
            else:
                ax_t.tick_params(axis='x', labelbottom=True)
            if i == 0:
                ax_t.set_ylabel(f"{top_ylabel} ({units})", fontsize=8)
                if bottom_active_flags[i]:
                    ax_b.set_ylabel(f"{bottom_ylabel} ({units})", fontsize=8)
            else:
                ax_t.tick_params(axis='y', labelleft=False)
                if bottom_active_flags[i]:
                    ax_b.tick_params(axis='y', labelleft=False)
    else:
        for i, ax_t in enumerate(top_axes):
            if i == 0:
                ax_t.set_ylabel(f"{top_ylabel} ({units})", fontsize=8)
            else:
                ax_t.tick_params(axis='y', labelleft=False)

    if top_xlabel is not None:
        for ax_t in top_axes:
            ax_t.set_xlabel(top_xlabel, fontsize=8)

    if has_bottom_row and bottom_xlabel is not None:
        for ax_b, active in zip(bottom_axes, bottom_active_flags):
            if active:
                ax_b.set_xlabel(bottom_xlabel, fontsize=8)

    # plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight', pad_inches=0.2)
    plt.show()



def create_horizontal_subplots(plots_info: List[Dict],
                               shared_extent: List,
                               shared_xlabel: str,
                               shared_ylabel: str,
                               filename: str):
    """
    Generates and saves a single figure with multiple horizontally arranged subplots.

    Parameters:
    - plots_info: A list of dictionaries, where each dictionary contains the
                  info for one subplot: {'data', 'title', 'cbar_label', 'norm'}.
    - shared_extent: The physical extent, shared by all subplots.
    - shared_xlabel: The x-axis label, shared by all subplots.
    - shared_ylabel: The y-axis label, shared by all subplots.
    - filename: Name of the file to save the entire figure.
    """
    num_plots = len(plots_info)
    # Create a subplot figure with 1 row and num_plots columns
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 8))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    for i, ax in enumerate(axes):
        
        info = plots_info[i]
        data_2d = info['data']
        norm = info.get('norm', None)

        print(f"Plotting subplot '{info['title']}'... Data Range: min={np.min(data_2d):.2e}, max={np.max(data_2d):.2e}")
        
        cmap = info.get('cmap', 'viridis')

        im = ax.imshow(data_2d,
                        origin='lower',
                        extent=shared_extent,
                        aspect='auto',
                        cmap=cmap,
                        norm=norm)
        
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,0))

        fig.colorbar(im, ax=ax, label=info['cbar_label'], format=formatter, pad=0.02)
        ax.set_title(info['title'])
        ax.set_xlabel(shared_xlabel)
            
    axes[0].set_ylabel(shared_ylabel)
    plt.subplots_adjust(wspace=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=800)
    plt.show()
    plt.close(fig)
    print("="*40)
    print(f"Subplots saved as {filename}")
    print("="*40)
    return fig, axes
