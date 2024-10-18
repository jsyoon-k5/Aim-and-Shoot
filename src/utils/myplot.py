import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns
import os
import numpy as np
from box import Box
from sympy import Q

from ..utils.mymath import get_r_squared


def figure_grid(n_row, n_col, size=None, size_ax=None):
    if size is None and size_ax is not None:
        size = np.array([n_col, n_row]) * size_ax
    fig, axs = plt.subplots(n_row, n_col, figsize=size, constrained_layout=True)
    return fig, axs


def figure_save(fig:Figure, path, dpi=100, save_svg=False, pad_inches=0.01):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    if save_svg:
        root, _ = os.path.splitext(path)
        path_svg = f"{root}.pdf"
        fig.savefig(path_svg, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    plt.close(fig)


def draw_r2_plot(ax:Axes, xdata, ydata, **kwargs):
    """
    kwargs
        edge_margin: float. Decides the edge margin of the plot
        full_fit_label: bool. Decides whether to include the equation in the label
        
        base_lw: y=x line's linewidth
        base_ls: y=x line's linestyle
        base_lc: y=x line's linecolor
    """
    settings = Box(dict(
        edge_margin = 0.1,
        full_fit_label = False,

        reg_lw = 2.5,
        reg_lc = 'red',
        reg_mc = 'black',
        reg_ma = 0.3,
        reg_ms = 5,

        base_lw = 1,
        base_ls = '--',
        base_lc = 'gray',

        xlabel = "X value",
        ylabel = "Y value",
        lb_fontsize = 10,

        legend_fontsize = 8,
        legend_pos = "lower right",

        grid_lw = 0.5,
        grid_lc = 'gray',
        grid_la = 0.5,
        grid_ls = '--',
    ))
    settings.update(kwargs)

    rsq_value = get_r_squared(xdata, ydata)

    max_val = max(max(xdata), max(ydata))
    min_val = min(min(xdata), min(ydata))
    dist_val = (max_val - min_val) * settings.edge_margin

    y_fit = np.polyfit(xdata, ydata, 1)
    if settings.full_fit_label:
        label = f"$({y_fit[0]:.2f})x + ({y_fit[1]:.2f}), R^2={rsq_value:.2f}$"
    else:
        label = f"$R^2={rsq_value:.2f}$"
    
    sns.regplot(
        x=xdata,
        y=ydata,
        scatter_kws={"color": settings.reg_mc, "alpha": settings.reg_ma, "s": settings.reg_ms},
        line_kws={"color": settings.reg_lc, "lw": settings.reg_lw},
        ax=ax
    )
    ax.plot(
        [min_val - dist_val, max_val + dist_val],
        [min_val - dist_val, max_val + dist_val],
        linewidth=1 if "base_lw" not in kwargs else kwargs["base_lw"],
        color="gray" if "base_lc" not in kwargs else kwargs["base_lc"],
        linestyle="--" if "base_ls" not in kwargs else kwargs["base_ls"],
    )
    ax.set_xlabel(settings.xlabel, fontsize=settings.lb_fontsize)
    ax.set_ylabel(settings.ylabel, fontsize=settings.lb_fontsize)
    ax.set_xlim([min_val - dist_val, max_val + dist_val])
    ax.set_ylim([min_val - dist_val, max_val + dist_val])
    ax.legend([
        Line2D([0], [0], color=settings.reg_mc, lw=settings.reg_lw)], 
        [label,], 
        fontsize=settings.legend_fontsize, 
        loc=settings.legend_pos
    )
    ax.grid(linestyle=settings.grid_ls, linewidth=settings.grid_lw, color=settings.grid_lc, alpha=settings.grid_la)
    ax.set_aspect("equal")
