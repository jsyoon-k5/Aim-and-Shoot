import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns
import os
import numpy as np

# from ..utils.mymath import get_r_squared


def figure_grid(n_row, n_col, size=None, size_ax=None):
    if size is None and size_ax is not None:
        size = tuple(np.array([n_col, n_row]) * np.asarray(size_ax))
    fig, axes = plt.subplots(n_row, n_col, figsize=size, constrained_layout=True, squeeze=False)
    return fig, axes


def figure_save(fig: Figure, path, dpi=150, save_svg=False, pad_inches=0.01):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    if save_svg:
        root, _ = os.path.splitext(path)
        path_pdf = f"{root}.pdf"
        fig.savefig(path_pdf, dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)

    plt.close(fig)